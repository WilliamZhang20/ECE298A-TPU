# backend.py
import torch
import torch._dynamo
from test_tpu import matmul
from torch._inductor.compile_fx import compile_fx
from torch._dynamo import register_backend, disable
import numpy as np
import asyncio

def is_quantized(tensor):
    return tensor.dtype == torch.int8 and hasattr(tensor, 'qscheme') or tensor.is_quantized

def dequantize_if_needed(tensor):
    if is_quantized(tensor):
        return tensor.dequantize()
    return tensor

def quantize_int8(tensor):
    # Simple symmetric per-tensor int8 quantization
    scale = 127.0 / tensor.abs().max()
    return torch.quantize_per_tensor(tensor, scale, 0, torch.qint8)

@disable
def dut_matmul_sync(dut, a, b, bias=None):
    """
    Handles both float and quantized int8 inputs.
    """
    # Step 1: Dequantize inputs if they are quantized
    if is_quantized(a):
        a_float = a.dequantize()
    else:
        a_float = a

    if is_quantized(b):
        b_float = b.dequantize()
    else:
        b_float = b

    # Step 2: Re-quantize to int8 for hardware (symmetric)
    a_q = quantize_int8(a_float)
    b_q = quantize_int8(b_float)

    # Step 3: Convert to numpy int8 for cocotb (or keep as torch)
    a_np = a_q.int_repr().numpy().astype(np.int8)
    b_np = b_q.int_repr().numpy().astype(np.int8)

    # Step 4: Run hardware matmul
    loop = asyncio.get_event_loop()
    c_int32 = loop.run_until_complete(
        matmul(dut, a_np, b_np, transpose=True, is_torch=False)
    )
    c = torch.from_numpy(c_int32)

    # Step 5: Add bias (if present and float)
    if bias is not None:
        if is_quantized(bias):
            bias = bias.dequantize()
        # Scale bias by input_scale * weight_scale
        scale_a = a_q.q_scale()
        scale_b = b_q.q_scale()
        bias_scaled = bias * scale_a * scale_b
        c = c.to(torch.float32) + bias_scaled.round()
        c = c.to(torch.int32)

    return c

from torch.ao.quantization.fx._decomposed import quantize_per_tensor, dequantize_per_tensor

def make_backend(dut):
    @register_backend(name="tpu_net")
    def _backend(gm: torch.fx.GraphModule, example_inputs):
        print("\n=== FX graph received ===")
        gm.graph.print_tabular()

        count = 0
        for node in list(gm.graph.nodes):
            # Match dequantize -> mm -> quantize pattern
            if (node.op == "call_function" and
                node.target in [torch.ops.aten.mm.default, torch.ops.aten.addmm.default] and
                len(node.args) >= 2):

                input_a = node.args[0]
                input_b = node.args[1]
                bias = node.args[2] if len(node.args) > 2 else None

                # Find parent dequantize nodes
                def get_dequantized_parent(n):
                    if n.op == "call_function" and n.target == dequantize_per_tensor:
                        return n.args[0]  # the int8 tensor
                    return None

                a_q = get_dequantized_parent(input_a)
                b_q = get_dequantized_parent(input_b)

                if a_q is not None and b_q is not None:
                    with gm.graph.inserting_before(node):
                        new_node = gm.graph.call_function(
                            dut_matmul_sync,
                            args=(dut, a_q, b_q, bias),
                        )
                        node.replace_all_uses_with(new_node)
                        gm.graph.eliminate_dead_code()
                        count += 1

        print(f"Replaced {count} matmul(s) with DUT")
        gm.recompile()
        print("\n=== Modified graph ===")
        gm.graph.print_tabular()

        return compile_fx(gm, example_inputs)

    return _backend