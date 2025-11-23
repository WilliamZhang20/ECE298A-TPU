import torch
from test_tpu import matmul
from torch._inductor.compile_fx import compile_fx
from torch._dynamo import register_backend
from torch.library import custom_op, register_fake
from torch import Tensor
from typing import Optional
from cocotb.triggers import RisingEdge
import asyncio
import cocotb
import concurrent
import torch.fx

dut = None  # Global variable to hold the DUT reference

@custom_op("tpu::matmul", mutates_args=())
def tpu_matmul(a_q: Tensor, b_q: Tensor, bias: Optional[Tensor] = None,
               a_scale: Optional[float] = None, b_scale: Optional[float] = None,
               a_zero: Optional[int] = None, b_zero: Optional[int] = None) -> Tensor:
    # Prepare centered int8 inputs for the TPU:
    # 1) convert to int32 so subtraction is safe
    # 2) subtract zero_point to center around 0
    # 3) clamp to int8 range and cast back to int8 for the hardware
    a_i32 = a_q.to(torch.int32)
    b_i32 = b_q.to(torch.int32)

    if a_zero is not None:
        a_center = a_i32 - int(a_zero)
    else:
        a_center = a_i32
    if b_zero is not None:
        b_center = b_i32 - int(b_zero)
    else:
        b_center = b_i32

    a_hw = a_center.clamp(-128, 127).to(torch.int8)
    b_hw = b_center.clamp(-128, 127).to(torch.int8)

    # Call TPU: provide int8 inputs; TPU returns int32 accumulators
    future = concurrent.futures.Future()
    async def wrapper():
        try:
            result_int32 = await matmul(dut, a_hw, b_hw, transpose=True, is_torch=True)
            future.set_result(result_int32)
        except Exception as e:
            future.set_exception(e)
    cocotb.start_soon(wrapper())
    result_int32 = future.result()

    # Dequantize to float using scale product (accumulators are int32)
    if a_scale is None or b_scale is None:
        out = result_int32.to(torch.float32)
    else:
        scale = float(a_scale) * float(b_scale)
        out = result_int32.to(torch.float32) * scale

    if bias is not None:
        out = out + bias

    return out

@register_fake("tpu::matmul")
def tpu_matmul_abstract(a: Tensor, b: Tensor, bias: Optional[Tensor] = None, *args, **kwargs) -> Tensor:
    M, N = a.shape[-1], b.shape[-2]
    out = a.new_zeros(a.shape[:-1] + (N,), dtype=torch.float32)
    if bias is not None:
        out = out + bias
    return out

def make_backend(dut_arg):
    global dut
    dut = dut_arg  # Set global DUT

    @register_backend(name=f"tpu_net")
    def _backend(gm: torch.fx.GraphModule, example_inputs):
        # print("\n=== FX graph received ===")
        # gm.graph.print_tabular()

        # ---- replace every linear but try to preserve quantized buffers + scale info ----
        dequant_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default
        for node in list(gm.graph.nodes):
            if node.target == torch.ops.aten.linear.default:
                x_node, w_node, bias = node.args

                def unwrap_dequant(n):
                    # If node is a dequantize_per_tensor node, return (q_tensor, scale, zero_point, node)
                    if isinstance(n, torch.fx.Node) and n.target == dequant_op:
                        # args: (quantized_tensor, scale, zero_point, min, max, dtype)
                        q_tensor = n.args[0]
                        scale = n.args[1]
                        zero_point = n.args[2]
                        return (q_tensor, scale, zero_point, n)
                    return None

                x_un = unwrap_dequant(x_node)
                w_un = unwrap_dequant(w_node)

                with gm.graph.inserting_before(node):
                    if x_un and w_un:
                        q_x, x_scale, x_zp, x_deq_node = x_un
                        q_w, w_scale, w_zp, w_deq_node = w_un
                        new_node = gm.graph.call_function(
                            torch.ops.tpu.matmul,
                            args=(q_x, q_w, bias, x_scale, w_scale, x_zp, w_zp),
                        )
                        node.replace_all_uses_with(new_node)
                        gm.graph.erase_node(node)
                        # try to erase the now-unused dequantize nodes
                        try:
                            gm.graph.erase_node(x_deq_node)
                        except Exception:
                            pass
                        try:
                            gm.graph.erase_node(w_deq_node)
                        except Exception:
                            pass
                    else:
                        # Fallback: if inputs are already raw tensors/float, call tpu.matmul with them
                        new_node = gm.graph.call_function(
                            torch.ops.tpu.matmul,
                            args=(x_node, w_node, bias),
                        )
                        node.replace_all_uses_with(new_node)
                        gm.graph.erase_node(node)

        gm.recompile()
        # print("\n=== Modified graph ===")
        # gm.graph.print_tabular()

        # Let Inductor compile the rest
        return compile_fx(gm, example_inputs)

    return _backend