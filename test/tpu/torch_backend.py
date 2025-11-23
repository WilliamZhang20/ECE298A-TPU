import torch
from test_tpu import matmul, reset_dut
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
    
    if a_zero is None:
        a_zero = 0
    if b_zero is None:
        b_zero = 0
    
    # Convert to int32 for safe arithmetic
    a_i32 = a_q.to(torch.int32)
    b_i32 = b_q.to(torch.int32)
    
    # Subtract zero points
    a_centered = a_i32 - int(a_zero)
    b_centered = b_i32 - int(b_zero)

    # For uint8-style quantization (zero_point=-128), shift to signed range
    shift_amount = 0
    if a_zero == -128:
        shift_amount = 128
        a_hw = (a_centered - shift_amount).clamp(-128, 127).to(torch.int8)
    else:
        a_hw = a_centered.clamp(-128, 127).to(torch.int8)
    
    b_hw = b_centered.clamp(-128, 127).to(torch.int8)
    
    # Call TPU
    future = concurrent.futures.Future()
    async def wrapper():
        try:
            await reset_dut(dut)
            result_int32 = await matmul(dut, a_hw, b_hw, transpose=True, is_torch=True)
            future.set_result(result_int32)
        except Exception as e:
            future.set_exception(e)
    cocotb.start_soon(wrapper())
    result_int32 = future.result()
    
    # Apply correction for the shift
    if shift_amount != 0:
        correction = shift_amount * b_centered.sum(dim=1, dtype=torch.int32)
        correction = correction.view(1, -1).expand_as(result_int32)
        result_int32 = result_int32 + correction
        
    # Convert to float and scale
    result_float = result_int32.to(torch.float32)
    
    if a_scale is not None and b_scale is not None:
        scale = float(a_scale) * float(b_scale)
        out = result_float * scale
    else:
        out = result_float
    
    # Add bias
    if bias is not None:
        out = out + bias
    return out

@register_fake("tpu::matmul")
def tpu_matmul_abstract(a: Tensor, b: Tensor, bias: Optional[Tensor] = None, *args, **kwargs) -> Tensor:
    # a: (..., K)  where K is input features
    # b: (N, K)    where N is output features (weight matrix, will be transposed)
    # output: (..., N) after computing a @ b.T
    
    # b.shape[0] is N (number of output features)
    N = b.shape[0]
    
    # Output shape: same batch dims as a, but last dim is N
    out_shape = list(a.shape[:-1]) + [N]
    out = a.new_zeros(out_shape, dtype=torch.float32)
    
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

        # Replace linear operations with TPU matmul
        dequant_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default
        
        for node in list(gm.graph.nodes):
            if node.target == torch.ops.aten.linear.default:
                x_node, w_node, bias = node.args

                def unwrap_dequant(n):
                    # If node is a dequantize_per_tensor node, return quantization params
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
                        
                        # Verify zero points are 0 for symmetric quantization
                        print(f"Replacing linear: x_zp={x_zp}, w_zp={w_zp}")
                        
                        new_node = gm.graph.call_function(
                            torch.ops.tpu.matmul,
                            args=(q_x, q_w, bias, x_scale, w_scale, x_zp, w_zp),
                        )
                        node.replace_all_uses_with(new_node)
                        gm.graph.erase_node(node)
                        
                        # Clean up unused dequantize nodes
                        if len(x_deq_node.users) == 0:
                            gm.graph.erase_node(x_deq_node)
                        if len(w_deq_node.users) == 0:
                            gm.graph.erase_node(w_deq_node)
                    else:
                        # Fallback for non-quantized tensors
                        print(f"Warning: Linear layer not quantized, using fallback")
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