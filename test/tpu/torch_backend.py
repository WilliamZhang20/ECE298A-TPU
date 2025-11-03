import torch
from test_tpu import matmul
from torch._inductor.compile_fx import compile_fx
from torch._dynamo import register_backend
from torch.library import custom_op, register_fake
from torch import Tensor
from typing import Optional
import asyncio
import cocotb
from cocotb.triggers import RisingEdge
import concurrent

dut = None  # Global variable to hold the DUT reference

@custom_op("tpu::matmul", mutates_args=())
def tpu_matmul(a: Tensor, b: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    a_q = a.clamp(-128, 127).to(torch.int8)
    b_q = b.clamp(-128, 127).to(torch.int8)
    future = concurrent.futures.Future()
    async def wrapper():
        try:
            result = await matmul(dut, a_q, b_q, transpose=True, is_torch=True)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
    cocotb.start_soon(wrapper())
    result = future.result()
    if bias is not None:
        result = result + bias.round().to(torch.int32)
    return result

@register_fake("tpu::matmul")
def tpu_matmul_abstract(a: Tensor, b: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    M, N = a.shape[-1], b.shape[-2]
    out = a.new_zeros(a.shape[:-1] + (N,))
    if bias is not None:
        out = out + bias
    return out

def make_backend(dut_arg):
    global dut
    dut = dut_arg  # Set global DUT

    @register_backend(name=f"tpu_net")
    def _backend(gm: torch.fx.GraphModule, example_inputs):
        print("\n=== FX graph received ===")
        gm.graph.print_tabular()

        # ---- replace every linear ----
        for node in list(gm.graph.nodes):
            if node.target == torch.ops.aten.linear.default:
                x, weight, bias = node.args
                with gm.graph.inserting_before(node):
                    new_node = gm.graph.call_function(
                        torch.ops.tpu.matmul,  # ← This is the key
                        args=(x, weight, bias),
                    )
                    node.replace_all_uses_with(new_node)
                    gm.graph.erase_node(node)

        gm.recompile()
        print("\n=== Modified graph ===")
        gm.graph.print_tabular()

        # Let Inductor compile the rest
        return compile_fx(gm, example_inputs)

    return _backend