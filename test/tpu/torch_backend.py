# backend.py
import torch
import torch._dynamo
from test_tpu import matmul
from torch._inductor.compile_fx import compile_fx
from torch._dynamo import register_backend
import numpy as np
import asyncio

async def dut_matmul_async(dut, a: torch.Tensor, b: torch.Tensor, bias=None):
    a_q = a.clamp(-128, 127).to(torch.int8).cpu().numpy()
    b_q = b.clamp(-128, 127).to(torch.int8).cpu().numpy()

    c = await matmul(dut, a_q, b_q)
    if bias is not None:
        c = c + bias
    return c

def dut_matmul_sync(dut, a, b, bias=None):
    """Synchronous wrapper – torch.compile expects a normal function."""
    return asyncio.run(dut_matmul_async(dut, a, b, bias))

def make_backend(dut):
    """
    Returns a *registered* backend that has the DUT baked in.
    The FX graph is the first argument.
    """
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
                        dut_matmul_sync,
                        args=(dut, x, weight.t(), bias),
                    )
                    node.replace_all_uses_with(new_node)
                    gm.graph.erase_op(node)

        gm.recompile()
        print("\n=== Modified graph ===")
        gm.graph.print_tabular()

        # Let Inductor compile the rest → returns the *compiled dot*
        return compile_fx(gm, example_inputs)

    return _backend