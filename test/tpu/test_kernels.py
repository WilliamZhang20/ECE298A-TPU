"""Validity tests for the matmul-backed kernel entry points."""

import cocotb
from cocotb.clock import Clock
import numpy as np
import torch
import torch.nn.functional as F

from kernels import conv1d, conv2d, conv3d, einsum
from test_tpu import reset_dut


async def start_dut(dut):
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    await reset_dut(dut)


def assert_matches_torch(actual, expected):
    actual_tensor = torch.as_tensor(np.asarray(actual).copy())
    torch.testing.assert_close(actual_tensor, expected, rtol=0, atol=0)


@cocotb.test()
async def test_conv1d_matches_torch(dut):
    await start_dut(dut)
    x = torch.tensor([[[1, -2, 3, 1, 2, -1], [2, 1, -1, 3, -2, 1]]], dtype=torch.int8)
    weight = torch.tensor(
        [[[1, 0, -1], [1, 1, 0]], [[-1, 1, 1], [0, 1, -1]]], dtype=torch.int8
    )
    bias = torch.tensor([2, -1], dtype=torch.int32)
    kwargs = {"stride": 2, "padding": 1, "dilation": 1, "groups": 1}

    actual = await conv1d(dut, x, weight, bias=bias, **kwargs)
    expected = F.conv1d(x.float(), weight.float(), bias.float(), **kwargs).to(torch.int32)
    assert_matches_torch(actual, expected)


@cocotb.test()
async def test_conv2d_matches_torch(dut):
    await start_dut(dut)
    x = torch.tensor(
        [[[[1, 2, -1, 0], [0, -2, 3, 1], [2, 1, 0, -1]],
          [[-1, 1, 2, 0], [2, 0, -1, 3], [1, -2, 1, 2]]]],
        dtype=torch.int8,
    )
    weight = torch.tensor(
        [[[[1, 0], [-1, 1]], [[0, 1], [1, -1]]],
         [[[-1, 1], [0, 1]], [[1, 0], [-1, 1]]]],
        dtype=torch.int8,
    )
    bias = torch.tensor([1, -2], dtype=torch.int32)
    kwargs = {"stride": (1, 2), "padding": (1, 0), "dilation": 1, "groups": 1}

    actual = await conv2d(dut, x, weight, bias=bias, **kwargs)
    expected = F.conv2d(x.float(), weight.float(), bias.float(), **kwargs).to(torch.int32)
    assert_matches_torch(actual, expected)


@cocotb.test()
async def test_conv3d_matches_torch(dut):
    await start_dut(dut)
    x = torch.arange(1, 1 + 1 * 1 * 3 * 3 * 4, dtype=torch.int8).reshape(1, 1, 3, 3, 4)
    x = (x % 7) - 3
    weight = torch.tensor(
        [[[[[1, 0], [-1, 1]], [[0, 1], [1, -1]]]],
         [[[[-1, 1], [0, 1]], [[1, 0], [-1, 1]]]]],
        dtype=torch.int8,
    )
    bias = torch.tensor([1, -1], dtype=torch.int32)
    kwargs = {"stride": 1, "padding": 0, "dilation": 1, "groups": 1}

    actual = await conv3d(dut, x, weight, bias=bias, **kwargs)
    expected = F.conv3d(x.float(), weight.float(), bias.float(), **kwargs).to(torch.int32)
    assert_matches_torch(actual, expected)


@cocotb.test()
async def test_einsum_matches_torch(dut):
    await start_dut(dut)
    a = torch.tensor([[1, -2, 3], [4, 0, -1]], dtype=torch.int8)
    b = torch.tensor([[2, 1], [-1, 3], [0, 2]], dtype=torch.int8)

    actual = await einsum(dut, "ik,kj->ij", a, b)
    expected = torch.einsum("ik,kj->ij", a.to(torch.int32), b.to(torch.int32))
    assert_matches_torch(actual, expected)
