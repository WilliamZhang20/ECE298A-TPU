"""Kernel entry points lowered through the TPU matmul test harness."""

from typing import Optional, Union

import numpy as np

from test_tpu import matmul


Array = Union[np.ndarray, "torch.Tensor"]


async def conv1d(
    dut,
    input: Array,
    weight: Array,
    bias: Optional[Array] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
):
    """Run a 1-D convolution using 2-D tiles submitted to ``test_tpu.matmul``.

    Expected layouts follow PyTorch: input ``(N, C_in, L)`` and weight
    ``(C_out, C_in / groups, kernel_size)``.
    """
    

    raise NotImplementedError("conv1d lowering is not implemented yet")


async def conv2d(
    dut,
    input: Array,
    weight: Array,
    bias: Optional[Array] = None,
    stride: Union[int, tuple] = 1,
    padding: Union[int, tuple] = 0,
    dilation: Union[int, tuple] = 1,
    groups: int = 1,
):
    """Run a 2-D convolution using the TPU matmul harness.

    Expected layouts follow PyTorch: input ``(N, C_in, H, W)`` and weight
    ``(C_out, C_in / groups, kernel_h, kernel_w)``.
    """
    
    raise NotImplementedError("conv2d lowering is not implemented yet")


async def conv3d(
    dut,
    input: Array,
    weight: Array,
    bias: Optional[Array] = None,
    stride: Union[int, tuple] = 1,
    padding: Union[int, tuple] = 0,
    dilation: Union[int, tuple] = 1,
    groups: int = 1,
):
    """Run a 3-D convolution using the TPU matmul harness.

    Expected layouts follow PyTorch: input ``(N, C_in, D, H, W)`` and weight
    ``(C_out, C_in / groups, kernel_d, kernel_h, kernel_w)``.
    """
    raise NotImplementedError("conv3d lowering is not implemented yet")


async def einsum(dut, equation: str, *operands: Array):
    """Evaluate an einsum by lowering supported contractions to matmul tiles.

    ``dut`` is passed to the same async harness used by ``test_tpu.matmul``.
    The supported equation subset and contraction lowering will be filled in
    with the implementation.
    """
    raise NotImplementedError("einsum lowering is not implemented yet")
