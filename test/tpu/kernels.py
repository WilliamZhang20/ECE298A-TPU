"""Kernel entry points lowered through the TPU matmul test harness."""

from math import prod
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from test_tpu import matmul


Array = Union[np.ndarray, "torch.Tensor"]


def _as_torch(x):
    """View a numpy array as a torch tensor; pass torch tensors through."""
    return torch.as_tensor(x) if isinstance(x, np.ndarray) else x


def _pair(v):
    """Normalize an int-or-(H, W) argument to an ``(H, W)`` tuple."""
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _triple(v):
    """Normalize an int-or-(D, H, W) argument to a ``(D, H, W)`` tuple."""
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1]), int(v[2])
    return int(v), int(v), int(v)


def _hw_int8(t):
    """Saturate to the int8 range carried by the hardware matmul tiles."""
    return t.to(torch.int32).clamp(-128, 127).to(torch.int8)


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
    was_numpy = isinstance(input, np.ndarray)
    x = _as_torch(input)
    w = _as_torch(weight)
    b = _as_torch(bias)

    N, C_in, L = x.shape
    C_out, C_in_g, K = w.shape
    C_out_g = C_out // groups

    L_out = (L + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    x_padded = F.pad(x, (padding, padding))

    out = torch.zeros((N, C_out, L_out), dtype=torch.int32)
    for j in range(groups):
        # Fuse the whole batch into one cols matrix, samples side by side:
        # cols[c*K + t, i*L_out + l_out] = x_padded[i, j*C_in_g + c, l_out*stride + t*dilation]
        # (row order matches W_g's flattened (channel, tap) columns).
        cols = torch.cat([
            torch.stack([
                x_padded[i, j * C_in_g + c, t * dilation :: stride][:L_out]
                for c in range(C_in_g)
                for t in range(K)
            ])                        # (C_in_g*K, L_out): one sample
            for i in range(N)
        ], dim=1)                     # (C_in_g*K, N*L_out): all samples
        W_g = w[j * C_out_g:(j + 1) * C_out_g].reshape(C_out_g, -1)
        Y = await matmul(dut, W_g, cols)  # (C_out_g, N*L_out), int32
        # Sample i lives in column block [i*L_out, (i+1)*L_out); unpack to (N, C_out_g, L_out).
        out[:, j * C_out_g:(j + 1) * C_out_g] = Y.reshape(C_out_g, N, L_out).permute(1, 0, 2)

    if b is not None:
        out += b.to(torch.int32).reshape(1, -1, 1)

    return out.numpy() if was_numpy else out


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
    """Run a 2-D convolution using 2-D tiles submitted to ``test_tpu.matmul``.

    Expected layouts follow PyTorch: input ``(N, C_in, H, W)`` and weight
    ``(C_out, C_in / groups, kernel_h, kernel_w)``.  ``stride``, ``padding``
    and ``dilation`` may be ints or ``(H, W)`` pairs.
    """
    was_numpy = isinstance(input, np.ndarray)
    x = _as_torch(input)
    w = _as_torch(weight)
    b = _as_torch(bias)

    sH, sW = _pair(stride)
    pH, pW = _pair(padding)
    dH, dW = _pair(dilation)

    N, C_in, H, W = x.shape
    C_out, C_in_g, kH, kW = w.shape
    C_out_g = C_out // groups

    H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    x_padded = F.pad(x, (pW, pW, pH, pH))

    out = torch.zeros((N, C_out, H_out, W_out), dtype=torch.int32)
    for j in range(groups):
        # Fuse the whole batch into one cols matrix, output positions side by
        # side: cols[(c*kH + kh)*kW + kw, i*H_out*W_out + h_out*W_out + w_out]
        #   = x_padded[i, j*C_in_g + c, h_out*sH + kh*dH, w_out*sW + kw*dW]
        # (row order matches W_g's flattened (channel, tap) columns).
        cols = torch.cat([                     # all samples
            torch.cat([                         # one sample
                torch.stack([                   # one output row h_out
                    x_padded[i, j * C_in_g + c, h * sH + kh * dH, kw * dW :: sW][:W_out]
                    for c in range(C_in_g)
                    for kh in range(kH)
                    for kw in range(kW)
                ])                              # (C_in_g*kH*kW, W_out)
                for h in range(H_out)
            ], dim=1)                           # (C_in_g*kH*kW, H_out*W_out)
            for i in range(N)
        ], dim=1)                               # (C_in_g*kH*kW, N*H_out*W_out)
        W_g = w[j * C_out_g:(j + 1) * C_out_g].reshape(C_out_g, -1)
        Y = await matmul(dut, W_g, cols)        # (C_out_g, N*H_out*W_out), int32
        # Sample i lives in column block [i*H_out*W_out, (i+1)*H_out*W_out);
        # unpack to (N, C_out_g, H_out, W_out).
        out[:, j * C_out_g:(j + 1) * C_out_g] = (
            Y.reshape(C_out_g, N, H_out, W_out).permute(1, 0, 2, 3)
        )

    if b is not None:
        out += b.to(torch.int32).reshape(1, -1, 1, 1)

    return out.numpy() if was_numpy else out


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
    """Run a 3-D convolution using 2-D tiles submitted to ``test_tpu.matmul``.

    Expected layouts follow PyTorch: input ``(N, C_in, D, H, W)`` and weight
    ``(C_out, C_in / groups, kernel_d, kernel_h, kernel_w)``.  ``stride``,
    ``padding`` and ``dilation`` may be ints or ``(D, H, W)`` triples.
    """
    was_numpy = isinstance(input, np.ndarray)
    x = _as_torch(input)
    w = _as_torch(weight)
    b = _as_torch(bias)

    sD, sH, sW = _triple(stride)
    pD, pH, pW = _triple(padding)
    dD, dH, dW = _triple(dilation)

    N, C_in, D, H, W = x.shape
    C_out, C_in_g, kD, kH, kW = w.shape
    C_out_g = C_out // groups

    D_out = (D + 2 * pD - dD * (kD - 1) - 1) // sD + 1
    H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    x_padded = F.pad(x, (pW, pW, pH, pH, pD, pD))

    out = torch.zeros((N, C_out, D_out, H_out, W_out), dtype=torch.int32)
    for j in range(groups):
        # Fuse the whole batch into one cols matrix, output positions side by
        # side (rows flattened as (c, kd, kh, kw), columns as (i, d, h, w)):
        #   cols[.., i*D_out*H_out*W_out + d_out*H_out*W_out + h_out*W_out + w_out]
        #   = x_padded[i, j*C_in_g + c, d_out*sD + kd*dD, h_out*sH + kh*dH,
        #              w_out*sW + kw*dW]
        # (row order matches W_g's flattened (channel, tap) columns).
        cols = torch.cat([                     # all samples
            torch.cat([                         # one sample
                torch.cat([                     # one output plane d_out
                    torch.stack([               # one output row h_out
                        x_padded[i, j * C_in_g + c, d * sD + kd * dD,
                                 h * sH + kh * dH, kw * dW :: sW][:W_out]
                        for c in range(C_in_g)
                        for kd in range(kD)
                        for kh in range(kH)
                        for kw in range(kW)
                    ])                          # (C_in_g*kD*kH*kW, W_out)
                    for h in range(H_out)
                ], dim=1)                       # (C_in_g*kD*kH*kW, H_out*W_out)
                for d in range(D_out)
            ], dim=1)                           # (C_in_g*kD*kH*kW, D_out*H_out*W_out)
            for i in range(N)
        ], dim=1)                               # (C_in_g*kD*kH*kW, N*D_out*H_out*W_out)
        W_g = w[j * C_out_g:(j + 1) * C_out_g].reshape(C_out_g, -1)
        Y = await matmul(dut, W_g, cols)        # (C_out_g, N*D_out*H_out*W_out), int32
        # Sample i lives in column block [i*D_out*H_out*W_out, ...); unpack to
        # (N, C_out_g, D_out, H_out, W_out).
        out[:, j * C_out_g:(j + 1) * C_out_g] = (
            Y.reshape(C_out_g, N, D_out, H_out, W_out).permute(1, 0, 2, 3, 4)
        )

    if b is not None:
        out += b.to(torch.int32).reshape(1, -1, 1, 1, 1)

    return out.numpy() if was_numpy else out


async def einsum(dut, equation: str, *operands: Array):
    """Evaluate an einsum by lowering supported contractions to matmul tiles.

    ``dut`` is passed to the same async harness used by ``test_tpu.matmul``.

    Supported equations use an explicit ``->`` output, no ellipsis and no
    repeated subscripts inside a single operand.  Terms are contracted left
    to right: shared subscripts that neither the output nor a remaining term
    needs become the matmul K axis, shared subscripts that are still needed
    become batched matmuls (one tile pair per batch index), unshared
    subscripts become the M/N axes (an empty K group lowers to an
    outer-product tile), and subscripts the output never mentions are summed
    out with a ones-column matmul tile.
    """
    eq = equation.replace(" ", "")
    if "..." in eq:
        raise NotImplementedError("ellipsis subscripts are not supported")
    if eq.count("->") != 1:
        raise ValueError("equation must contain exactly one '->'")
    lhs, out_labels = eq.split("->")
    input_labels = lhs.split(",") if lhs else [""]

    if len(input_labels) != len(operands):
        raise ValueError("Number of subscripts must match operands")
    if len(set(out_labels)) != len(out_labels):
        raise ValueError("output repeats a subscript")
    for labels in input_labels:
        if len(set(labels)) != len(labels):
            raise NotImplementedError(
                f"repeated subscripts in '{labels}' (diagonals) are not supported"
            )

    was_numpy = all(isinstance(op, np.ndarray) for op in operands)
    tensors = [_as_torch(op).to(torch.int32) for op in operands]

    # One size per label, consistent across every operand.
    sizes = {}
    for labels, t in zip(input_labels, tensors):
        if len(labels) != t.dim():
            raise ValueError(f"subscript '{labels}' does not match operand rank {t.dim()}")
        for label, size in zip(labels, t.shape):
            if sizes.setdefault(label, size) != size:
                raise ValueError(f"subscript '{label}' has mismatched operand sizes")
    for label in out_labels:
        if label not in sizes:
            raise ValueError(f"output subscript '{label}' is missing from the operands")

    terms = [
        [labels, tensor]
        for labels, tensor in zip(input_labels, tensors)
    ]

    while len(terms) > 1:
        lhs_labels, lhs_tensor = terms[0]
        rhs_labels, rhs_tensor = terms[1]
        shared = set(lhs_labels) & set(rhs_labels)

        # 2. Decide which shared labels can be contracted.
        # A label must not be needed by the output or remaining terms.
        needed_later = set(out_labels)
        for labels, _ in terms[2:]:
            needed_later.update(labels)
        contract = shared - needed_later

        # 3. Arrange dimensions into M, K, N groups.  Shared labels that are
        # still needed become batch dims (same order on both sides), unshared
        # lhs/rhs labels become the M/N dims, and contracted labels are the
        # K dims.  An empty group is a size-1 axis, so a pair with nothing to
        # contract lowers to an outer-product tile.
        batch = [l for l in lhs_labels if l in shared and l not in contract]
        m_labels = [l for l in lhs_labels if l not in shared]
        k_labels = [l for l in lhs_labels if l in contract]
        n_labels = [l for l in rhs_labels if l not in shared]

        # 4. Transpose and reshape each tensor into stacked (M, K)/(K, N)
        # matrices, one tile pair per batch index.
        A = lhs_tensor.permute([lhs_labels.index(l) for l in batch + m_labels + k_labels])
        B = rhs_tensor.permute([rhs_labels.index(l) for l in batch + k_labels + n_labels])
        b = prod(sizes[l] for l in batch)
        m = prod(sizes[l] for l in m_labels)
        k = prod(sizes[l] for l in k_labels)
        n = prod(sizes[l] for l in n_labels)
        A = A.reshape(b, m, k)
        B = B.reshape(b, k, n)

        # 5. Call the matmul implementation (tiles saturate to int8).
        blocks = [await matmul(dut, _hw_int8(A[i]), _hw_int8(B[i])) for i in range(b)]

        # 6. Reshape the result and assign its labels.
        new_labels = "".join(batch + m_labels + n_labels)
        new_tensor = torch.stack(blocks).reshape([sizes[l] for l in new_labels])

        # 7. Replace the two terms with the new term.
        terms = [[new_labels, new_tensor]] + terms[2:]

    # Subscripts the output never asked for are summed out with a
    # ones-column matmul tile (x @ 1 == sum(x)).
    labels, out = terms[0]
    out_set = set(out_labels)
    keep = [l for l in labels if l in out_set]
    drop = [l for l in labels if l not in out_set]
    if drop:
        A = out.permute([labels.index(l) for l in keep + drop]).reshape(
            prod(sizes[l] for l in keep), prod(sizes[l] for l in drop)
        )
        ones = torch.ones((prod(sizes[l] for l in drop), 1), dtype=torch.int8)
        out = (await matmul(dut, _hw_int8(A), ones)).reshape([sizes[l] for l in keep])
        labels = "".join(keep)

    # Relabel into the requested output order.
    if out_labels and labels != out_labels:
        out = out.permute([labels.index(l) for l in out_labels])

    return out.numpy() if was_numpy else out
