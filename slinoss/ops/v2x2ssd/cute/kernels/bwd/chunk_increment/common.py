"""Common helpers for CuTe ``v2x2ssd`` chunk-increment backward.

This stage now follows the same package-level split as the cleaner staged
backward stacks:

- ``db`` owns the main ``dB`` contraction and value-lane transport
- ``du`` owns the main ``dU`` contraction and boundary scatter
- ``boundary`` owns the rank-1 boundary branch
- ``param_scan`` owns the scalar-heavy parameter scan-backward

All four consume the same explicit backward context prepared here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_increment import (
    _complex_dtype_from_real,
    _pad_time_partial,
    _prepared_suffix_products,
    _resolve_dtypes,
    _resolve_prev0,
    _to_complex_scalar,
    _to_complex_taps,
    _validate_chunk_increment_inputs,
)
from slinoss.ops.v2x2ssd.reference import _as_complex_pairs


@dataclass(frozen=True)
class ChunkIncrementBwdContext:
    batch_size: int
    n_heads: int
    T: int
    T_pad: int
    n_chunks: int
    L: int
    N: int
    P: int
    D: int
    BHC: int
    rdtype: torch.dtype
    cplx_dtype: torch.dtype
    device: torch.device
    m_blk: torch.Tensor
    k_prev_blk: torch.Tensor
    k_curr_blk: torch.Tensor
    b_blk: torch.Tensor
    suffix_after: torch.Tensor
    b_prev_chunk0: torch.Tensor


def prepare_chunk_increment_bwd_context(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
) -> ChunkIncrementBwdContext:
    """Build the explicit per-stage context shared by the backward kernels."""
    batch_size, n_heads, T, N, P = _validate_chunk_increment_inputs(
        U, M, K, B, B_prev, U_prev, int(U.shape[2])
    )
    if U.device.type != "cuda":
        raise ValueError("CuTe chunk_increment backward requires CUDA tensors.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive. Got {chunk_size}.")

    D = 2 * N
    rdtype, _ = _resolve_dtypes(
        input_dtypes=[U.dtype, M.dtype, K.dtype, B.dtype],
        compute_dtype=compute_dtype,
        output_dtype=torch.float32,
        default_output_dtype=torch.float32,
    )
    if rdtype != torch.float32:
        raise ValueError(
            "The current CuTe chunk_increment backward supports only float32 "
            f"compute. Got compute_dtype={rdtype}."
        )

    cplx_dtype = _complex_dtype_from_real(rdtype)
    device = U.device
    U_f, M_f, K_f, B_f, _, n_chunks = _pad_time_partial(
        U, M, K, B, chunk_size=chunk_size, real_dtype=rdtype
    )
    T_pad = int(U_f.shape[2])
    L = int(chunk_size)

    m = _to_complex_scalar(M_f, name="M").to(dtype=cplx_dtype)
    k = _to_complex_taps(K_f, name="K").to(dtype=cplx_dtype)
    k_prev, k_curr = k[..., 0], k[..., 1]
    b_t = _as_complex_pairs(B_f, name="B").to(dtype=cplx_dtype)

    b_prev0, _ = _resolve_prev0(
        B_prev,
        U_prev,
        batch_size=batch_size,
        n_heads=n_heads,
        D=D,
        P=P,
        device=device,
        real_dtype=rdtype,
        complex_dtype=cplx_dtype,
    )

    m_blk = m.reshape(batch_size, n_heads, n_chunks, L)
    k_prev_blk = k_prev.reshape(batch_size, n_heads, n_chunks, L)
    k_curr_blk = k_curr.reshape(batch_size, n_heads, n_chunks, L)
    b_blk = b_t.reshape(batch_size, n_heads, n_chunks, L, N)
    suffix_after, _ = _prepared_suffix_products(m_blk)

    b_prev_chunk0 = torch.empty(
        (batch_size, n_heads, n_chunks, N), device=device, dtype=cplx_dtype
    )
    b_prev_chunk0[:, :, 0, :] = b_prev0
    if n_chunks > 1:
        b_prev_chunk0[:, :, 1:, :] = b_blk[:, :, :-1, -1, :]

    return ChunkIncrementBwdContext(
        batch_size=batch_size,
        n_heads=n_heads,
        T=T,
        T_pad=T_pad,
        n_chunks=n_chunks,
        L=L,
        N=N,
        P=P,
        D=D,
        BHC=batch_size * n_heads * n_chunks,
        rdtype=rdtype,
        cplx_dtype=cplx_dtype,
        device=device,
        m_blk=m_blk,
        k_prev_blk=k_prev_blk,
        k_curr_blk=k_curr_blk,
        b_blk=b_blk,
        suffix_after=suffix_after,
        b_prev_chunk0=b_prev_chunk0,
    )


def reshape_d_inc(d_inc: torch.Tensor, ctx: ChunkIncrementBwdContext) -> torch.Tensor:
    """Validate and flatten ``d_inc`` to ``(BHC, P, D)``."""
    if not d_inc.is_contiguous():
        raise ValueError(f"d_inc must be contiguous; got strides {d_inc.stride()}.")
    expected = (ctx.batch_size, ctx.n_heads, ctx.n_chunks, ctx.P, ctx.D)
    if d_inc.shape != expected:
        raise ValueError(
            "d_inc must be (batch, heads, chunks, P, 2N) = "
            f"{expected}. Got {tuple(d_inc.shape)}."
        )
    return d_inc.reshape(ctx.BHC, ctx.P, ctx.D).contiguous()


def validate_prepared_state(
    *,
    A_main: torch.Tensor,
    B_main: torch.Tensor,
    u_head: torch.Tensor,
    b_head: torch.Tensor,
    ctx: ChunkIncrementBwdContext,
) -> None:
    """Validate the forward-prepared state consumed by the split backward."""
    expected = {
        "A_main": (ctx.BHC, ctx.L, ctx.P),
        "B_main": (ctx.BHC, ctx.L, ctx.D),
        "u_head": (ctx.BHC, ctx.P),
        "b_head": (ctx.BHC, ctx.D),
    }
    for name, tensor in (
        ("A_main", A_main),
        ("B_main", B_main),
        ("u_head", u_head),
        ("b_head", b_head),
    ):
        if not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous; got strides {tensor.stride()}."
            )
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(f"{name} must be {expected[name]}. Got {tuple(tensor.shape)}.")


def validate_d_m_chunk(
    d_m_chunk: torch.Tensor,
    ctx: ChunkIncrementBwdContext,
) -> torch.Tensor:
    """Validate ``d_m_chunk`` against the shared backward context."""
    expected = (ctx.batch_size, ctx.n_heads, ctx.n_chunks, 2)
    if not d_m_chunk.is_contiguous():
        raise ValueError(
            f"d_m_chunk must be contiguous; got strides {d_m_chunk.stride()}."
        )
    if tuple(d_m_chunk.shape) != expected:
        raise ValueError(
            "d_m_chunk must be (batch, heads, chunks, 2) = "
            f"{expected}. Got {tuple(d_m_chunk.shape)}."
        )
    return d_m_chunk.reshape(ctx.BHC, 2).to(dtype=ctx.rdtype).contiguous()


def trim_time(tensor: torch.Tensor, *, T: int) -> torch.Tensor:
    """Trim a padded time-major tensor back to the original sequence length."""
    return tensor[:, :, :T, :].contiguous()


def _scalar_grad_from_vec(base: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Reduce ``grad`` against complex-vector ``base`` to a complex scalar grad.

    For ``y = s * base`` with complex scalar ``s`` and complex-vector output
    ``y``, the real-valued gradient w.r.t. ``s`` is the packed-complex scalar
    induced by the underlying 2x2 real multiplication:

    - ``d_re = Σ (g_re * x_re + g_im * x_im)``
    - ``d_im = Σ (-g_re * x_im + g_im * x_re)``
    """

    return torch.complex(
        (grad.real * base.real + grad.imag * base.imag).sum(dim=-1),
        (-grad.real * base.imag + grad.imag * base.real).sum(dim=-1),
    )
