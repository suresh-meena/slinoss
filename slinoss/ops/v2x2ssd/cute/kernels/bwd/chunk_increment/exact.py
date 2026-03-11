"""Exact host-side helpers for the chunk-increment backward stage."""

from __future__ import annotations

import torch

from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_increment import (
    _prepared_suffix_products,
)
from slinoss.ops.v2x2ssd.reference import (
    _as_complex_pairs,
    _complex_dtype_from_real,
    _pack_complex_pairs,
    _pad_time_partial,
    _resolve_dtypes,
    _resolve_prev0,
    _to_complex_scalar,
    _to_complex_taps,
    _validate_chunk_increment_inputs,
)

from .tail import chunk_increment_bwd_tail_exact_cute


def chunk_increment_bwd_exact_from_prepared_intermediates(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    dA_main: torch.Tensor,
    dB_main: torch.Tensor,
    d_u_head: torch.Tensor,
    d_b_head: torch.Tensor,
    d_m_chunk: torch.Tensor,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Backpropagate the prepared chunk summaries through operand prep exactly."""

    batch_size, n_heads, T, N, P = _validate_chunk_increment_inputs(
        U, M, K, B, B_prev, U_prev, int(U.shape[2])
    )
    if U.device.type != "cuda":
        raise ValueError("CuTe chunk_increment backward requires CUDA tensors.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive. Got {chunk_size}.")
    D = 2 * N

    rdtype, _ = _resolve_dtypes(
        input_dtypes=[
            U.dtype,
            M.dtype,
            K.dtype,
            B.dtype,
            dA_main.dtype,
            dB_main.dtype,
            d_u_head.dtype,
            d_b_head.dtype,
            d_m_chunk.dtype,
        ],
        compute_dtype=compute_dtype,
        output_dtype=torch.float32,
        default_output_dtype=torch.float32,
    )
    if rdtype != torch.float32:
        raise ValueError(
            "The current CuTe chunk_increment backward supports only float32 "
            f"compute. Got compute_dtype={rdtype}."
        )

    n_chunks = (T + int(chunk_size) - 1) // int(chunk_size)
    L = int(chunk_size)
    BHC = batch_size * n_heads * n_chunks
    if dA_main.shape != (BHC, L, P):
        raise ValueError(
            f"dA_main must be {(BHC, L, P)}. Got {tuple(dA_main.shape)}."
        )
    if dB_main.shape != (BHC, L, D):
        raise ValueError(
            f"dB_main must be {(BHC, L, D)}. Got {tuple(dB_main.shape)}."
        )
    if d_u_head.shape != (BHC, P):
        raise ValueError(
            f"d_u_head must be {(BHC, P)}. Got {tuple(d_u_head.shape)}."
        )
    if d_b_head.shape != (BHC, D):
        raise ValueError(
            f"d_b_head must be {(BHC, D)}. Got {tuple(d_b_head.shape)}."
        )
    if d_m_chunk.shape != (batch_size, n_heads, n_chunks, 2):
        raise ValueError(
            "d_m_chunk must be (batch, heads, chunks, 2) = "
            f"{(batch_size, n_heads, n_chunks, 2)}. Got {tuple(d_m_chunk.shape)}."
        )

    cplx_dtype = _complex_dtype_from_real(rdtype)
    device = U.device
    U_f, M_f, K_f, B_f, _, n_chunks = _pad_time_partial(
        U, M, K, B, chunk_size=chunk_size, real_dtype=rdtype
    )
    T_pad = int(U_f.shape[2])

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

    d_alpha = _as_complex_pairs(
        dB_main.reshape(batch_size, n_heads, n_chunks, L, D), name="dB_main"
    ).to(dtype=cplx_dtype)
    d_boundary = _as_complex_pairs(
        d_b_head.reshape(batch_size, n_heads, n_chunks, D), name="d_b_head"
    ).to(dtype=cplx_dtype)

    b_prev_chunk0 = torch.empty(
        (batch_size, n_heads, n_chunks, N), device=device, dtype=cplx_dtype
    )
    b_prev_chunk0[:, :, 0, :] = b_prev0
    if n_chunks > 1:
        b_prev_chunk0[:, :, 1:, :] = b_blk[:, :, :-1, -1, :]

    dU_blk = dA_main.reshape(batch_size, n_heads, n_chunks, L, P)
    dU_prev_blk = d_u_head.reshape(batch_size, n_heads, n_chunks, P)
    dB_blk = torch.zeros(
        (batch_size, n_heads, n_chunks, L, N), device=device, dtype=cplx_dtype
    )
    dB_blk += (
        torch.conj(suffix_after.unsqueeze(-1) * k_curr_blk.unsqueeze(-1)) * d_alpha
    )

    if L > 1:
        d_alpha_shift = d_alpha[..., :-1, :]
        dB_blk[..., :-1, :] += (
            torch.conj(
                suffix_after[..., 1:].unsqueeze(-1) * k_prev_blk[..., 1:].unsqueeze(-1)
            )
            * d_alpha_shift
        )

    d_b_prev_chunk0 = (
        torch.conj(
            suffix_after[..., 0].unsqueeze(-1) * k_prev_blk[..., 0].unsqueeze(-1)
        )
        * d_boundary
    )
    dK_prev_blk_r, dK_curr_blk_r, _d_suffix_after_r, dM_blk_r = (
        chunk_increment_bwd_tail_exact_cute(
            torch.view_as_real(suffix_after.reshape(BHC, L))
            .to(dtype=rdtype)
            .contiguous(),
            torch.view_as_real(k_prev_blk.reshape(BHC, L))
            .to(dtype=rdtype)
            .contiguous(),
            torch.view_as_real(k_curr_blk.reshape(BHC, L))
            .to(dtype=rdtype)
            .contiguous(),
            _pack_complex_pairs(
                b_blk.reshape(batch_size * n_heads * n_chunks, L, N),
                real_dtype=rdtype,
            )
            .reshape(batch_size * n_heads * n_chunks, L, D)
            .contiguous(),
            _pack_complex_pairs(
                b_prev_chunk0.reshape(batch_size * n_heads * n_chunks, N),
                real_dtype=rdtype,
            )
            .reshape(batch_size * n_heads * n_chunks, D)
            .contiguous(),
            _pack_complex_pairs(
                d_alpha.reshape(batch_size * n_heads * n_chunks, L, N),
                real_dtype=rdtype,
            )
            .reshape(batch_size * n_heads * n_chunks, L, D)
            .contiguous(),
            _pack_complex_pairs(
                d_boundary.reshape(batch_size * n_heads * n_chunks, N),
                real_dtype=rdtype,
            )
            .reshape(batch_size * n_heads * n_chunks, D)
            .contiguous(),
            torch.view_as_real(m_blk.reshape(BHC, L)).to(dtype=rdtype).contiguous(),
            d_m_chunk.reshape(batch_size * n_heads * n_chunks, 2)
            .to(dtype=rdtype)
            .contiguous(),
        )
    )
    dK_prev_blk = torch.view_as_complex(dK_prev_blk_r.contiguous()).reshape_as(k_prev_blk)
    dK_curr_blk = torch.view_as_complex(dK_curr_blk_r.contiguous()).reshape_as(k_curr_blk)

    dU_prev = dU_prev_blk[:, :, 0, :].contiguous()
    if n_chunks > 1:
        dU_blk[:, :, :-1, -1, :] += dU_prev_blk[:, :, 1:, :]

    dB_prev_c = d_b_prev_chunk0[:, :, 0, :].contiguous()
    if n_chunks > 1:
        dB_blk[:, :, :-1, -1, :] += d_b_prev_chunk0[:, :, 1:, :]

    dM_blk = torch.view_as_complex(dM_blk_r.contiguous()).reshape_as(m_blk)

    dU = dU_blk.reshape(batch_size, n_heads, T_pad, P)[:, :, :T, :].contiguous()
    dM = torch.view_as_real(dM_blk.reshape(batch_size, n_heads, T_pad))
    dM = dM.to(dtype=rdtype)[:, :, :T, :].contiguous()

    dK_blk = torch.stack((dK_prev_blk, dK_curr_blk), dim=-1)
    dK = torch.view_as_real(dK_blk.reshape(batch_size, n_heads, T_pad, 2))
    dK = dK.reshape(batch_size, n_heads, T_pad, 2, 2).to(dtype=rdtype)
    dK = dK[:, :, :T, :, :].contiguous()

    dB = _pack_complex_pairs(
        dB_blk.reshape(batch_size, n_heads, T_pad, N), real_dtype=rdtype
    )
    dB = dB[:, :, :T, :].contiguous()
    dB_prev = _pack_complex_pairs(dB_prev_c, real_dtype=rdtype)

    return dU, dM, dK, dB, dB_prev, dU_prev.to(dtype=rdtype).contiguous()


__all__ = ["chunk_increment_bwd_exact_from_prepared_intermediates"]
