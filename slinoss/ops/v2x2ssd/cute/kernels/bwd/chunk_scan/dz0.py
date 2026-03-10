"""CuTe backward slice for ``chunk_scan`` gradients into ``chunk_starts``.

This is the ``dZ0`` analogue of the v3 ``chunk_scan`` backward split. It only
computes the gradient of the chunk off-term with respect to ``chunk_starts``:

``y_off = Re((conj(C) * prefix) @ chunk_starts^T)``

Given real upstream ``dOut``, the packed-real gradient is:

``dZ0 = dOut^T @ pack(conj(conj(C) * prefix))``

So after a small SO(2)-specific host prep, the dense work is another batched
GEMM on the same prepared-input contract already used by ``chunk_increment``.
"""

from __future__ import annotations

import torch

from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_increment import (
    _get_compiled_gemm,
    _mark_output,
    _mark_prepared_input,
)
from slinoss.ops.v2x2ssd.reference import (
    _as_complex_pairs,
    _chunked_transition_prefix_parts,
    _complex_dtype_from_real,
    _pack_complex_pairs,
    _pad_time_full,
    _resolve_dtypes,
    _to_complex_scalar,
)


def chunk_scan_bwd_dz0_cute(
    M: torch.Tensor,
    C: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute ``d_chunk_starts`` for the ``chunk_scan`` off-term in fp32."""
    if (
        M.device.type != "cuda"
        or C.device.type != "cuda"
        or d_out.device.type != "cuda"
    ):
        raise ValueError("CuTe chunk_scan backward requires CUDA tensors.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive. Got {chunk_size}.")
    if M.ndim != 4 or M.shape[-1] != 2:
        raise ValueError(f"M must be (batch, heads, T, 2). Got {tuple(M.shape)}.")
    if C.ndim != 4 or d_out.ndim != 4:
        raise ValueError("C and d_out must be rank-4 tensors.")
    if M.shape[:3] != C.shape[:3] or M.shape[:3] != d_out.shape[:3]:
        raise ValueError("Leading (batch, heads, T) dims of M/C/d_out must match.")
    if C.shape[-1] % 2 != 0:
        raise ValueError(f"C must have an even 2N trailing dim. Got {tuple(C.shape)}.")
    if not (M.is_contiguous() and C.is_contiguous() and d_out.is_contiguous()):
        raise ValueError("M, C, and d_out must be contiguous.")

    rdtype, _ = _resolve_dtypes(
        input_dtypes=[M.dtype, C.dtype, d_out.dtype],
        compute_dtype=compute_dtype,
        output_dtype=torch.float32,
        default_output_dtype=torch.float32,
    )
    if rdtype != torch.float32:
        raise ValueError(
            "The current CuTe chunk_scan dZ0 path supports only float32 "
            f"compute. Got compute_dtype={rdtype}."
        )

    batch_size, n_heads, T, P = map(int, d_out.shape)
    D = int(C.shape[-1])
    N = D // 2
    cplx_dtype = _complex_dtype_from_real(rdtype)
    device = C.device

    # This is a hot backward slice. Like the forward CuTe path, it assumes the
    # caller stays in the normal operating region and does not do full-tensor
    # finite scans here.

    dummy_u = torch.empty((batch_size, n_heads, T, P), device=device, dtype=rdtype)
    dummy_k = torch.empty((batch_size, n_heads, T, 2, 2), device=device, dtype=rdtype)
    dummy_b = torch.empty((batch_size, n_heads, T, D), device=device, dtype=rdtype)
    _, M_f, _, _, C_f, T_pad, n_chunks = _pad_time_full(
        dummy_u, M, dummy_k, dummy_b, C, chunk_size=chunk_size, real_dtype=rdtype
    )
    L = int(chunk_size)

    m = _to_complex_scalar(M_f, name="M").to(dtype=cplx_dtype)
    c_conj = torch.conj(_as_complex_pairs(C_f, name="C").to(dtype=cplx_dtype))

    m_blk = m.reshape(batch_size, n_heads, n_chunks, L)
    c_blk = c_conj.reshape(batch_size, n_heads, n_chunks, L, N)
    _, _, prefix = _chunked_transition_prefix_parts(m_blk)

    q_off = torch.conj(c_blk * prefix.unsqueeze(-1)).resolve_conj()
    q_packed = _pack_complex_pairs(
        q_off.reshape(batch_size * n_heads * n_chunks, L, N),
        real_dtype=rdtype,
    )

    d_out_blk = d_out.to(dtype=rdtype)
    if T_pad != T:
        pad = T_pad - T
        d_out_blk = torch.cat(
            [
                d_out_blk,
                torch.zeros((batch_size, n_heads, pad, P), device=device, dtype=rdtype),
            ],
            dim=2,
        )
    d_out_blk = d_out_blk.reshape(batch_size * n_heads * n_chunks, L, P)

    A3 = d_out_blk.permute(2, 1, 0)
    B3 = q_packed.permute(2, 1, 0)
    dZ0 = torch.empty(
        (batch_size * n_heads * n_chunks, P, D), device=device, dtype=torch.float32
    )
    C3 = dZ0.permute(1, 2, 0)

    compiled = _get_compiled_gemm(A3, B3, C3)
    compiled(_mark_prepared_input(A3), _mark_prepared_input(B3), _mark_output(C3))
    return dZ0.reshape(batch_size, n_heads, n_chunks, P, D).contiguous()


__all__ = ["chunk_scan_bwd_dz0_cute"]
