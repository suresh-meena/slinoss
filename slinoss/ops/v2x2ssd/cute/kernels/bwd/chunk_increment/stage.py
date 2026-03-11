"""CuTe backward orchestration for the ``v2x2ssd`` chunk-increment stage.

Logical contract
----------------
- inputs:
  - ``U``: ``(B, H, T, P)``
  - ``M``: ``(B, H, T, 2)``
  - ``K``: ``(B, H, T, 2, 2)``
  - ``B``: ``(B, H, T, D)`` with ``D = 2N`` interleaved complex lanes
  - ``B_prev``: ``(B, H, D)``
  - ``U_prev``: ``(B, H, P)``
  - ``d_inc``: ``(B, H, C, P, D)``
  - ``d_m_chunk``: ``(B, H, C, 2)``
- outputs:
  - ``dU``: ``(B, H, T, P)``
  - ``dM``: ``(B, H, T, 2)``
  - ``dK``: ``(B, H, T, 2, 2)``
  - ``dB``: ``(B, H, T, D)``
  - ``dB_prev``: ``(B, H, D)``
  - ``dU_prev``: ``(B, H, P)``

Layout / launch
---------------
The dense backward contractions reuse the forward SGEMM kernel on prepared
operand views:

- ``dA_main = d_inc @ B_main^T`` via one batched GEMM
- ``dB_main = A_main @ d_inc`` via a second batched GEMM

Those prepared views keep the same feature-major stride contract as the forward
kernel, so no new CuTe layout family is introduced for this stage.

The exact short-axis operand-prep backward lives in ``exact.py``. This stage
keeps only the prepared tensor-core contractions and package-level
orchestration.
"""

from __future__ import annotations

import torch

from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_increment import (
    _get_compiled_gemm,
    _mark_output,
    _mark_prepared_input,
    _prepare_chunk_increment_operands,
)

from .exact import chunk_increment_bwd_exact_from_prepared_intermediates


def _prepared_dA_main_cute(
    d_inc_flat: torch.Tensor,
    B_main: torch.Tensor,
) -> torch.Tensor:
    """Compute ``dA_main`` in ``(BHC, L, P)`` by reusing the forward SGEMM."""
    dinc_t = d_inc_flat.transpose(1, 2).contiguous()
    B_main_t = B_main.transpose(1, 2).contiguous()

    A3 = dinc_t.permute(2, 1, 0)
    B3 = B_main_t.permute(2, 1, 0)
    dA_buf = torch.empty(
        (d_inc_flat.shape[0], d_inc_flat.shape[1], B_main.shape[1]),
        device=d_inc_flat.device,
        dtype=torch.float32,
    )
    C3 = dA_buf.permute(1, 2, 0)

    compiled = _get_compiled_gemm(A3, B3, C3)
    compiled(_mark_prepared_input(A3), _mark_prepared_input(B3), _mark_output(C3))
    return dA_buf.transpose(1, 2).contiguous()


def _prepared_dB_main_cute(
    A_main: torch.Tensor,
    d_inc_flat: torch.Tensor,
) -> torch.Tensor:
    """Compute ``dB_main`` in ``(BHC, L, D)`` by reusing the forward SGEMM."""
    A_main_t = A_main.transpose(1, 2).contiguous()

    A3 = A_main_t.permute(2, 1, 0)
    B3 = d_inc_flat.permute(2, 1, 0)
    dB_main = torch.empty(
        (A_main.shape[0], A_main.shape[1], d_inc_flat.shape[-1]),
        device=A_main.device,
        dtype=torch.float32,
    )
    C3 = dB_main.permute(1, 2, 0)

    compiled = _get_compiled_gemm(A3, B3, C3)
    compiled(_mark_prepared_input(A3), _mark_prepared_input(B3), _mark_output(C3))
    return dB_main


def _chunk_increment_bwd_from_prepared_operands(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    A_main: torch.Tensor,
    B_main: torch.Tensor,
    u_head: torch.Tensor,
    b_head: torch.Tensor,
    d_inc: torch.Tensor,
    d_m_chunk: torch.Tensor,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Low-level chunk-increment backward using prepared forward operands."""
    batch_size, n_heads, T, P = map(int, U.shape)
    D = int(B.shape[-1])
    n_chunks = (T + int(chunk_size) - 1) // int(chunk_size)

    for name, tensor in (
        ("A_main", A_main),
        ("B_main", B_main),
        ("u_head", u_head),
        ("b_head", b_head),
        ("d_inc", d_inc),
        ("d_m_chunk", d_m_chunk),
    ):
        if not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous; got strides {tensor.stride()}."
            )

    if d_inc.shape != (batch_size, n_heads, n_chunks, P, D):
        raise ValueError(
            "d_inc must be (batch, heads, chunks, P, 2N) = "
            f"{(batch_size, n_heads, n_chunks, P, D)}. Got {tuple(d_inc.shape)}."
        )
    if d_m_chunk.shape != (batch_size, n_heads, n_chunks, 2):
        raise ValueError(
            "d_m_chunk must be (batch, heads, chunks, 2) = "
            f"{(batch_size, n_heads, n_chunks, 2)}. Got {tuple(d_m_chunk.shape)}."
        )

    BHC = int(A_main.shape[0])
    d_inc_flat = d_inc.reshape(BHC, P, D).contiguous()

    dA_main = _prepared_dA_main_cute(d_inc_flat, B_main)
    dB_main = _prepared_dB_main_cute(A_main, d_inc_flat)
    d_u_head = torch.einsum("bpd,bd->bp", d_inc_flat, b_head)
    d_b_head = torch.einsum("bpd,bp->bd", d_inc_flat, u_head)
    return chunk_increment_bwd_exact_from_prepared_intermediates(
        U,
        M,
        K,
        B,
        dA_main=dA_main,
        dB_main=dB_main,
        d_u_head=d_u_head,
        d_b_head=d_b_head,
        d_m_chunk=d_m_chunk,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
    )


def chunk_increment_bwd_stage_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    d_inc: torch.Tensor,
    d_m_chunk: torch.Tensor,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Low-level stage implementation for ``chunk_increment`` backward."""
    A_main, B_main, u_head, b_head, _, _, _, _, _ = _prepare_chunk_increment_operands(
        U,
        M,
        K,
        B,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
    )
    return _chunk_increment_bwd_from_prepared_operands(
        U,
        M,
        K,
        B,
        A_main=A_main,
        B_main=B_main,
        u_head=u_head,
        b_head=b_head,
        d_inc=d_inc,
        d_m_chunk=d_m_chunk,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
    )


__all__ = ["chunk_increment_bwd_stage_cute", "_chunk_increment_bwd_from_prepared_operands"]
