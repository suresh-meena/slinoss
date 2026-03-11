"""Main ``dB`` path for the CuTe ``v2x2ssd`` chunk-increment backward."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_increment import (
    _get_compiled_gemm,
    _mark_output,
    _mark_prepared_input,
)
from slinoss.ops.v2x2ssd.reference import _as_complex_pairs

from .common import ChunkIncrementBwdContext


@dataclass(frozen=True)
class ChunkIncrementBwdDBResult:
    d_alpha: torch.Tensor
    dB_blk: torch.Tensor


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


def chunk_increment_bwd_db_cute(
    *,
    A_main: torch.Tensor,
    d_inc_flat: torch.Tensor,
    ctx: ChunkIncrementBwdContext,
) -> ChunkIncrementBwdDBResult:
    """Run the main ``dB`` contraction and local value-lane transport."""
    dB_main = _prepared_dB_main_cute(A_main, d_inc_flat)
    d_alpha = _as_complex_pairs(
        dB_main.reshape(ctx.batch_size, ctx.n_heads, ctx.n_chunks, ctx.L, ctx.D),
        name="dB_main",
    ).to(dtype=ctx.cplx_dtype)

    dB_blk = torch.zeros(
        (ctx.batch_size, ctx.n_heads, ctx.n_chunks, ctx.L, ctx.N),
        device=ctx.device,
        dtype=ctx.cplx_dtype,
    )
    dB_blk += (
        torch.conj(ctx.suffix_after.unsqueeze(-1) * ctx.k_curr_blk.unsqueeze(-1))
        * d_alpha
    )
    if ctx.L > 1:
        dB_blk[..., :-1, :] += (
            torch.conj(
                ctx.suffix_after[..., 1:].unsqueeze(-1)
                * ctx.k_prev_blk[..., 1:].unsqueeze(-1)
            )
            * d_alpha[..., :-1, :]
        )

    return ChunkIncrementBwdDBResult(d_alpha=d_alpha, dB_blk=dB_blk)


__all__ = ["ChunkIncrementBwdDBResult", "chunk_increment_bwd_db_cute"]
