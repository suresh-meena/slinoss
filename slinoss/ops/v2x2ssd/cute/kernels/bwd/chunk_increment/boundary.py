"""Boundary branch for the CuTe ``v2x2ssd`` chunk-increment backward."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from slinoss.ops.v2x2ssd.reference import _as_complex_pairs, _pack_complex_pairs

from .common import ChunkIncrementBwdContext


@dataclass(frozen=True)
class ChunkIncrementBwdBoundaryResult:
    d_boundary: torch.Tensor
    d_b_prev_chunk0: torch.Tensor
    dB_prev: torch.Tensor


def chunk_increment_bwd_boundary_cute(
    *,
    u_head: torch.Tensor,
    d_inc_flat: torch.Tensor,
    ctx: ChunkIncrementBwdContext,
) -> ChunkIncrementBwdBoundaryResult:
    """Run the rank-1 boundary branch for ``chunk_increment`` backward."""
    d_b_head = torch.einsum("bpd,bp->bd", d_inc_flat, u_head)
    d_boundary = _as_complex_pairs(
        d_b_head.reshape(ctx.batch_size, ctx.n_heads, ctx.n_chunks, ctx.D),
        name="d_b_head",
    ).to(dtype=ctx.cplx_dtype)

    d_b_prev_chunk0 = (
        torch.conj(
            ctx.suffix_after[..., 0].unsqueeze(-1)
            * ctx.k_prev_blk[..., 0].unsqueeze(-1)
        )
        * d_boundary
    )
    dB_prev = _pack_complex_pairs(
        d_b_prev_chunk0[:, :, 0, :].contiguous(),
        real_dtype=ctx.rdtype,
    )
    return ChunkIncrementBwdBoundaryResult(
        d_boundary=d_boundary,
        d_b_prev_chunk0=d_b_prev_chunk0,
        dB_prev=dB_prev,
    )


__all__ = ["ChunkIncrementBwdBoundaryResult", "chunk_increment_bwd_boundary_cute"]
