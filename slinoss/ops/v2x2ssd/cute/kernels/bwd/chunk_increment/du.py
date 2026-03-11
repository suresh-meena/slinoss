"""Main ``dU`` path for the CuTe ``v2x2ssd`` chunk-increment backward."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_increment import (
    _get_compiled_gemm,
    _mark_output,
    _mark_prepared_input,
)

from .common import ChunkIncrementBwdContext


@dataclass(frozen=True)
class ChunkIncrementBwdDUResult:
    dU: torch.Tensor
    dU_prev: torch.Tensor


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


def chunk_increment_bwd_du_cute(
    *,
    B_main: torch.Tensor,
    b_head: torch.Tensor,
    d_inc_flat: torch.Tensor,
    ctx: ChunkIncrementBwdContext,
) -> ChunkIncrementBwdDUResult:
    """Run the main ``dU`` contraction and boundary scatter."""
    dA_main = _prepared_dA_main_cute(d_inc_flat, B_main)
    d_u_head = torch.einsum("bpd,bd->bp", d_inc_flat, b_head)

    dU_blk = dA_main.reshape(ctx.batch_size, ctx.n_heads, ctx.n_chunks, ctx.L, ctx.P)
    dU_prev_blk = d_u_head.reshape(ctx.batch_size, ctx.n_heads, ctx.n_chunks, ctx.P)

    dU_prev = dU_prev_blk[:, :, 0, :].contiguous()
    if ctx.n_chunks > 1:
        dU_blk[:, :, :-1, -1, :] += dU_prev_blk[:, :, 1:, :]

    dU = dU_blk.reshape(ctx.batch_size, ctx.n_heads, ctx.T_pad, ctx.P)
    dU = dU[:, :, : ctx.T, :].contiguous()
    return ChunkIncrementBwdDUResult(dU=dU, dU_prev=dU_prev.to(dtype=ctx.rdtype))


__all__ = ["ChunkIncrementBwdDUResult", "chunk_increment_bwd_du_cute"]
