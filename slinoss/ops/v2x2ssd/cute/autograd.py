"""Training-only autograd wrapper for the CuTe ``v2x2ssd`` operator."""

from __future__ import annotations

from typing import cast

import torch

from slinoss.perf import record_region
from slinoss.ops.v2x2ssd.cute.kernels.bwd import v2x2ssd_bwd_cute
from slinoss.ops.v2x2ssd.cute.kernels.fwd import v2x2ssd_fwd_cute


class _V2x2SSDCuTeTrainingFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        U: torch.Tensor,
        M: torch.Tensor,
        K: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk_size: int,
        compute_dtype: torch.dtype | None,
        output_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        ctx.chunk_size = int(chunk_size)
        ctx.compute_dtype = compute_dtype
        ctx.output_dtype = output_dtype

        U_d = U.detach()
        M_d = M.detach()
        K_d = K.detach()
        B_d = B.detach()
        C_d = C.detach()

        Y, m_chunk, chunk_starts = v2x2ssd_fwd_cute(
            U_d,
            M_d,
            K_d,
            B_d,
            C_d,
            chunk_size=ctx.chunk_size,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype or U.dtype,
        )

        ctx.save_for_backward(U_d, M_d, K_d, B_d, C_d, m_chunk, chunk_starts)
        return Y

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        dY: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
    ]:
        with record_region("backward.v2x2ssd.custom_op_total"):
            U, M, K, B, C, m_chunk, chunk_starts = ctx.saved_tensors

            if dY is None:
                with record_region("backward.v2x2ssd.autograd_wrapper"):
                    return (
                        torch.zeros_like(U),
                        torch.zeros_like(M),
                        torch.zeros_like(K),
                        torch.zeros_like(B),
                        torch.zeros_like(C),
                        None,
                        None,
                        None,
                    )

            with record_region("backward.v2x2ssd.autograd_wrapper"):
                dY_contig = dY if dY.is_contiguous() else dY.contiguous()

            dU_scan, dM_scan, dK_scan, dB_scan, dC_scan = cast(
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
                v2x2ssd_bwd_cute(
                    U,
                    M,
                    K,
                    B,
                    C,
                    m_chunk,
                    chunk_starts,
                    dY_contig,
                    chunk_size=ctx.chunk_size,
                    compute_dtype=ctx.compute_dtype,
                ),
            )

            return (
                dU_scan,
                dM_scan,
                dK_scan,
                dB_scan,
                dC_scan,
                None,
                None,
                None,
            )


def v2x2ssd_cute_training_autograd(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return cast(
        torch.Tensor,
        _V2x2SSDCuTeTrainingFn.apply(
            U,
            M,
            K,
            B,
            C,
            int(chunk_size),
            compute_dtype,
            output_dtype,
        ),
    )


__all__ = ["v2x2ssd_cute_training_autograd"]
