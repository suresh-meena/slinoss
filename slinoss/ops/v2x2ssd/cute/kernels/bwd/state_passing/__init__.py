"""Backward kernels for the CuTe ``v2x2ssd`` state-passing stage."""

from __future__ import annotations

from collections.abc import Callable

import torch
from cutlass.cute.runtime import from_dlpack

from .m import _get_compiled_m_kernel, state_passing_bwd_m_cute
from .state import _get_compiled_state_kernel, state_passing_bwd_state_cute


def compile_state_passing_bwd_kernels(
    chunk_starts: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    d_chunk_starts: torch.Tensor,
    d_final: torch.Tensor,
    return_launchers: bool = False,
) -> (
    tuple[object, object, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[
        object,
        object,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Callable[[], None],
    ]
):
    """Compile and allocate the staged ``state_passing`` backward pipeline.

    Returns ``(compiled_state, compiled_m, d_inc, d_m_chunk, d_initial)`` and,
    when ``return_launchers=True``, appends a ``launch_pipeline`` callable that
    executes the two kernels sequentially on the current stream.
    """
    if chunk_starts.device.type != "cuda":
        raise ValueError("CuTe state_passing backward requires CUDA tensors.")
    if chunk_starts.dtype not in (torch.float16, torch.float32):
        raise ValueError("chunk_starts must be fp16 or fp32.")
    if d_chunk_starts.dtype != torch.float32 or d_final.dtype != torch.float32:
        raise ValueError("d_chunk_starts and d_final must be fp32.")
    if m_chunk.dtype != torch.float32:
        raise ValueError("m_chunk must be fp32.")
    if chunk_starts.ndim != 5 or d_chunk_starts.ndim != 5:
        raise ValueError("chunk_starts and d_chunk_starts must be rank-5.")
    if d_final.ndim != 4 or m_chunk.ndim != 4:
        raise ValueError("d_final must be rank-4 and m_chunk must be rank-4.")
    if chunk_starts.shape != d_chunk_starts.shape:
        raise ValueError("chunk_starts and d_chunk_starts must have identical shapes.")
    if chunk_starts.shape[:3] != m_chunk.shape[:3]:
        raise ValueError("Leading (B,H,C) dims of chunk_starts and m_chunk must match.")
    if chunk_starts.shape[-1] % 2 != 0:
        raise ValueError("The flattened D dimension must be even.")

    chunk_starts_c = chunk_starts.contiguous()
    d_chunk_starts_c = d_chunk_starts.contiguous()
    d_final_c = d_final.contiguous()
    m_chunk_c = m_chunk.contiguous()

    B, H, C, P, D = map(int, chunk_starts_c.shape)
    if d_final_c.shape != (B, H, P, D):
        raise ValueError(
            f"d_final must be {(B, H, P, D)}. Got {tuple(d_final_c.shape)}."
        )
    if m_chunk_c.shape != (B, H, C, 2):
        raise ValueError(
            f"m_chunk must be {(B, H, C, 2)}. Got {tuple(m_chunk_c.shape)}."
        )

    d_inc = torch.empty_like(d_chunk_starts_c)
    d_initial = torch.empty_like(d_final_c)
    d_m_chunk = torch.empty(
        (B, H, C, 2), device=chunk_starts.device, dtype=torch.float32
    )

    compiled_state = _get_compiled_state_kernel(
        d_chunk_starts_c,
        d_final_c,
        m_chunk_c,
        d_inc,
        d_initial,
    )
    compiled_m = _get_compiled_m_kernel(chunk_starts_c, d_inc, d_m_chunk)

    def launch_pipeline() -> None:
        compiled_state(
            from_dlpack(
                d_chunk_starts_c, assumed_align=d_chunk_starts_c.element_size()
            ),
            from_dlpack(d_final_c, assumed_align=d_final_c.element_size()),
            from_dlpack(m_chunk_c, assumed_align=max(m_chunk_c.element_size(), 8)),
            from_dlpack(d_inc, assumed_align=d_inc.element_size()),
            from_dlpack(d_initial, assumed_align=d_initial.element_size()),
        )
        compiled_m(
            from_dlpack(chunk_starts_c, assumed_align=chunk_starts_c.element_size()),
            from_dlpack(d_inc, assumed_align=d_inc.element_size()),
            from_dlpack(d_m_chunk, assumed_align=max(d_m_chunk.element_size(), 8)),
        )

    base = (compiled_state, compiled_m, d_inc, d_m_chunk, d_initial)
    if return_launchers:
        return (*base, launch_pipeline)
    return base


def state_passing_bwd_cute(
    d_chunk_starts,
    d_final,
    chunk_starts,
    m_chunk,
):
    """Run the full CuTe backward for the ``state_passing`` stage.

    Returns ``(d_inc, d_m_chunk, d_initial)`` in fp32. ``chunk_starts`` may be
    fp16 or fp32 transport storage; the reduction kernel accumulates in fp32.
    """
    d_inc, d_initial = state_passing_bwd_state_cute(d_chunk_starts, d_final, m_chunk)
    d_m_chunk = state_passing_bwd_m_cute(chunk_starts, d_inc)
    return d_inc, d_m_chunk, d_initial


__all__ = [
    "compile_state_passing_bwd_kernels",
    "state_passing_bwd_state_cute",
    "state_passing_bwd_m_cute",
    "state_passing_bwd_cute",
]
