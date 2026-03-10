"""Backward kernels for the CuTe ``v2x2ssd`` state-passing stage."""

from .m import state_passing_bwd_m_cute
from .state import state_passing_bwd_state_cute


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
    "state_passing_bwd_state_cute",
    "state_passing_bwd_m_cute",
    "state_passing_bwd_cute",
]
