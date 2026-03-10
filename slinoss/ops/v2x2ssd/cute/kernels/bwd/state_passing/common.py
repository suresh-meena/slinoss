"""Common helpers for CuTe ``v2x2ssd`` state-passing backward kernels.

Logical contract
----------------
- ``chunk_starts``: ``(B, H, C, P, D)``
- ``d_chunk_starts``: ``(B, H, C, P, D)``
- ``d_final``: ``(B, H, P, D)``
- ``m_chunk``: ``(B, H, C, 2)``
- ``d_inc``: ``(B, H, C, P, D)``
- ``d_initial``: ``(B, H, P, D)``
- ``d_m_chunk``: ``(B, H, C, 2)``

The hot contiguous axis is always ``S = P * D`` with ``D = 2N`` interleaved
complex pairs. Each thread therefore owns whole ``(re, im)`` pairs, never
strided scalar lanes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _elem_bits(dt: torch.dtype) -> int:
    if dt == torch.float32:
        return 32
    if dt in (torch.float16, torch.bfloat16):
        return 16
    raise TypeError(f"Unsupported dtype: {dt}")


def _choose_copy_bits_for_linear_tiles(
    t: torch.Tensor,
    tile_stride_elems: int,
    *,
    elems_per_thread: int,
    candidates_bits: tuple[int, ...] = (128, 64, 32),
) -> int:
    """Pick the widest CopyUniversalOp width safe for all linear tile starts."""
    eb = _elem_bits(t.dtype)
    elem_bytes = t.element_size()
    stride_bytes = tile_stride_elems * elem_bytes

    best = eb
    for bits in candidates_bits:
        if bits < eb or bits % eb != 0:
            continue
        vec_elems = bits // eb
        if elems_per_thread % vec_elems != 0:
            continue
        align = bits // 8
        if (t.data_ptr() % align) == 0 and (stride_bytes % align) == 0:
            best = bits
            break
    return best


@dataclass(frozen=True)
class _TileConfig:
    num_threads: int = 128
    pairs_per_thread: int = 8

    @property
    def elems_per_thread(self) -> int:
        return 2 * int(self.pairs_per_thread)

    @property
    def tile(self) -> int:
        return int(self.num_threads) * self.elems_per_thread
