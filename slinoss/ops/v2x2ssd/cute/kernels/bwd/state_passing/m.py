from __future__ import annotations

from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .common import _TileConfig, _choose_copy_bits_for_linear_tiles


_CompiledKey = tuple[
    int,
    torch.dtype,
    torch.dtype,
    tuple[int, int, int, int, int],
    tuple[int, int, int, int, int],
]
_COMPILED_STATE_PASSING_BWD_M: dict[_CompiledKey, Callable[..., object]] = {}


class _StatePassingBwdMAmpere:
    """Backward kernel for ``d_m_chunk``.

    Logical shapes:
    - ``chunk_starts``: ``(B, H, C, P, D)``
    - ``d_inc``: ``(B, H, C, P, D)``
    - output ``d_m_chunk``: ``(B, H, C, 2)``

    Layout / launch:
    - flatten ``S = P * D`` as the contiguous axis
    - grid ``(C, B * H, 1)``
    - each CTA reduces one ``(batch, head, chunk)`` row over all ``S`` lanes
    - each thread owns whole complex pairs and accumulates fp32 partial sums

    Numerical contract:
    - for ``y = m * z`` with ``m=(mr, mi)``, ``z=(zr, zi)``, upstream
      ``g=(gr, gi)``, the local contributions are:
      ``d_mr += gr*zr + gi*zi``
      ``d_mi += -gr*zi + gi*zr``
    """

    def __init__(self, cfg: _TileConfig, *, copy_bits_starts: int, copy_bits_dinc: int):
        self.cfg = cfg
        self.copy_bits_starts = int(copy_bits_starts)
        self.copy_bits_dinc = int(copy_bits_dinc)

    @cute.jit
    def __call__(
        self,
        chunk_starts: cute.Tensor,
        d_inc: cute.Tensor,
        d_m_chunk: cute.Tensor,
    ) -> None:
        B, H, C, P, D = chunk_starts.shape
        BH = B * H
        S = P * D

        layout_bcs = cute.make_layout((BH, C, S), stride=(C * S, S, 1))
        layout_bcm = cute.make_layout((BH, C, 2), stride=(C * 2, 2, 1))

        starts_flat = cute.make_tensor(chunk_starts.iterator, layout_bcs)
        dinc_flat = cute.make_tensor(d_inc.iterator, layout_bcs)
        dm_flat = cute.make_tensor(d_m_chunk.iterator, layout_bcm)

        tv_layout = cute.make_layout(
            (self.cfg.num_threads, self.cfg.elems_per_thread),
            stride=(self.cfg.elems_per_thread, 1),
        )
        idS = cute.make_identity_tensor(S)
        cS = cute.zipped_divide(idS, tiler=cute.make_layout(self.cfg.tile))

        self.kernel(starts_flat, dinc_flat, dm_flat, cS, tv_layout).launch(
            grid=[C, BH, 1],
            block=[self.cfg.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        starts_flat: cute.Tensor,
        dinc_flat: cute.Tensor,
        dm_flat: cute.Tensor,
        cS: cute.Tensor,
        tv_layout: cute.Layout,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        c, bh, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()

        S = starts_flat.shape[2]
        n_tiles = (S + cutlass.Int32(self.cfg.tile) - 1) // cutlass.Int32(self.cfg.tile)

        copy_starts_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            starts_flat.element_type,
            num_bits_per_copy=self.copy_bits_starts,
        )
        copy_starts_scalar = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            starts_flat.element_type,
            num_bits_per_copy=starts_flat.element_type.width,
        )
        copy_dinc_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dinc_flat.element_type,
            num_bits_per_copy=self.copy_bits_dinc,
        )
        copy_dinc_scalar = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dinc_flat.element_type,
            num_bits_per_copy=dinc_flat.element_type.width,
        )

        tile_layout = cute.make_layout(cS.shape[0])

        d_mr = cutlass.Float32(0.0)
        d_mi = cutlass.Float32(0.0)
        pairs_per_thread = self.cfg.elems_per_thread // 2

        for tile_idx in cutlass.range(n_tiles, unroll=1):
            tile_start = cutlass.Int32(self.cfg.tile) * tile_idx
            residue = S - tile_start
            is_partial_tile = cute.elem_less(residue, cutlass.Int32(self.cfg.tile))

            cta_coord = (None, tile_idx)
            cta_crd = cS[cta_coord]
            tid_crd = cute.composition(cta_crd, tv_layout)
            thr_crd = tid_crd[tidx, None]

            frg_pred = cute.make_rmem_tensor(thr_crd.shape, cutlass.Boolean)
            frg_pred.fill(cutlass.Boolean(True))
            if is_partial_tile:
                for i in cutlass.range_constexpr(cute.size(frg_pred)):
                    frg_pred[i] = cute.elem_less(thr_crd[i], S)

            g_starts = starts_flat[bh, c, None]
            g_dinc = dinc_flat[bh, c, None]

            t_starts = cute.zipped_divide(g_starts, tiler=tile_layout)
            t_dinc = cute.zipped_divide(g_dinc, tiler=tile_layout)
            cta_starts = t_starts[cta_coord]
            cta_dinc = t_dinc[cta_coord]
            tid_starts = cute.composition(cta_starts, tv_layout)
            tid_dinc = cute.composition(cta_dinc, tv_layout)
            thr_starts = tid_starts[tidx, None]
            thr_dinc = tid_dinc[tidx, None]

            frg_starts = cute.make_rmem_tensor_like(thr_starts)
            frg_dinc = cute.make_rmem_tensor_like(thr_dinc)
            frg_starts.fill(0)
            frg_dinc.fill(0)
            if is_partial_tile:
                cute.copy(copy_starts_scalar, thr_starts, frg_starts, pred=frg_pred)
                cute.copy(copy_dinc_scalar, thr_dinc, frg_dinc, pred=frg_pred)
            else:
                cute.copy(copy_starts_vec, thr_starts, frg_starts)
                cute.copy(copy_dinc_vec, thr_dinc, frg_dinc)

            z = frg_starts.load().to(cutlass.Float32)
            g = frg_dinc.load().to(cutlass.Float32)

            for v in cutlass.range_constexpr(pairs_per_thread):
                base = v * 2
                zr = z[base + 0]
                zi = z[base + 1]
                gr = g[base + 0]
                gi = g[base + 1]
                d_mr += (gr * zr) + (gi * zi)
                d_mi += (-gr * zi) + (gi * zr)

        for offset in (16, 8, 4, 2, 1):
            d_mr += cute.arch.shuffle_sync_bfly(
                d_mr, offset=offset, mask=-1, mask_and_clamp=31
            )
            d_mi += cute.arch.shuffle_sync_bfly(
                d_mi, offset=offset, mask=-1, mask_and_clamp=31
            )

        smem = cutlass.utils.SmemAllocator()
        warp_mr = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((32,), stride=(1,)), byte_alignment=4
        )
        warp_mi = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((32,), stride=(1,)), byte_alignment=4
        )

        if lane == cutlass.Int32(0):
            warp_mr[warp] = d_mr
            warp_mi[warp] = d_mi

        cute.arch.barrier()

        if warp == cutlass.Int32(0):
            num_warps = cutlass.Int32(self.cfg.num_threads // 32)
            w = lane
            mr_val = cutlass.select_(w < num_warps, warp_mr[w], cutlass.Float32(0.0))
            mi_val = cutlass.select_(w < num_warps, warp_mi[w], cutlass.Float32(0.0))

            for offset in (16, 8, 4, 2, 1):
                mr_val += cute.arch.shuffle_sync_bfly(
                    mr_val, offset=offset, mask=-1, mask_and_clamp=31
                )
                mi_val += cute.arch.shuffle_sync_bfly(
                    mi_val, offset=offset, mask=-1, mask_and_clamp=31
                )

            if lane == cutlass.Int32(0):
                out = dm_flat[bh, c, None]
                frg_out = cute.make_rmem_tensor_like(out, dtype=cutlass.Float32)
                frg_out[0] = mr_val
                frg_out[1] = mi_val
                out.store(frg_out.load())


def _compiled_key(
    chunk_starts: torch.Tensor,
    d_inc: torch.Tensor,
) -> _CompiledKey:
    device_index = (
        0 if chunk_starts.device.index is None else int(chunk_starts.device.index)
    )
    return (
        device_index,
        chunk_starts.dtype,
        d_inc.dtype,
        tuple(int(x) for x in chunk_starts.shape),
        tuple(int(x) for x in chunk_starts.stride()),
    )


def _get_compiled_m_kernel(
    chunk_starts: torch.Tensor,
    d_inc: torch.Tensor,
    d_m_chunk: torch.Tensor,
) -> Callable[..., object]:
    key = _compiled_key(chunk_starts, d_inc)
    compiled = _COMPILED_STATE_PASSING_BWD_M.get(key)
    if compiled is not None:
        return compiled

    _, _, _, P, D = map(int, chunk_starts.shape)
    S = P * D
    cfg = _TileConfig()

    copy_bits_starts = _choose_copy_bits_for_linear_tiles(
        chunk_starts,
        tile_stride_elems=S,
        elems_per_thread=cfg.elems_per_thread,
    )
    copy_bits_dinc = _choose_copy_bits_for_linear_tiles(
        d_inc,
        tile_stride_elems=S,
        elems_per_thread=cfg.elems_per_thread,
    )

    m_starts = from_dlpack(
        chunk_starts,
        assumed_align=max(chunk_starts.element_size(), copy_bits_starts // 8),
    )
    m_dinc = from_dlpack(
        d_inc,
        assumed_align=max(d_inc.element_size(), copy_bits_dinc // 8),
    )
    m_dm = from_dlpack(d_m_chunk, assumed_align=max(d_m_chunk.element_size(), 8))

    kernel = _StatePassingBwdMAmpere(
        cfg,
        copy_bits_starts=copy_bits_starts,
        copy_bits_dinc=copy_bits_dinc,
    )
    compiled = cute.compile(kernel, m_starts, m_dinc, m_dm)
    _COMPILED_STATE_PASSING_BWD_M[key] = compiled
    return compiled


def state_passing_bwd_m_cute(
    chunk_starts: torch.Tensor,
    d_inc: torch.Tensor,
) -> torch.Tensor:
    """Compute ``d_m_chunk`` for the chunk recurrence in fp32."""
    if chunk_starts.device.type != "cuda":
        raise ValueError("CuTe state_passing backward requires CUDA tensors.")
    if d_inc.dtype != torch.float32:
        raise ValueError("state_passing_bwd_m_cute expects fp32 d_inc.")
    if chunk_starts.dtype not in (torch.float16, torch.float32):
        raise ValueError("state_passing_bwd_m_cute expects fp16/fp32 chunk_starts.")
    if chunk_starts.ndim != 5 or d_inc.ndim != 5:
        raise ValueError("Invalid tensor ranks for state_passing backward inputs.")
    if chunk_starts.shape != d_inc.shape:
        raise ValueError(
            "chunk_starts and d_inc must have identical (B,H,C,P,D) shapes."
        )
    if chunk_starts.shape[-1] % 2 != 0:
        raise ValueError(
            "The flattened D dimension must be even (interleaved complex pairs)."
        )

    chunk_starts_c = chunk_starts.contiguous()
    d_inc_c = d_inc.contiguous()
    B, H, C, _, _ = map(int, chunk_starts_c.shape)
    d_m_chunk = torch.empty(
        (B, H, C, 2), device=chunk_starts.device, dtype=torch.float32
    )

    compiled = _get_compiled_m_kernel(chunk_starts_c, d_inc_c, d_m_chunk)
    compiled(
        from_dlpack(chunk_starts_c, assumed_align=chunk_starts_c.element_size()),
        from_dlpack(d_inc_c, assumed_align=d_inc_c.element_size()),
        from_dlpack(d_m_chunk, assumed_align=max(d_m_chunk.element_size(), 8)),
    )
    return d_m_chunk


__all__ = ["state_passing_bwd_m_cute"]
