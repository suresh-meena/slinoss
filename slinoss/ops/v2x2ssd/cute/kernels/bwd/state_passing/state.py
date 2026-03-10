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
    torch.dtype,
    tuple[int, int, int, int, int],
    tuple[int, int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
]
_COMPILED_STATE_PASSING_BWD_STATE: dict[_CompiledKey, Callable[..., object]] = {}


class _StatePassingBwdStateAmpere:
    """Backward kernel for ``(d_inc, d_initial)``.

    Logical shapes:
    - ``d_chunk_starts``: ``(B, H, C, P, D)``
    - ``d_final``: ``(B, H, P, D)``
    - ``m_chunk``: ``(B, H, C, 2)``
    - outputs ``d_inc``: ``(B, H, C, P, D)``, ``d_initial``: ``(B, H, P, D)``

    Layout / launch:
    - flatten ``S = P * D`` as the contiguous axis
    - grid ``(ceil_div(S, tile), B * H, 1)``
    - each CTA owns one contiguous ``S`` tile for one ``(batch, head)`` row
    - each thread owns ``pairs_per_thread`` whole ``(re, im)`` pairs

    Numerical contract:
    - the transpose action is the real 2x2 transpose of ``m_chunk``, which is
      equivalent to multiplying by ``conj(m_chunk)`` on each complex pair
    - compute and outputs are fp32
    """

    def __init__(
        self,
        cfg: _TileConfig,
        *,
        copy_bits_in: int,
        copy_bits_out: int,
    ) -> None:
        self.cfg = cfg
        self.copy_bits_in = int(copy_bits_in)
        self.copy_bits_out = int(copy_bits_out)

    @cute.jit
    def __call__(
        self,
        d_chunk_starts: cute.Tensor,
        d_final: cute.Tensor,
        m_chunk: cute.Tensor,
        d_inc: cute.Tensor,
        d_initial: cute.Tensor,
    ) -> None:
        B, H, C, P, D = d_inc.shape
        BH = B * H
        S = P * D

        layout_bcs = cute.make_layout((BH, C, S), stride=(C * S, S, 1))
        layout_bs = cute.make_layout((BH, S), stride=(S, 1))
        layout_bcm = cute.make_layout((BH, C, 2), stride=(C * 2, 2, 1))

        dstarts_flat = cute.make_tensor(d_chunk_starts.iterator, layout_bcs)
        dfinal_flat = cute.make_tensor(d_final.iterator, layout_bs)
        m_flat = cute.make_tensor(m_chunk.iterator, layout_bcm)
        dinc_flat = cute.make_tensor(d_inc.iterator, layout_bcs)
        dinitial_flat = cute.make_tensor(d_initial.iterator, layout_bs)

        tv_layout = cute.make_layout(
            (self.cfg.num_threads, self.cfg.elems_per_thread),
            stride=(self.cfg.elems_per_thread, 1),
        )
        idS = cute.make_identity_tensor(S)
        cS = cute.zipped_divide(idS, tiler=cute.make_layout(self.cfg.tile))

        self.kernel(
            dstarts_flat,
            dfinal_flat,
            m_flat,
            dinc_flat,
            dinitial_flat,
            cS,
            tv_layout,
        ).launch(
            grid=[cute.ceil_div(S, self.cfg.tile), BH, 1],
            block=[self.cfg.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        dstarts_flat: cute.Tensor,
        dfinal_flat: cute.Tensor,
        m_flat: cute.Tensor,
        dinc_flat: cute.Tensor,
        dinitial_flat: cute.Tensor,
        cS: cute.Tensor,
        tv_layout: cute.Layout,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        tile_idx, bh, _ = cute.arch.block_idx()

        S = dinc_flat.shape[2]
        C = dinc_flat.shape[1]

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

        copy_in_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dstarts_flat.element_type,
            num_bits_per_copy=self.copy_bits_in,
        )
        copy_in_scalar = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dstarts_flat.element_type,
            num_bits_per_copy=dstarts_flat.element_type.width,
        )
        copy_out_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dinc_flat.element_type,
            num_bits_per_copy=self.copy_bits_out,
        )
        copy_out_scalar = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dinc_flat.element_type,
            num_bits_per_copy=dinc_flat.element_type.width,
        )
        copy_df_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dfinal_flat.element_type,
            num_bits_per_copy=self.copy_bits_in,
        )
        copy_df_scalar = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dfinal_flat.element_type,
            num_bits_per_copy=dfinal_flat.element_type.width,
        )
        copy_m = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            m_flat.element_type,
            num_bits_per_copy=m_flat.element_type.width * 2,
        )

        tile_layout = cute.make_layout(cS.shape[0])

        g_final = dfinal_flat[bh, None]
        t_final = cute.zipped_divide(g_final, tiler=tile_layout)
        cta_final = t_final[cta_coord]
        tid_final = cute.composition(cta_final, tv_layout)
        thr_final = tid_final[tidx, None]

        acc_g = cute.make_rmem_tensor(thr_final.shape, cutlass.Float32)
        acc_g.fill(0.0)
        frg_final = cute.make_rmem_tensor_like(thr_final)
        frg_final.fill(0)
        if is_partial_tile:
            cute.copy(copy_df_scalar, thr_final, frg_final, pred=frg_pred)
        else:
            cute.copy(copy_df_vec, thr_final, frg_final)
        acc_g.store(frg_final.load().to(cutlass.Float32))

        frg_in = cute.make_rmem_tensor_like(thr_final)
        frg_out = cute.make_rmem_tensor(thr_final.shape, dinc_flat.element_type)
        pairs_per_thread = cute.size(acc_g) // 2

        for c_it in cutlass.range(C, unroll=1):
            c = C - 1 - c_it

            g_out = dinc_flat[bh, c, None]
            t_out = cute.zipped_divide(g_out, tiler=tile_layout)
            cta_out = t_out[cta_coord]
            tid_out = cute.composition(cta_out, tv_layout)
            thr_out = tid_out[tidx, None]

            frg_out.store(acc_g.load().to(dinc_flat.element_type))
            if is_partial_tile:
                cute.copy(copy_out_scalar, frg_out, thr_out, pred=frg_pred)
            else:
                cute.copy(copy_out_vec, frg_out, thr_out)

            g_in = dstarts_flat[bh, c, None]
            t_in = cute.zipped_divide(g_in, tiler=tile_layout)
            cta_in = t_in[cta_coord]
            tid_in = cute.composition(cta_in, tv_layout)
            thr_in = tid_in[tidx, None]

            frg_in.fill(0)
            if is_partial_tile:
                cute.copy(copy_in_scalar, thr_in, frg_in, pred=frg_pred)
            else:
                cute.copy(copy_in_vec, thr_in, frg_in)
            dstart_f32 = frg_in.load().to(cutlass.Float32)

            g_m = m_flat[bh, c, None]
            frg_m = cute.make_rmem_tensor_like(g_m)
            cute.copy(copy_m, g_m, frg_m)
            m_val = frg_m.load().to(cutlass.Float32)
            mr, mi = m_val[0], m_val[1]

            for v in cutlass.range_constexpr(pairs_per_thread):
                base = v * 2
                gr = acc_g[base + 0]
                gi = acc_g[base + 1]

                acc_g[base + 0] = (mr * gr + mi * gi) + dstart_f32[base + 0]
                acc_g[base + 1] = (-mi * gr + mr * gi) + dstart_f32[base + 1]

        g_initial = dinitial_flat[bh, None]
        t_initial = cute.zipped_divide(g_initial, tiler=tile_layout)
        cta_initial = t_initial[cta_coord]
        tid_initial = cute.composition(cta_initial, tv_layout)
        thr_initial = tid_initial[tidx, None]

        frg_out.store(acc_g.load().to(dinitial_flat.element_type))
        if is_partial_tile:
            cute.copy(copy_out_scalar, frg_out, thr_initial, pred=frg_pred)
        else:
            cute.copy(copy_out_vec, frg_out, thr_initial)


def _compiled_key(
    d_chunk_starts: torch.Tensor,
    d_final: torch.Tensor,
    m_chunk: torch.Tensor,
    d_inc: torch.Tensor,
    d_initial: torch.Tensor,
) -> _CompiledKey:
    device_index = (
        0 if d_chunk_starts.device.index is None else int(d_chunk_starts.device.index)
    )
    return (
        device_index,
        d_chunk_starts.dtype,
        d_final.dtype,
        d_inc.dtype,
        tuple(int(x) for x in d_chunk_starts.shape),
        tuple(int(x) for x in d_chunk_starts.stride()),
        tuple(int(x) for x in m_chunk.shape),
        tuple(int(x) for x in m_chunk.stride()),
    )


def _get_compiled_state_kernel(
    d_chunk_starts: torch.Tensor,
    d_final: torch.Tensor,
    m_chunk: torch.Tensor,
    d_inc: torch.Tensor,
    d_initial: torch.Tensor,
) -> Callable[..., object]:
    key = _compiled_key(d_chunk_starts, d_final, m_chunk, d_inc, d_initial)
    compiled = _COMPILED_STATE_PASSING_BWD_STATE.get(key)
    if compiled is not None:
        return compiled

    _, _, _, P, D = map(int, d_chunk_starts.shape)
    S = P * D
    cfg = _TileConfig()

    copy_bits_in = _choose_copy_bits_for_linear_tiles(
        d_chunk_starts,
        tile_stride_elems=S,
        elems_per_thread=cfg.elems_per_thread,
    )
    copy_bits_out = _choose_copy_bits_for_linear_tiles(
        d_inc,
        tile_stride_elems=S,
        elems_per_thread=cfg.elems_per_thread,
    )

    m_dstarts = from_dlpack(
        d_chunk_starts,
        assumed_align=max(d_chunk_starts.element_size(), copy_bits_in // 8),
    )
    m_dfinal = from_dlpack(
        d_final,
        assumed_align=max(d_final.element_size(), copy_bits_in // 8),
    )
    m_m = from_dlpack(m_chunk, assumed_align=max(m_chunk.element_size(), 8))
    m_dinc = from_dlpack(
        d_inc,
        assumed_align=max(d_inc.element_size(), copy_bits_out // 8),
    )
    m_dinitial = from_dlpack(
        d_initial,
        assumed_align=max(d_initial.element_size(), copy_bits_out // 8),
    )

    kernel = _StatePassingBwdStateAmpere(
        cfg,
        copy_bits_in=copy_bits_in,
        copy_bits_out=copy_bits_out,
    )
    compiled = cute.compile(kernel, m_dstarts, m_dfinal, m_m, m_dinc, m_dinitial)
    _COMPILED_STATE_PASSING_BWD_STATE[key] = compiled
    return compiled


def state_passing_bwd_state_cute(
    d_chunk_starts: torch.Tensor,
    d_final: torch.Tensor,
    m_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(d_inc, d_initial)`` for the chunk recurrence in fp32."""
    if d_chunk_starts.device.type != "cuda":
        raise ValueError("CuTe state_passing backward requires CUDA tensors.")
    if d_chunk_starts.dtype != torch.float32 or d_final.dtype != torch.float32:
        raise ValueError("state_passing_bwd_state_cute expects fp32 upstream grads.")
    if m_chunk.dtype != torch.float32:
        raise ValueError("state_passing_bwd_state_cute expects fp32 m_chunk.")
    if d_chunk_starts.ndim != 5 or d_final.ndim != 4 or m_chunk.ndim != 4:
        raise ValueError("Invalid tensor ranks for state_passing backward inputs.")
    if d_chunk_starts.shape[:3] != m_chunk.shape[:3]:
        raise ValueError(
            "Leading (B,H,C) dims of d_chunk_starts and m_chunk must match."
        )
    if d_chunk_starts.shape[:2] != d_final.shape[:2]:
        raise ValueError("Leading (B,H) dims of d_chunk_starts and d_final must match.")
    if d_chunk_starts.shape[-1] % 2 != 0:
        raise ValueError(
            "The flattened D dimension must be even (interleaved complex pairs)."
        )

    B, H, C, P, D = map(int, d_chunk_starts.shape)
    if d_final.shape != (B, H, P, D):
        raise ValueError(f"d_final must be {(B, H, P, D)}. Got {tuple(d_final.shape)}.")
    if m_chunk.shape != (B, H, C, 2):
        raise ValueError(f"m_chunk must be {(B, H, C, 2)}. Got {tuple(m_chunk.shape)}.")

    d_chunk_starts_c = d_chunk_starts.contiguous()
    d_final_c = d_final.contiguous()
    m_chunk_c = m_chunk.contiguous()

    d_inc = torch.empty_like(d_chunk_starts_c)
    d_initial = torch.empty_like(d_final_c)

    compiled = _get_compiled_state_kernel(
        d_chunk_starts_c,
        d_final_c,
        m_chunk_c,
        d_inc,
        d_initial,
    )

    compiled(
        from_dlpack(d_chunk_starts_c, assumed_align=d_chunk_starts_c.element_size()),
        from_dlpack(d_final_c, assumed_align=d_final_c.element_size()),
        from_dlpack(m_chunk_c, assumed_align=max(m_chunk_c.element_size(), 8)),
        from_dlpack(d_inc, assumed_align=d_inc.element_size()),
        from_dlpack(d_initial, assumed_align=d_initial.element_size()),
    )
    return d_inc, d_initial


__all__ = ["state_passing_bwd_state_cute"]
