"""CuTe forward kernel for the v2x2ssd state-passing stage.

This mirrors the structure of ``v3x3ssd.cute.kernels.fwd.state_passing`` while
adapting the transport from scaled quaternions to packed complex scalars.

The algorithm is bandwidth-oriented:
  - sequential over chunks,
  - parallel over the flattened state axis ``S = P * D`` where ``D = 2N``.

Per chunk c:
  - write ``chunk_starts[..., c, :, :] = z`` (the pre-update state),
  - update ``z = m_chunk[..., c] * z + inc[..., c, :, :]``.

Outputs are float32, matching the reference path.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute


class StatePassingFwdAmpere:
    """Ampere state-passing kernel (CopyUniversalOp + fp32 math)."""

    def __init__(
        self,
        *,
        num_threads: int = 128,
        vecs_per_thread: int = 8,
        copy_bits_in: int,
        copy_bits_out: int,
        has_init: bool,
    ):
        self.num_threads = int(num_threads)
        self.vecs_per_thread = int(vecs_per_thread)
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive.")
        if self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32.")
        if self.vecs_per_thread <= 0:
            raise ValueError("vecs_per_thread must be positive.")

        self.elems_per_thread = 2 * self.vecs_per_thread
        self.tile = self.num_threads * self.elems_per_thread

        self.copy_bits_in = int(copy_bits_in)
        self.copy_bits_out = int(copy_bits_out)
        self.has_init = bool(has_init)

    @cute.jit
    def __call__(
        self,
        inc: cute.Tensor,  # (B,H,C,P,D)
        m_chunk: cute.Tensor,  # (B,H,C,2)
        out_starts: cute.Tensor,  # (B,H,C,P,D) fp32
        out_final: cute.Tensor,  # (B,H,P,D) fp32
        init_or_dummy: cute.Tensor,  # (B,H,P,D) or ignored when has_init=False
    ):
        B, H, C, P, D = inc.shape
        BH = B * H
        S = P * D

        layout_bcs = cute.make_layout((BH, C, S), stride=(C * S, S, 1))
        layout_bcm = cute.make_layout((BH, C, 2), stride=(C * 2, 2, 1))
        layout_bs = cute.make_layout((BH, S), stride=(S, 1))

        inc_flat = cute.make_tensor(inc.iterator, layout_bcs)
        m_flat = cute.make_tensor(m_chunk.iterator, layout_bcm)
        out_starts_flat = cute.make_tensor(out_starts.iterator, layout_bcs)
        out_final_flat = cute.make_tensor(out_final.iterator, layout_bs)
        init_flat = cute.make_tensor(init_or_dummy.iterator, layout_bs)

        tv_layout = cute.make_layout(
            (self.num_threads, self.elems_per_thread),
            stride=(self.elems_per_thread, 1),
        )

        idS = cute.make_identity_tensor(S)
        cS = cute.zipped_divide(idS, tiler=cute.make_layout(self.tile))

        grid_x = cute.ceil_div(S, self.tile)
        grid_y = BH

        self.kernel(
            inc_flat,
            m_flat,
            out_starts_flat,
            out_final_flat,
            init_flat,
            cS,
            tv_layout,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        inc_flat: cute.Tensor,  # (BH, C, S)
        m_flat: cute.Tensor,  # (BH, C, 2)
        out_starts_flat: cute.Tensor,  # (BH, C, S) fp32
        out_final_flat: cute.Tensor,  # (BH, S) fp32
        init_flat: cute.Tensor,  # (BH, S)
        cS: cute.Tensor,  # (tile, ntiles)
        tv_layout: cute.Layout,  # (tid, vid) -> linear coord in [0, tile)
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        bh = bidy
        tile_idx = bidx

        S = inc_flat.shape[2]
        C = inc_flat.shape[1]

        tile_start = cutlass.Int32(self.tile) * tile_idx
        residue = S - tile_start
        is_partial_tile = cute.elem_less(residue, cutlass.Int32(self.tile))

        cta_coord = (None, tile_idx)
        ctaCrd = cS[cta_coord]
        tidCrd = cute.composition(ctaCrd, tv_layout)
        thrCrd = tidCrd[tidx, None]

        frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
        frgPred.fill(cutlass.Boolean(True))
        if is_partial_tile:
            for i in cutlass.range_constexpr(cute.size(frgPred)):
                frgPred[i] = cute.elem_less(thrCrd[i], S)

        copy_in_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            inc_flat.element_type,
            num_bits_per_copy=self.copy_bits_in,
        )
        copy_in_scalar = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            inc_flat.element_type,
            num_bits_per_copy=inc_flat.element_type.width,
        )
        copy_out_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            out_starts_flat.element_type,
            num_bits_per_copy=self.copy_bits_out,
        )
        copy_out_scalar = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            out_starts_flat.element_type,
            num_bits_per_copy=out_starts_flat.element_type.width,
        )
        copy_m = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            m_flat.element_type,
            num_bits_per_copy=m_flat.element_type.width * 2,
        )
        copy_init = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            init_flat.element_type,
            num_bits_per_copy=init_flat.element_type.width,
        )

        tile_layout = cute.make_layout(cS.shape[0])

        gInit = init_flat[bh, None]
        tInit = cute.zipped_divide(gInit, tiler=tile_layout)
        ctaInit = tInit[cta_coord]
        tidInit = cute.composition(ctaInit, tv_layout)
        thrInit = tidInit[tidx, None]

        accZ = cute.make_rmem_tensor(thrInit.shape, cutlass.Float32)
        accZ.fill(0.0)
        if cutlass.const_expr(self.has_init):
            frgInit = cute.make_rmem_tensor_like(thrInit)
            frgInit.fill(0)
            if is_partial_tile:
                cute.copy(copy_init, thrInit, frgInit, pred=frgPred)
            else:
                cute.copy(copy_init, thrInit, frgInit)
            accZ.store(frgInit.load().to(cutlass.Float32))

        frgIn = cute.make_rmem_tensor_like(thrInit)
        frgOut = cute.make_rmem_tensor(thrInit.shape, cutlass.Float32)

        pairs_per_thread = cute.size(accZ) // 2

        for c in cutlass.range(C, unroll=1):
            gOut = out_starts_flat[bh, c, None]
            tOut = cute.zipped_divide(gOut, tiler=tile_layout)
            ctaOut = tOut[cta_coord]
            tidOut = cute.composition(ctaOut, tv_layout)
            thrOut = tidOut[tidx, None]

            frgOut.store(accZ.load())
            if is_partial_tile:
                cute.copy(copy_out_scalar, frgOut, thrOut, pred=frgPred)
            else:
                cute.copy(copy_out_vec, frgOut, thrOut)

            gInc = inc_flat[bh, c, None]
            tInc = cute.zipped_divide(gInc, tiler=tile_layout)
            ctaInc = tInc[cta_coord]
            tidInc = cute.composition(ctaInc, tv_layout)
            thrInc = tidInc[tidx, None]

            frgIn.fill(0)
            if is_partial_tile:
                cute.copy(copy_in_scalar, thrInc, frgIn, pred=frgPred)
            else:
                cute.copy(copy_in_vec, thrInc, frgIn)
            inc_f32 = frgIn.load().to(cutlass.Float32)

            gM = m_flat[bh, c, None]
            frgM = cute.make_rmem_tensor_like(gM)
            cute.copy(copy_m, gM, frgM)
            m = frgM.load().to(cutlass.Float32)
            mr, mi = m[0], m[1]

            for v in cutlass.range_constexpr(pairs_per_thread):
                base = v * 2
                zr = accZ[base + 0]
                zi = accZ[base + 1]

                rr = mr * zr - mi * zi
                ri = mr * zi + mi * zr

                accZ[base + 0] = rr + inc_f32[base + 0]
                accZ[base + 1] = ri + inc_f32[base + 1]

        gF = out_final_flat[bh, None]
        tF = cute.zipped_divide(gF, tiler=tile_layout)
        ctaF = tF[cta_coord]
        tidF = cute.composition(ctaF, tv_layout)
        thrF = tidF[tidx, None]

        frgOut.store(accZ.load())
        if is_partial_tile:
            cute.copy(copy_out_scalar, frgOut, thrF, pred=frgPred)
        else:
            cute.copy(copy_out_vec, frgOut, thrF)

        return
