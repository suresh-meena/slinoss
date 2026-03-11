"""CuTe backward slice for ``chunk_scan`` gradients into ``U`` and ``U_prev``.

Logical contract
----------------
This slice consumes cached reverse-time packed tensors instead of the raw
public ``U/M/K/B/C`` inputs:

- ``Q_rev``: ``flip(Q, dim=1)``, shape ``(BHC, L, 1, D)``
- ``Kprev_rev``: ``flip(Kprev, dim=1)``, shape ``(BHC, L, 1, D)``
- ``Kcurr_rev``: ``flip(Kcurr, dim=1)``, shape ``(BHC, L, 1, D)``
- ``neg_logprefix_half_rev``: ``-flip(logprefix_half, dim=1)``, shape ``(BHC, L)``
- ``d_out``: ``(B, H, T, P)``

Why this contract
-----------------
The packed-real value gradient is another causal attention-like pass after:

- reversing time,
- moving the forward packed key rows into the query role,
- using reversed ``d_out`` as the value vectors,
- and reusing the reversed/negated half-logprefix metadata.

The old path abused the forward inner kernel twice and then finished the real
work with a second full-width scatter over the intermediate ``dV`` buffers.
This file closes that engineering gap:

- a dedicated reverse-time tensor-core workhorse writes public ``dU`` rows
  directly,
- only the single cross-chunk boundary row is handled by a tiny CuTe epilogue,
  with no Python-side accumulation pass and no full-width post-scatter.

Numerical contract
------------------
This path is intentionally approximate in the same principled way as ``v3``:

- transport uses fp16/bf16 on tensor-core-friendly packed operands,
- the two dense contractions accumulate in fp32,
- the per-score scale is formed directly from the cached logprefix values, with
  no reciprocal factorization and no ``0 * inf`` pattern,
- the public writeout and boundary epilogue run in fp32.

That is the right contract here. This slice is not on the exact autograd path,
so the goal is a numerically sane tensor-core approximation, not bitwise parity
with the full fp32 oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack


LOG2_E = 1.4426950408889634

_CompiledRawKey = tuple[
    int,
    torch.dtype,
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int],
    tuple[int, int, int],
    tuple[int, int],
    int,
]
_CompiledBoundaryKey = tuple[
    int,
    tuple[int, int],
    tuple[int, int],
    tuple[int, int, int],
    int,
    int,
]
_ScratchKey = tuple[int, torch.dtype, int, int, int, int]

_COMPILED_DU_RAW: dict[_CompiledRawKey, object] = {}
_COMPILED_DU_BOUNDARY: dict[_CompiledBoundaryKey, object] = {}


@dataclass
class _ChunkScanBwdDUScratch:
    dU_pad: torch.Tensor
    dU_prev: torch.Tensor
    boundary: torch.Tensor


_SCRATCH_DU: dict[_ScratchKey, _ChunkScanBwdDUScratch] = {}


def _choose_feat_tile(P: int) -> int:
    return 32 if P >= 32 else 16


@dataclass(frozen=True)
class _ChunkScanBwdDURawConfig:
    D: int
    P: int
    L: int
    tile: int = 32
    num_threads: int = 128

    def __post_init__(self) -> None:
        if self.tile != 32:
            raise ValueError("The current DU raw kernel expects tile=32.")
        if self.L % self.tile != 0:
            raise ValueError("L must be divisible by 32.")
        if self.num_threads != 128:
            raise ValueError("The current DU raw kernel expects 128 threads.")

    @property
    def D_padded(self) -> int:
        return ((self.D + 31) // 32) * 32

    @property
    def P_padded(self) -> int:
        return ((self.P + 31) // 32) * 32


class _ChunkScanBwdDURawAmpereTc:
    """Ampere tensor-core workhorse for packed reverse-time ``dU``.

    Logical tensors
    ---------------
    - ``mQueryPrev/mQueryCurr``: ``(BHC, L, 1, D)``, low-precision branch query rows
    - ``mKey``: ``(BHC, L, 1, D)``, low-precision shared key rows
    - ``mDOut``: ``(BHC, L, 1, P)``, low-precision reversed ``d_out``
    - ``mLogprefix``: ``(BHC, L)``, fp32 cached half-logprefix in reverse time
    - ``mDUPad``: ``(BH, T_pad, P)``, fp32 public value gradient
    - ``mBoundary``: ``(BHC, P)``, fp32 chunk-boundary carry rows

    Layout / launch contract
    ------------------------
    - One CTA owns one ``(feat_tile, bhc)`` output stripe and loops over every
      source tile in the chunk.
    - ``n`` is the fixed source tile and ``m`` is the current target tile;
      both are 32 so the dense core is a clean ``32x32`` tensor-core block.
    - Keys and queries use swizzled shared-memory ``D`` tiles.
    - ``d_out`` uses swizzled shared-memory ``P`` tiles and is only staged once
      per target tile.
    - The scaled causal score block is spilled to shared and reloaded through
      the tensor-core A path, matching the stable ``v3`` pattern.

    Correctness invariants
    ----------------------
    - ``L`` must be a multiple of 32.
    - ``mBoundary`` stores the packed ``dV_prev`` boundary row for each chunk;
      a tiny epilogue uses that to patch the single cross-chunk public row.
    """

    def __init__(self, dtype: type[cutlass.Numeric], cfg: _ChunkScanBwdDURawConfig):
        self.cfg = cfg
        self.ab_dtype = dtype
        self.acc_dtype = cutlass.Float32
        self.mma_inst_shape = (16, 8, 16)
        self.warp_layout_mnk = (2, 2, 1)

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
                (acc_layout_col_major.shape[0][0], acc_layout_col_major.shape[2]),
            ),
            stride=(
                (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
                (acc_layout_col_major.stride[0][0], acc_layout_col_major.stride[2]),
            ),
        )
        return cute.make_tensor(
            acc.iterator,
            cute.composition(acc.layout, acc_layout_mn),
        )

    @cute.jit
    def __call__(
        self,
        mQueryPrev: cute.Tensor,
        mQueryCurr: cute.Tensor,
        mKey: cute.Tensor,
        mDOut: cute.Tensor,
        mLogprefix: cute.Tensor,
        mDUPad: cute.Tensor,
        mBoundary: cute.Tensor,
        n_chunks: cutlass.Int32,
    ) -> None:
        if cutlass.const_expr(
            not (
                mQueryPrev.element_type
                == mQueryCurr.element_type
                == mKey.element_type
                == mDOut.element_type
                == self.ab_dtype
            )
        ):
            raise TypeError("Query/Key/DOut must share the tensor-core transport dtype.")
        if cutlass.const_expr(
            not (
                self.ab_dtype == cutlass.Float16 or self.ab_dtype == cutlass.BFloat16
            )
        ):
            raise TypeError("DU raw kernel supports only Float16/BFloat16 inputs.")
        if cutlass.const_expr(
            mLogprefix.element_type != cutlass.Float32
            or mDUPad.element_type != cutlass.Float32
            or mBoundary.element_type != cutlass.Float32
        ):
            raise TypeError("logprefix and output must be Float32.")
        if cutlass.const_expr(
            mQueryPrev.shape[2] != 1
            or mQueryCurr.shape[2] != 1
            or mKey.shape[2] != 1
            or mDOut.shape[2] != 1
        ):
            raise ValueError("Packed DU tensors must have singleton dim2.")

        Dp = self.cfg.D_padded
        Pp = self.cfg.P_padded
        n = self.cfg.tile
        m = self.cfg.tile
        p = self.cfg.tile

        smem_k_block_size_D = 64 if Dp % 64 == 0 else 32
        swizzle_bits_D = 3 if smem_k_block_size_D == 64 else 2
        sD_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_D, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_D), stride=(smem_k_block_size_D, 1)),
        )
        sK_layout = cute.tile_to_shape(sD_layout_atom, (n, Dp), (0, 1))
        sQ_layout = cute.tile_to_shape(sD_layout_atom, (m, Dp), (0, 1))

        sP_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(2, 3, 3),
            0,
            cute.make_layout((8, p), stride=(p, 1)),
        )
        sDY_layout = cute.tile_to_shape(sP_layout_atom, (m, p), (0, 1))

        sBlk_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, n), stride=(n, 1)),
        )
        sS_layout = cute.tile_to_shape(sBlk_layout_atom, (n, m), (0, 1))
        sQlog_layout = cute.make_layout((m,), stride=(1,))
        sKlog_layout = cute.make_layout((n,), stride=(1,))
        sPrev_layout = cute.make_layout((n, p), stride=(p, 1))
        sCarry_layout = cute.make_layout((p,), stride=(1,))

        universal_copy_bits = 128
        async_elems_in = universal_copy_bits // mKey.element_type.width
        atom_async_copy_in = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mKey.element_type,
            num_bits_per_copy=universal_copy_bits,
        )
        tD_shape_dim_1 = sD_layout_atom.outer.shape[1] // async_elems_in
        tD_layout = cute.make_layout(
            (self.cfg.num_threads // tD_shape_dim_1, tD_shape_dim_1),
            stride=(tD_shape_dim_1, 1),
        )
        v_in_layout = cute.make_layout((1, async_elems_in))
        gmem_tiled_copy_D = cute.make_tiled_copy_tv(
            atom_async_copy_in, tD_layout, v_in_layout
        )

        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.mma_inst_shape
        )
        perm = (
            self.warp_layout_mnk[0] * self.mma_inst_shape[0],
            self.warp_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.warp_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            op,
            cute.make_layout(self.warp_layout_mnk),
            permutation_mnk=perm,
        )

        smem_size = 0
        smem_size += cute.size_in_bytes(self.ab_dtype, sK_layout)
        smem_size += cute.size_in_bytes(self.ab_dtype, sQ_layout)
        smem_size += cute.size_in_bytes(self.ab_dtype, sDY_layout)
        smem_size += cute.size_in_bytes(self.ab_dtype, sS_layout)
        smem_size += cute.size_in_bytes(cutlass.Float32, sQlog_layout)
        smem_size += cute.size_in_bytes(cutlass.Float32, sKlog_layout)
        smem_size += cute.size_in_bytes(cutlass.Float32, sPrev_layout)
        smem_size += cute.size_in_bytes(cutlass.Float32, sCarry_layout)
        smem_size += 512

        grid_x = Pp // p
        grid_z = cute.size(mKey.shape[0])
        self.kernel(
            mQueryPrev,
            mQueryCurr,
            mKey,
            mDOut,
            mLogprefix,
            mDUPad,
            mBoundary,
            n_chunks,
            sK_layout,
            sQ_layout,
            sDY_layout,
            sS_layout,
            sQlog_layout,
            sKlog_layout,
            sPrev_layout,
            sCarry_layout,
            gmem_tiled_copy_D,
            tiled_mma,
        ).launch(
            grid=(grid_x, 1, grid_z),
            block=[self.cfg.num_threads, 1, 1],
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        mQueryPrev: cute.Tensor,
        mQueryCurr: cute.Tensor,
        mKey: cute.Tensor,
        mDOut: cute.Tensor,
        mLogprefix: cute.Tensor,
        mDUPad: cute.Tensor,
        mBoundary: cute.Tensor,
        n_chunks: cutlass.Int32,
        sK_layout: cute.ComposedLayout,
        sQ_layout: cute.ComposedLayout,
        sDY_layout: cute.ComposedLayout,
        sS_layout: cute.ComposedLayout,
        sQlog_layout: cute.Layout,
        sKlog_layout: cute.Layout,
        sPrev_layout: cute.Layout,
        sCarry_layout: cute.Layout,
        gmem_tiled_copy_D: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        p_block, _, bhc = cute.arch.block_idx()
        n = self.cfg.tile
        m = self.cfg.tile
        p = self.cfg.tile
        Dp = self.cfg.D_padded
        L = self.cfg.L
        n_tiles = L // n

        smem = utils.SmemAllocator()
        sK = smem.allocate_tensor(self.ab_dtype, sK_layout, 16)
        sQ = smem.allocate_tensor(self.ab_dtype, sQ_layout, 16)
        sDY = smem.allocate_tensor(self.ab_dtype, sDY_layout, 16)
        sS = smem.allocate_tensor(self.ab_dtype, sS_layout, 16)
        s_lp_q = smem.allocate_tensor(cutlass.Float32, sQlog_layout, 4)
        s_lp_k = smem.allocate_tensor(cutlass.Float32, sKlog_layout, 4)
        sPrev = smem.allocate_tensor(cutlass.Float32, sPrev_layout, 16)
        sCarry = smem.allocate_tensor(cutlass.Float32, sCarry_layout, 16)

        g_thr_D = gmem_tiled_copy_D.get_slice(tidx)

        thr_mma = tiled_mma.get_slice(tidx)
        tSrK = thr_mma.make_fragment_A(thr_mma.partition_A(sK))
        tSrQ = thr_mma.make_fragment_B(thr_mma.partition_B(sQ))
        sDYt = cute.composition(sDY, cute.make_layout((p, m), stride=(m, 1)))
        tSrDY = thr_mma.make_fragment_B(thr_mma.partition_B(sDYt))
        acc_shape_S = thr_mma.partition_shape_C((n, m))
        acc_shape_DV = thr_mma.partition_shape_C((n, p))
        acc_DV_prev = cute.make_rmem_tensor(acc_shape_DV, cutlass.Float32)
        acc_DV_prev.fill(0.0)
        acc_DV_curr = cute.make_rmem_tensor(acc_shape_DV, cutlass.Float32)
        acc_DV_curr.fill(0.0)
        mcS = cute.make_identity_tensor(
            (mKey.shape[0], mKey.shape[1], mKey.shape[2], mQueryPrev.shape[1])
        )

        smem_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.ab_dtype,
        )
        smem_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.ab_dtype,
        )
        smem_copy_atom_BT = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.ab_dtype,
        )
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_tiled_copy_BT = cute.make_tiled_copy_B(smem_copy_atom_BT, tiled_mma)
        thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        thr_copy_B = smem_tiled_copy_B.get_slice(tidx)
        thr_copy_BT = smem_tiled_copy_BT.get_slice(tidx)
        tSsK = thr_copy_A.partition_S(sK)
        tSrK_view = thr_copy_A.retile(tSrK)
        tSrS = thr_mma.make_fragment_A(thr_mma.partition_A(sS))
        tSsS = thr_copy_A.partition_S(sS)
        tSrS_view = thr_copy_A.retile(tSrS)
        tSsQ = thr_copy_B.partition_S(sQ)
        tSrQ_view = thr_copy_B.retile(tSrQ)
        tSsDYt = thr_copy_BT.partition_S(sDYt)
        tSrDY_view = thr_copy_BT.retile(tSrDY)
        bh = bhc // n_chunks
        chunk = bhc - bh * n_chunks
        p_base = p_block * p

        if tidx < cutlass.Int32(p):
            sCarry[tidx] = cutlass.Float32(0.0)
        cute.arch.barrier()

        mcKD = cute.make_identity_tensor(mKey.layout.shape)
        mcQD_prev = cute.make_identity_tensor(mQueryPrev.layout.shape)
        mcQD_curr = cute.make_identity_tensor(mQueryCurr.layout.shape)
        mcS = cute.make_identity_tensor(
            (mKey.shape[0], mKey.shape[1], mKey.shape[2], mQueryPrev.shape[1])
        )
        mcOut = cute.make_identity_tensor((1, L, 1, self.cfg.P_padded))

        for n_block in range(n_tiles):
            gK = cute.local_tile(mKey[bhc, None, 0, None], (n, Dp), (n_block, 0))
            tKg = g_thr_D.partition_S(gK)
            tKs = g_thr_D.partition_D(sK)
            cKD = cute.local_tile(mcKD[bhc, None, 0, None], (n, Dp), (n_block, 0))
            tKc = g_thr_D.partition_S(cKD)
            tKp = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tKs.shape[0][1],
                        cute.size(tKs, mode=[1]),
                        cute.size(tKs, mode=[2]),
                    ),
                    stride=(cute.size(tKs, mode=[2]), 0, 1),
                ),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range_constexpr(tKp.shape[0]):
                for vi in cutlass.range_constexpr(tKp.shape[1]):
                    for rest_k in cutlass.range_constexpr(tKp.shape[2]):
                        coord = tKc[(0, rest_v), vi, rest_k]
                        tKp[rest_v, vi, rest_k] = cute.elem_less(
                            coord[1], mKey.shape[1]
                        ) and cute.elem_less(coord[3], mKey.shape[3])
            for vi in cutlass.range_constexpr(cute.size(tKs.shape[1])):
                cute.copy(
                    gmem_tiled_copy_D,
                    tKg[None, vi, None],
                    tKs[None, vi, None],
                    pred=tKp[None, vi, None],
                )
            cute.arch.cp_async_commit_group()

            if tidx < cutlass.Int32(n):
                row = n_block * n + tidx
                s_lp_k[tidx] = cutlass.select_(
                    cute.elem_less(row, mLogprefix.shape[1]),
                    cutlass.Float32(mLogprefix[bhc, row]),
                    cutlass.Float32(0.0),
                )
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            acc_DV_prev = cute.make_rmem_tensor(acc_shape_DV, cutlass.Float32)
            acc_DV_prev.fill(0.0)
            acc_DV_curr = cute.make_rmem_tensor(acc_shape_DV, cutlass.Float32)
            acc_DV_curr.fill(0.0)

            cute.copy(smem_tiled_copy_A, tSsK[None, None, 0], tSrK_view[None, None, 0])
            for kk in cutlass.range_constexpr(cute.size(tSsK.shape[2])):
                kk_next = (kk + 1) % cute.size(tSsK.shape[2])
                cute.copy(
                    smem_tiled_copy_A,
                    tSsK[None, None, kk_next],
                    tSrK_view[None, None, kk_next],
                )

            for m_block in range(0, n_block + 1):
                gQ = cute.local_tile(
                    mQueryPrev[bhc, None, 0, None], (m, Dp), (m_block, 0)
                )
                tQg = g_thr_D.partition_S(gQ)
                tQs = g_thr_D.partition_D(sQ)
                cQD = cute.local_tile(
                    mcQD_prev[bhc, None, 0, None], (m, Dp), (m_block, 0)
                )
                tQc = g_thr_D.partition_S(cQD)
                for rest_v in cutlass.range_constexpr(tKp.shape[0]):
                    for vi in cutlass.range_constexpr(tKp.shape[1]):
                        for rest_k in cutlass.range_constexpr(tKp.shape[2]):
                            coord = tQc[(0, rest_v), vi, rest_k]
                            tKp[rest_v, vi, rest_k] = cute.elem_less(
                                coord[1], mQueryPrev.shape[1]
                            ) and cute.elem_less(coord[3], mQueryPrev.shape[3])
                for vi in cutlass.range_constexpr(cute.size(tQs.shape[1])):
                    cute.copy(
                        gmem_tiled_copy_D,
                        tQg[None, vi, None],
                        tQs[None, vi, None],
                        pred=tKp[None, vi, None],
                    )
                cute.arch.cp_async_commit_group()

                if tidx < cutlass.Int32(m):
                    row = m_block * m + tidx
                    s_lp_q[tidx] = cutlass.select_(
                        cute.elem_less(row, mLogprefix.shape[1]),
                        cutlass.Float32(mLogprefix[bhc, row]),
                        cutlass.Float32(0.0),
                    )

                idx_p = cutlass.Int32(tidx)
                total_dy = cutlass.Int32(m * p)
                while cute.elem_less(idx_p, total_dy):
                    rr = idx_p // p
                    cc = idx_p - rr * p
                    grow = m_block * m + rr
                    gcol = p_base + cc
                    val = cutlass.Float32(0.0)
                    if cute.elem_less(grow, mDOut.shape[1]) and cute.elem_less(
                        gcol, mDOut.shape[3]
                    ):
                        val = cutlass.Float32(mDOut[bhc, grow, 0, gcol])
                    sDY[rr, cc] = val.to(mDOut.element_type)
                    idx_p = idx_p + self.cfg.num_threads

                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
                acc_S.fill(0.0)
                cute.copy(
                    smem_tiled_copy_B, tSsQ[None, None, 0], tSrQ_view[None, None, 0]
                )
                for kk in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                    kk_next = (kk + 1) % cute.size(tSsQ.shape[2])
                    cute.copy(
                        smem_tiled_copy_B,
                        tSsQ[None, None, kk_next],
                        tSrQ_view[None, None, kk_next],
                    )
                    cute.gemm(
                        tiled_mma,
                        acc_S,
                        tSrK[None, None, kk],
                        tSrQ[None, None, kk],
                        acc_S,
                    )

                cS = cute.local_tile(mcS[bhc, None, 0, None], (n, m), (n_block, m_block))
                tScS = thr_mma.partition_C(cS)
                tScS_mn = self._make_acc_tensor_mn_view(tScS)
                acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
                for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
                    row_idx = cutlass.Int32(tScS_mn[r, 0][1])
                    for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                        col_idx = cutlass.Int32(tScS_mn[0, c][3])
                        val = cutlass.Float32(acc_S_mn[r, c])
                        src_scale = cutlass.Float32(1.0)
                        if cute.elem_less(row_idx, mKey.shape[1]):
                            src_scale = cute.math.exp2(
                                cutlass.Float32(2.0)
                                * cutlass.Float32(s_lp_k[row_idx - n_block * n])
                                * cutlass.Float32(LOG2_E),
                            )
                        if cute.elem_less(row_idx, col_idx) or cute.elem_less(
                            mQueryPrev.shape[1], col_idx + 1
                        ):
                            acc_S_mn[r, c] = cutlass.Float32(0.0)
                        else:
                            tgt_scale = cute.math.exp2(
                                -cutlass.Float32(2.0)
                                * cutlass.Float32(s_lp_q[col_idx - m_block * m])
                                * cutlass.Float32(LOG2_E),
                            )
                            acc_S_mn[r, c] = val * src_scale * tgt_scale
                        s_row = row_idx - n_block * n
                        s_col = col_idx - m_block * m
                        if cute.elem_less(s_row, n) and cute.elem_less(s_col, m):
                            sS[s_row, s_col] = acc_S_mn[r, c].to(self.ab_dtype)

                cute.arch.barrier()
                cute.copy(
                    smem_tiled_copy_A, tSsS[None, None, 0], tSrS_view[None, None, 0]
                )
                cute.copy(
                    smem_tiled_copy_BT,
                    tSsDYt[None, None, 0],
                    tSrDY_view[None, None, 0],
                )
                for kk in cutlass.range_constexpr(cute.size(tSrS.shape[2])):
                    kk_next = (kk + 1) % cute.size(tSrS.shape[2])
                    cute.copy(
                        smem_tiled_copy_A,
                        tSsS[None, None, kk_next],
                        tSrS_view[None, None, kk_next],
                    )
                    cute.copy(
                        smem_tiled_copy_BT,
                        tSsDYt[None, None, kk_next],
                        tSrDY_view[None, None, kk_next],
                    )
                    cute.gemm(
                        tiled_mma,
                        acc_DV_prev,
                        tSrS[None, None, kk],
                        tSrDY[None, None, kk],
                        acc_DV_prev,
                    )

                gQ = cute.local_tile(
                    mQueryCurr[bhc, None, 0, None], (m, Dp), (m_block, 0)
                )
                tQg = g_thr_D.partition_S(gQ)
                tQs = g_thr_D.partition_D(sQ)
                cQD = cute.local_tile(
                    mcQD_curr[bhc, None, 0, None], (m, Dp), (m_block, 0)
                )
                tQc = g_thr_D.partition_S(cQD)
                for rest_v in cutlass.range_constexpr(tKp.shape[0]):
                    for vi in cutlass.range_constexpr(tKp.shape[1]):
                        for rest_k in cutlass.range_constexpr(tKp.shape[2]):
                            coord = tQc[(0, rest_v), vi, rest_k]
                            tKp[rest_v, vi, rest_k] = cute.elem_less(
                                coord[1], mQueryCurr.shape[1]
                            ) and cute.elem_less(coord[3], mQueryCurr.shape[3])
                for vi in cutlass.range_constexpr(cute.size(tQs.shape[1])):
                    cute.copy(
                        gmem_tiled_copy_D,
                        tQg[None, vi, None],
                        tQs[None, vi, None],
                        pred=tKp[None, vi, None],
                    )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                acc_S.fill(0.0)
                cute.copy(
                    smem_tiled_copy_B, tSsQ[None, None, 0], tSrQ_view[None, None, 0]
                )
                for kk in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                    kk_next = (kk + 1) % cute.size(tSsQ.shape[2])
                    cute.copy(
                        smem_tiled_copy_B,
                        tSsQ[None, None, kk_next],
                        tSrQ_view[None, None, kk_next],
                    )
                    cute.gemm(
                        tiled_mma,
                        acc_S,
                        tSrK[None, None, kk],
                        tSrQ[None, None, kk],
                        acc_S,
                    )

                for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
                    row_idx = cutlass.Int32(tScS_mn[r, 0][1])
                    for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                        col_idx = cutlass.Int32(tScS_mn[0, c][3])
                        val = cutlass.Float32(acc_S_mn[r, c])
                        src_scale = cutlass.Float32(1.0)
                        if cute.elem_less(row_idx, mKey.shape[1]):
                            src_scale = cute.math.exp2(
                                cutlass.Float32(2.0)
                                * cutlass.Float32(s_lp_k[row_idx - n_block * n])
                                * cutlass.Float32(LOG2_E),
                            )
                        if cute.elem_less(row_idx, col_idx) or cute.elem_less(
                            mQueryCurr.shape[1], col_idx + 1
                        ):
                            acc_S_mn[r, c] = cutlass.Float32(0.0)
                        else:
                            tgt_scale = cute.math.exp2(
                                -cutlass.Float32(2.0)
                                * cutlass.Float32(s_lp_q[col_idx - m_block * m])
                                * cutlass.Float32(LOG2_E),
                            )
                            acc_S_mn[r, c] = val * src_scale * tgt_scale
                        s_row = row_idx - n_block * n
                        s_col = col_idx - m_block * m
                        if cute.elem_less(s_row, n) and cute.elem_less(s_col, m):
                            sS[s_row, s_col] = acc_S_mn[r, c].to(self.ab_dtype)

                cute.arch.barrier()
                cute.copy(
                    smem_tiled_copy_A, tSsS[None, None, 0], tSrS_view[None, None, 0]
                )
                for kk in cutlass.range_constexpr(cute.size(tSrS.shape[2])):
                    kk_next = (kk + 1) % cute.size(tSrS.shape[2])
                    cute.copy(
                        smem_tiled_copy_A,
                        tSsS[None, None, kk_next],
                        tSrS_view[None, None, kk_next],
                    )
                    cute.gemm(
                        tiled_mma,
                        acc_DV_curr,
                        tSrS[None, None, kk],
                        tSrDY[None, None, kk],
                        acc_DV_curr,
                    )

            cOut = cute.local_tile(mcOut[0, None, 0, None], (n, p), (n_block, p_block))
            tOcOut = thr_mma.partition_C(cOut)
            tOcOut_mn = self._make_acc_tensor_mn_view(tOcOut)
            acc_DV_prev_mn = self._make_acc_tensor_mn_view(acc_DV_prev)
            acc_DV_curr_mn = self._make_acc_tensor_mn_view(acc_DV_curr)
            for r in cutlass.range_constexpr(cute.size(acc_DV_prev_mn.shape[0])):
                for c in cutlass.range_constexpr(cute.size(acc_DV_prev_mn.shape[1])):
                    row_idx = cutlass.Int32(tOcOut_mn[r, c][1])
                    col_idx = cutlass.Int32(tOcOut_mn[r, c][3])
                    row_local = row_idx - cutlass.Int32(n_block * n)
                    col_local = col_idx - cutlass.Int32(p_base)
                    if cute.elem_less(row_idx, cutlass.Int32(L)) and cute.elem_less(
                        col_idx, cutlass.Int32(self.cfg.P)
                    ):
                        sPrev[row_local, col_local] = cutlass.Float32(
                            acc_DV_prev_mn[r, c]
                        )
            cute.arch.barrier()

            for r in cutlass.range_constexpr(cute.size(acc_DV_curr_mn.shape[0])):
                for c in cutlass.range_constexpr(cute.size(acc_DV_curr_mn.shape[1])):
                    row_idx = cutlass.Int32(tOcOut_mn[r, c][1])
                    col_idx = cutlass.Int32(tOcOut_mn[r, c][3])
                    row_local = row_idx - cutlass.Int32(n_block * n)
                    col_local = col_idx - cutlass.Int32(p_base)
                    if cute.elem_less(row_idx, cutlass.Int32(L)) and cute.elem_less(
                        col_idx, cutlass.Int32(self.cfg.P)
                    ):
                        prev_f32 = cutlass.Float32(0.0)
                        if cutlass.Int32(0) < row_local:
                            prev_f32 = cutlass.Float32(
                                sPrev[row_local - cutlass.Int32(1), col_local]
                            )
                        elif cutlass.Int32(0) < cutlass.Int32(n_block):
                            prev_f32 = cutlass.Float32(sCarry[col_local])
                        global_t = chunk * cutlass.Int32(L) + (
                            cutlass.Int32(L - 1) - row_idx
                        )
                        mDUPad[bh, global_t, col_idx] = (
                            cutlass.Float32(acc_DV_curr_mn[r, c]) + prev_f32
                        )
            cute.arch.barrier()

            if tidx < cutlass.Int32(p):
                feat = p_base + tidx
                if cute.elem_less(feat, cutlass.Int32(self.cfg.P)):
                    sCarry[tidx] = cutlass.Float32(
                        sPrev[cutlass.Int32(n - 1), tidx]
                    )
            cute.arch.barrier()

        if tidx < cutlass.Int32(p):
            feat = p_base + tidx
            if cute.elem_less(feat, cutlass.Int32(self.cfg.P)):
                mBoundary[bhc, feat] = cutlass.Float32(sCarry[tidx])


class _ChunkScanBwdDUBoundary:
    """Device-side chunk-boundary epilogue for public ``dU``.

    Logical shape
    -------------
    - ``boundary``: ``(BHC, P)``, fp32 chunk-boundary ``dV_prev`` rows
    - ``dU_pad``: ``(BH, T_pad, P)`` in fp32
    - ``dU_prev``: ``(BH, P)`` in fp32

    Mapping
    -------
    - one thread owns one feature
    - the main DU kernel already wrote every public row except the final row of
      each chunk, which needs the next chunk's boundary carry
    - this epilogue patches exactly that row and materializes ``dU_prev``
    """

    def __init__(self, *, feat_tile: int, num_threads: int = 128) -> None:
        self.feat_tile = int(feat_tile)
        self.num_threads = int(num_threads)
        if self.feat_tile <= 0 or self.num_threads % self.feat_tile != 0:
            raise ValueError("num_threads must be divisible by feat_tile.")
        self.row_tile = self.num_threads // self.feat_tile

    @cute.jit
    def __call__(
        self,
        mBoundary: cute.Tensor,
        mDUPad: cute.Tensor,
        mDUPrev: cute.Tensor,
        n_chunks: cutlass.Int32,
        chunk_size: cutlass.Int32,
    ) -> None:
        if cutlass.const_expr(
            not (
                mBoundary.element_type
                == mDUPad.element_type
                == mDUPrev.element_type
                == cutlass.Float32
            )
        ):
            raise TypeError("DU boundary epilogue expects Float32 tensors.")
        BHC = cute.size(mBoundary.shape[0])
        P = cute.size(mDUPad.shape[2])
        grid_x = cute.ceil_div(P, self.feat_tile)
        self.kernel(mBoundary, mDUPad, mDUPrev, n_chunks, chunk_size).launch(
            grid=[grid_x, BHC, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mBoundary: cute.Tensor,
        mDUPad: cute.Tensor,
        mDUPrev: cute.Tensor,
        n_chunks: cutlass.Int32,
        chunk_size: cutlass.Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        feat_tile_idx, bhc, _ = cute.arch.block_idx()

        feat_local = tidx % self.feat_tile
        feat = feat_tile_idx * self.feat_tile + feat_local

        if cute.elem_less(feat, mDUPad.shape[2]):
            bh = bhc // n_chunks
            chunk = bhc - bh * n_chunks
            if chunk == cutlass.Int32(0):
                mDUPrev[bh, feat] = cutlass.Float32(mBoundary[bhc, feat])
            next_chunk = chunk + cutlass.Int32(1)
            if cute.elem_less(next_chunk, n_chunks):
                tail_t = chunk * chunk_size + (chunk_size - cutlass.Int32(1))
                mDUPad[bh, tail_t, feat] = cutlass.Float32(
                    mDUPad[bh, tail_t, feat]
                ) + cutlass.Float32(mBoundary[bhc + cutlass.Int32(1), feat])


def prepare_chunk_scan_bwd_du_operands(
    Q: torch.Tensor,
    Kprev: torch.Tensor,
    Kcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the cached reverse-time contract for ``chunk_scan`` value grads."""
    if Q.ndim != 4 or Kprev.ndim != 4 or Kcurr.ndim != 4:
        raise ValueError("Q/Kprev/Kcurr must be rank-4 tensors.")
    if Q.shape != Kprev.shape or Q.shape != Kcurr.shape:
        raise ValueError(
            "Q, Kprev, and Kcurr must have the same packed-inner shape. Got "
            f"{tuple(Q.shape)}, {tuple(Kprev.shape)}, {tuple(Kcurr.shape)}."
        )
    if Q.shape[2] != 1:
        raise ValueError("Packed Q/K tensors must be shaped as (BHC, L, 1, D).")
    if logprefix_half.shape != Q.shape[:2]:
        raise ValueError(
            "logprefix_half must be (BHC, L) matching Q/K. Got "
            f"{tuple(logprefix_half.shape)} for Q shape {tuple(Q.shape)}."
        )
    if not (
        Q.is_contiguous()
        and Kprev.is_contiguous()
        and Kcurr.is_contiguous()
        and logprefix_half.is_contiguous()
    ):
        raise ValueError(
            "Q, Kprev, Kcurr, and logprefix_half must be contiguous cached "
            "forward tensors."
        )

    return (
        torch.flip(Q, dims=[1]).contiguous(),
        torch.flip(Kprev, dims=[1]).contiguous(),
        torch.flip(Kcurr, dims=[1]).contiguous(),
        (-torch.flip(logprefix_half, dims=[1])).contiguous(),
    )


def _get_du_scratch(
    *,
    q_rev: torch.Tensor,
    batch_size: int,
    n_heads: int,
    P: int,
) -> _ChunkScanBwdDUScratch:
    device_index = 0 if q_rev.device.index is None else int(q_rev.device.index)
    BHC, L, _, _ = map(int, q_rev.shape)
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Q_rev leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    key: _ScratchKey = (device_index, q_rev.dtype, BHC, L, P, BH)
    scratch = _SCRATCH_DU.get(key)
    if scratch is not None:
        return scratch

    dU_pad = torch.empty((BH, T_pad, P), device=q_rev.device, dtype=torch.float32)
    dU_prev = torch.empty((BH, P), device=q_rev.device, dtype=torch.float32)
    boundary = torch.empty((BHC, P), device=q_rev.device, dtype=torch.float32)
    scratch = _ChunkScanBwdDUScratch(
        dU_pad=dU_pad,
        dU_prev=dU_prev,
        boundary=boundary,
    )
    _SCRATCH_DU[key] = scratch
    return scratch


def _compiled_du_raw_key(
    query_prev_rev: torch.Tensor,
    query_curr_rev: torch.Tensor,
    key_rev: torch.Tensor,
    d_out_rev: torch.Tensor,
    neg_logprefix_half_rev: torch.Tensor,
    dU_pad: torch.Tensor,
    boundary: torch.Tensor,
    *,
    device_index: int,
    n_chunks: int,
) -> _CompiledRawKey:
    return (
        device_index,
        query_prev_rev.dtype,
        tuple(int(x) for x in query_prev_rev.shape),
        tuple(int(x) for x in query_curr_rev.shape),
        tuple(int(x) for x in key_rev.shape),
        tuple(int(x) for x in d_out_rev.shape),
        tuple(int(x) for x in neg_logprefix_half_rev.shape),
        tuple(int(x) for x in dU_pad.shape),
        tuple(int(x) for x in boundary.shape),
        int(n_chunks),
    )


def _get_compiled_du_raw(
    query_prev_rev: torch.Tensor,
    query_curr_rev: torch.Tensor,
    key_rev: torch.Tensor,
    d_out_rev: torch.Tensor,
    neg_logprefix_half_rev: torch.Tensor,
    dU_pad: torch.Tensor,
    boundary: torch.Tensor,
    *,
    n_chunks: int,
) -> object:
    device_index = (
        0 if query_prev_rev.device.index is None else int(query_prev_rev.device.index)
    )
    key = _compiled_du_raw_key(
        query_prev_rev,
        query_curr_rev,
        key_rev,
        d_out_rev,
        neg_logprefix_half_rev,
        dU_pad,
        boundary,
        device_index=device_index,
        n_chunks=n_chunks,
    )
    compiled = _COMPILED_DU_RAW.get(key)
    if compiled is not None:
        return compiled

    _, L, _, D = map(int, query_prev_rev.shape)
    P = int(d_out_rev.shape[-1])
    cfg = _ChunkScanBwdDURawConfig(D=D, P=P, L=L)
    cutlass_dtype = (
        cutlass.Float16
        if query_prev_rev.dtype == torch.float16
        else cutlass.BFloat16
    )
    kernel = _ChunkScanBwdDURawAmpereTc(cutlass_dtype, cfg)
    compiled = cute.compile(
        kernel,
        from_dlpack(query_prev_rev, assumed_align=16),
        from_dlpack(query_curr_rev, assumed_align=16),
        from_dlpack(key_rev, assumed_align=16),
        from_dlpack(d_out_rev, assumed_align=16),
        from_dlpack(neg_logprefix_half_rev, assumed_align=16),
        from_dlpack(dU_pad, assumed_align=16),
        from_dlpack(boundary, assumed_align=16),
        int(n_chunks),
    )
    _COMPILED_DU_RAW[key] = compiled
    return compiled


def _get_compiled_du_boundary(
    boundary: torch.Tensor,
    dU_pad: torch.Tensor,
    dU_prev: torch.Tensor,
    *,
    n_chunks: int,
    chunk_size: int,
) -> object:
    device_index = 0 if boundary.device.index is None else int(boundary.device.index)
    key: _CompiledBoundaryKey = (
        device_index,
        tuple(int(x) for x in boundary.shape),
        tuple(int(x) for x in dU_prev.shape),
        tuple(int(x) for x in dU_pad.shape),
        int(n_chunks),
        int(chunk_size),
    )
    compiled = _COMPILED_DU_BOUNDARY.get(key)
    if compiled is not None:
        return compiled

    feat_tile = _choose_feat_tile(int(dU_pad.shape[-1]))
    kernel = _ChunkScanBwdDUBoundary(feat_tile=feat_tile)
    compiled = cute.compile(
        kernel,
        from_dlpack(boundary, assumed_align=16),
        from_dlpack(dU_pad, assumed_align=16),
        from_dlpack(dU_prev, assumed_align=16),
        int(n_chunks),
        int(chunk_size),
    )
    _COMPILED_DU_BOUNDARY[key] = compiled
    return compiled


def chunk_scan_bwd_du_cute(
    Q_rev: torch.Tensor,
    Kprev_rev: torch.Tensor,
    Kcurr_rev: torch.Tensor,
    neg_logprefix_half_rev: torch.Tensor,
    d_out: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(dU, dU_prev)`` from cached reverse-time packed operands."""
    if (
        Q_rev.device.type != "cuda"
        or Kprev_rev.device.type != "cuda"
        or Kcurr_rev.device.type != "cuda"
        or neg_logprefix_half_rev.device.type != "cuda"
        or d_out.device.type != "cuda"
    ):
        raise ValueError("CuTe chunk_scan backward requires CUDA tensors.")
    if not (
        Q_rev.is_contiguous()
        and Kprev_rev.is_contiguous()
        and Kcurr_rev.is_contiguous()
        and neg_logprefix_half_rev.is_contiguous()
        and d_out.is_contiguous()
    ):
        raise ValueError(
            "chunk_scan backward cached operands and d_out must be contiguous."
        )
    if Q_rev.ndim != 4 or Kprev_rev.ndim != 4 or Kcurr_rev.ndim != 4:
        raise ValueError("Q_rev/Kprev_rev/Kcurr_rev must be rank-4 tensors.")
    if Q_rev.shape != Kprev_rev.shape or Q_rev.shape != Kcurr_rev.shape:
        raise ValueError(
            "Q_rev, Kprev_rev, and Kcurr_rev must have the same shape. Got "
            f"{tuple(Q_rev.shape)}, {tuple(Kprev_rev.shape)}, "
            f"{tuple(Kcurr_rev.shape)}."
        )
    if Q_rev.shape[2] != 1:
        raise ValueError("Packed reverse-time Q/K tensors must be (BHC, L, 1, D).")
    if neg_logprefix_half_rev.shape != Q_rev.shape[:2]:
        raise ValueError(
            "neg_logprefix_half_rev must be (BHC, L) matching Q_rev. Got "
            f"{tuple(neg_logprefix_half_rev.shape)} for Q_rev shape "
            f"{tuple(Q_rev.shape)}."
        )
    if d_out.ndim != 4:
        raise ValueError("d_out must be rank-4 (B, H, T, P).")
    if d_out.shape[:2] != (batch_size, n_heads):
        raise ValueError(
            "Leading d_out dims must match (batch_size, n_heads). Got "
            f"{tuple(d_out.shape[:2])} vs {(batch_size, n_heads)}."
        )
    if int(d_out.shape[2]) != T:
        raise ValueError(f"d_out T must match T={T}. Got {int(d_out.shape[2])}.")
    if Q_rev.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("DU tensor-core path expects float16/bfloat16 packed caches.")

    BHC, L, _, _ = map(int, Q_rev.shape)
    P = int(d_out.shape[-1])
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Q_rev leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    if T > T_pad:
        raise ValueError(
            f"T={T} exceeds the cached padded length T_pad={T_pad} implied by Q_rev."
        )

    if T_pad != T:
        d_out = torch.cat(
            [
                d_out,
                torch.zeros(
                    (batch_size, n_heads, T_pad - T, P),
                    device=d_out.device,
                    dtype=d_out.dtype,
                ),
            ],
            dim=2,
        )
    d_out_rev = torch.flip(
        d_out.reshape(BHC, L, 1, P).to(dtype=Q_rev.dtype), dims=[1]
    ).contiguous()
    return _chunk_scan_bwd_du_prepared_cute(
        Q_rev,
        Kprev_rev,
        Kcurr_rev,
        neg_logprefix_half_rev,
        d_out_rev,
        batch_size=batch_size,
        n_heads=n_heads,
        T=T,
    )


def _chunk_scan_bwd_du_prepared_cute(
    Q_rev: torch.Tensor,
    Kprev_rev: torch.Tensor,
    Kcurr_rev: torch.Tensor,
    neg_logprefix_half_rev: torch.Tensor,
    d_out_rev: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(dU, dU_prev)`` from already padded reverse-time ``d_out``."""
    if not d_out_rev.is_contiguous():
        raise ValueError("d_out_rev must be contiguous.")

    BHC, L, _, _ = map(int, Q_rev.shape)
    P = int(d_out_rev.shape[-1])
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Q_rev leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L

    scratch = _get_du_scratch(
        q_rev=Q_rev,
        batch_size=batch_size,
        n_heads=n_heads,
        P=P,
    )

    compiled_raw = _get_compiled_du_raw(
        Kprev_rev,
        Kcurr_rev,
        Q_rev,
        d_out_rev,
        neg_logprefix_half_rev,
        scratch.dU_pad,
        scratch.boundary,
        n_chunks=n_chunks,
    )
    compiled_raw(
        from_dlpack(Kprev_rev, assumed_align=16),
        from_dlpack(Kcurr_rev, assumed_align=16),
        from_dlpack(Q_rev, assumed_align=16),
        from_dlpack(d_out_rev, assumed_align=16),
        from_dlpack(neg_logprefix_half_rev, assumed_align=16),
        from_dlpack(scratch.dU_pad, assumed_align=16),
        from_dlpack(scratch.boundary, assumed_align=16),
        n_chunks,
    )

    compiled_boundary = _get_compiled_du_boundary(
        scratch.boundary,
        scratch.dU_pad,
        scratch.dU_prev,
        n_chunks=n_chunks,
        chunk_size=L,
    )
    compiled_boundary(
        from_dlpack(scratch.boundary, assumed_align=16),
        from_dlpack(scratch.dU_pad, assumed_align=16),
        from_dlpack(scratch.dU_prev, assumed_align=16),
        n_chunks,
        L,
    )

    return (
        scratch.dU_pad.reshape(batch_size, n_heads, T_pad, P)[:, :, :T, :].contiguous(),
        scratch.dU_prev.reshape(batch_size, n_heads, P).contiguous(),
    )


__all__ = [
    "prepare_chunk_scan_bwd_du_operands",
    "chunk_scan_bwd_du_cute",
]
