"""CuTe backward ``param_scan`` workhorse for the ``v2x2ssd`` chunk-scan stage.

This file is intentionally written in the same overall shape as the
``v3x3ssd`` ``chunk_scan`` ``param_scan`` workhorse:

- one monolithic stage-native kernel class
- ``__call__`` is only validation plus launch over ``(BHC, n_splits)``
- one thread owns one ``BHC`` lane for one split
- one reverse-time scalar scan owns the full metadata / parameter slice

The adaptation is only in the scan algebra:

- quaternion scan-backward becomes unit-complex scan-backward
- 3D axis-angle transport becomes packed complex transition transport
- 3x3 rotation partials become 2x2 real phase-matrix partials
- 4-coefficient taps become 2-coefficient complex taps
- the metadata output is public ``dM`` rather than a larger ``dtrans`` record
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from .common import complex_mul


class ChunkScanBwdParamScanAmpere:
    """Param scan-backward + ``dlog`` reverse scan for the ``v2`` chunk scan.

    One thread handles one ``BHC`` index for one split and computes:
      - reverse inclusive cumsum of ``dlp`` into the log-magnitude part of ``dM``
      - reverse phase scan-backward into the phase part of ``dM``
      - raw complex-tap grads ``dKprev`` / ``dKcurr``

    Contracts:
      - ``M`` is raw packed complex transitions ``(re, im)``
      - ``K`` stores two raw complex taps per step: ``prev`` and ``curr``
      - ``dMprev`` / ``dMcurr`` are gradients of the transformed packed tap
        coefficients before the reverse metadata scan
      - ``dR`` is the gradient of the 2x2 real phase matrix
        ``A(p) = [[pr, pi], [pi, -pr]]``
    """

    def __init__(
        self,
        *,
        chunk_size: int,
        num_threads: int = 32,
        assume_dmprev_zero: bool = False,
    ):
        self.L = int(chunk_size)
        self.num_threads = int(num_threads)
        self.assume_dmprev_zero = bool(assume_dmprev_zero)
        if self.L <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32.")

    @cute.jit
    def __call__(
        self,
        mM: cute.Tensor,  # (BHC, L, 2) fp32
        mK: cute.Tensor,  # (BHC, L, 2, 2) fp32
        mDLp: cute.Tensor,  # (BHC, n_splits, L) fp32
        mDMprev: cute.Tensor,  # (BHC, n_splits, L, 2) fp32
        mDMcurr: cute.Tensor,  # (BHC, n_splits, L, 2) fp32
        mDR: cute.Tensor,  # (BHC, n_splits, L, 4) fp32
        mDM: cute.Tensor,  # (BHC, n_splits, L, 2) fp32
        mDKprev: cute.Tensor,  # (BHC, n_splits, L, 2) fp32
        mDKcurr: cute.Tensor,  # (BHC, n_splits, L, 2) fp32
    ):
        if cutlass.const_expr(mM.element_type != cutlass.Float32):
            raise TypeError("M must be Float32.")
        if cutlass.const_expr(mK.element_type != cutlass.Float32):
            raise TypeError("K must be Float32.")
        if cutlass.const_expr(mDLp.element_type != cutlass.Float32):
            raise TypeError("dLp must be Float32.")
        if cutlass.const_expr(
            mDMprev.element_type != cutlass.Float32
            or mDMcurr.element_type != cutlass.Float32
            or mDR.element_type != cutlass.Float32
        ):
            raise TypeError("dMprev/dMcurr/dR must be Float32.")
        if cutlass.const_expr(
            mDM.element_type != cutlass.Float32
            or mDKprev.element_type != cutlass.Float32
            or mDKcurr.element_type != cutlass.Float32
        ):
            raise TypeError("dM/dK outputs must be Float32.")

        if cutlass.const_expr(mM.shape[1] != self.L or mM.shape[2] != 2):
            raise ValueError("M must be (BHC, L, 2).")
        if cutlass.const_expr(
            mK.shape[1] != self.L or mK.shape[2] != 2 or mK.shape[3] != 2
        ):
            raise ValueError("K must be (BHC, L, 2, 2).")
        if cutlass.const_expr(mDLp.shape[2] != self.L):
            raise ValueError("dLp must be (BHC, n_splits, L).")
        if cutlass.const_expr(
            mDMprev.shape[2] != self.L
            or mDMprev.shape[3] != 2
            or mDMcurr.shape[2] != self.L
            or mDMcurr.shape[3] != 2
        ):
            raise ValueError("dMprev/dMcurr must be (BHC, n_splits, L, 2).")
        if cutlass.const_expr(mDR.shape[2] != self.L or mDR.shape[3] != 4):
            raise ValueError("dR must be (BHC, n_splits, L, 4).")
        if cutlass.const_expr(mDM.shape != mDMprev.shape):
            raise ValueError("dM output must match dMprev shape.")
        if cutlass.const_expr(mDKprev.shape != mDMprev.shape):
            raise ValueError("dKprev must match dMprev shape.")
        if cutlass.const_expr(mDKcurr.shape != mDMcurr.shape):
            raise ValueError("dKcurr must match dMcurr shape.")

        BHC = cute.size(mM.shape[0])
        n_splits = cute.size(mDLp.shape[1])
        grid_x = cute.ceil_div(BHC, self.num_threads)
        self.kernel(
            mM,
            mK,
            mDLp,
            mDMprev,
            mDMcurr,
            mDR,
            mDM,
            mDKprev,
            mDKcurr,
            BHC,
            n_splits,
        ).launch(
            grid=[cute.size(grid_x), n_splits, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mM: cute.Tensor,
        mK: cute.Tensor,
        mDLp: cute.Tensor,
        mDMprev: cute.Tensor,
        mDMcurr: cute.Tensor,
        mDR: cute.Tensor,
        mDM: cute.Tensor,
        mDKprev: cute.Tensor,
        mDKcurr: cute.Tensor,
        BHC: cutlass.Int32,
        n_splits: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        b0, b1, _ = cute.arch.block_idx()
        bhc = b0 * self.num_threads + tidx
        split = b1

        bhc_valid = cute.elem_less(bhc, BHC)
        bhc_safe = cutlass.min(bhc, BHC - cutlass.Int32(1))

        eps_norm = cutlass.Float32(1.0e-20)

        # ------------------------------------------------------------------
        # Precompute the total unit-complex phase product over the chunk.
        # ------------------------------------------------------------------
        ptr = cutlass.Float32(1.0)
        pti = cutlass.Float32(0.0)
        for t_it in cutlass.range(self.L, unroll=1):
            t = (self.L - 1) - t_it
            mr = cutlass.Float32(mM[bhc_safe, t, 0])
            mi = cutlass.Float32(mM[bhc_safe, t, 1])
            mag2 = mr * mr + mi * mi
            inv_mag = cutlass.Float32(cute.math.rsqrt(mag2 + eps_norm))
            ur = mr * inv_mag
            ui = mi * inv_mag
            nr, ni = complex_mul(ptr, pti, ur, ui)
            n2 = nr * nr + ni * ni
            invn = cutlass.Float32(cute.math.rsqrt(n2 + eps_norm))
            ptr = nr * invn
            pti = ni * invn

        # ------------------------------------------------------------------
        # Reverse-time scan-backward:
        #   - suffix phase state
        #   - carry for the prefix-phase product
        #   - reverse inclusive cumsum of dlp
        # ------------------------------------------------------------------
        suf_r = cutlass.Float32(1.0)
        suf_i = cutlass.Float32(0.0)
        carry_r = cutlass.Float32(0.0)
        carry_i = cutlass.Float32(0.0)
        dlog_running = cutlass.Float32(0.0)

        for t_it in cutlass.range(self.L, unroll=1):
            t = (self.L - 1) - t_it

            mr = cutlass.Float32(mM[bhc_safe, t, 0])
            mi = cutlass.Float32(mM[bhc_safe, t, 1])
            mag2 = mr * mr + mi * mi
            inv_mag = cutlass.Float32(cute.math.rsqrt(mag2 + eps_norm))
            mag = mag2 * inv_mag
            ur = mr * inv_mag
            ui = mi * inv_mag

            kp_re = cutlass.Float32(mK[bhc_safe, t, 0, 0])
            kp_im = cutlass.Float32(mK[bhc_safe, t, 0, 1])
            kc_re = cutlass.Float32(mK[bhc_safe, t, 1, 0])
            kc_im = cutlass.Float32(mK[bhc_safe, t, 1, 1])

            if cutlass.const_expr(self.assume_dmprev_zero):
                dmp0 = cutlass.Float32(0.0)
                dmp1 = cutlass.Float32(0.0)
            else:
                dmp0 = cutlass.Float32(mDMprev[bhc_safe, split, t, 0])
                dmp1 = cutlass.Float32(mDMprev[bhc_safe, split, t, 1])
            dmc0 = cutlass.Float32(mDMcurr[bhc_safe, split, t, 0])
            dmc1 = cutlass.Float32(mDMcurr[bhc_safe, split, t, 1])

            # prefix_t = total * conj(suffix_{>t}); SO(2) is commutative.
            pr, pi = complex_mul(ptr, pti, suf_r, -suf_i)
            pn2 = pr * pr + pi * pi
            invpn = cutlass.Float32(cute.math.rsqrt(pn2 + eps_norm))
            pr = pr * invpn
            pi = pi * invpn

            # A(p) = [[pr, pi], [pi, -pr]]
            # dR addends from transformed-coefficient paths:
            #   y_prev = A(p) @ k_prev
            #   y_curr = A(p) @ k_curr
            dR00 = dmp0 * kp_re + dmc0 * kc_re
            dR01 = dmp0 * kp_im + dmc0 * kc_im
            dR10 = dmp1 * kp_re + dmc1 * kc_re
            dR11 = dmp1 * kp_im + dmc1 * kc_im

            dR00 = dR00 + cutlass.Float32(mDR[bhc_safe, split, t, 0])
            dR01 = dR01 + cutlass.Float32(mDR[bhc_safe, split, t, 1])
            dR10 = dR10 + cutlass.Float32(mDR[bhc_safe, split, t, 2])
            dR11 = dR11 + cutlass.Float32(mDR[bhc_safe, split, t, 3])

            # dK = A(p)^T @ dM
            dkprev_re = pr * dmp0 + pi * dmp1
            dkprev_im = pi * dmp0 - pr * dmp1
            dkcurr_re = pr * dmc0 + pi * dmc1
            dkcurr_im = pi * dmc0 - pr * dmc1

            if bhc_valid:
                mDKprev[bhc_safe, split, t, 0] = dkprev_re
                mDKprev[bhc_safe, split, t, 1] = dkprev_im
                mDKcurr[bhc_safe, split, t, 0] = dkcurr_re
                mDKcurr[bhc_safe, split, t, 1] = dkcurr_im

            # VJP of A(p) wrt phase p = (pr, pi).
            dphase_local_r = dR00 - dR11
            dphase_local_i = dR01 + dR10

            dphase_r = dphase_local_r + carry_r
            dphase_i = dphase_local_i + carry_i

            # prefix_{t-1} = total * conj(suffix_{>=t})
            suf_prev_r, suf_prev_i = complex_mul(suf_r, suf_i, ur, ui)
            suf_prev_n2 = suf_prev_r * suf_prev_r + suf_prev_i * suf_prev_i
            invsuf = cutlass.Float32(cute.math.rsqrt(suf_prev_n2 + eps_norm))
            suf_prev_r = suf_prev_r * invsuf
            suf_prev_i = suf_prev_i * invsuf
            pprev_r, pprev_i = complex_mul(ptr, pti, suf_prev_r, -suf_prev_i)

            # Product VJP for prefix_t = prefix_{t-1} * unit_t.
            dunit_r = dphase_r * pprev_r + dphase_i * pprev_i
            dunit_i = -dphase_r * pprev_i + dphase_i * pprev_r
            carry_r = dphase_r * ur + dphase_i * ui
            carry_i = -dphase_r * ui + dphase_i * ur

            # VJP from unit phase back to raw M through normalization.
            dot = ur * dunit_r + ui * dunit_i
            dphase_m_re = (dunit_r - ur * dot) / mag
            dphase_m_im = (dunit_i - ui * dot) / mag

            # Reverse inclusive cumsum of dlp onto the log-magnitude channel.
            dlog_running = dlog_running + cutlass.Float32(mDLp[bhc_safe, split, t])
            scale = cutlass.Float32(0.5) * dlog_running / (mag2 + eps_norm)
            dmag_m_re = scale * mr
            dmag_m_im = scale * mi

            if bhc_valid:
                mDM[bhc_safe, split, t, 0] = dphase_m_re + dmag_m_re
                mDM[bhc_safe, split, t, 1] = dphase_m_im + dmag_m_im

            suf_r = suf_prev_r
            suf_i = suf_prev_i

        return


__all__ = ["ChunkScanBwdParamScanAmpere"]
