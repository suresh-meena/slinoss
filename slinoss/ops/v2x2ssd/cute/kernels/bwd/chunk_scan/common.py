"""Common CuTe helpers for the standalone ``v2x2ssd`` chunk-scan bwd kernels."""

from __future__ import annotations

import cutlass.cute as cute


LOG2_E = 1.4426950408889634074
TWO_LOG2_E = 2.0 * LOG2_E


@cute.jit
def complex_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@cute.jit
def conj_mul_phase(xr, xi, pr, pi):
    """Return ``conj(x) * p`` for packed complex scalars."""
    return xr * pr + xi * pi, xr * pi - xi * pr


@cute.jit
def mul_conj_phase(xr, xi, pr, pi):
    """Return ``x * conj(p)`` for packed complex scalars."""
    return xr * pr + xi * pi, xi * pr - xr * pi


@cute.jit
def apply_complex_tap(xr, xi, kr, ki):
    """Apply packed complex tap ``k`` to packed complex value ``x``."""
    return kr * xr - ki * xi, kr * xi + ki * xr


@cute.jit
def apply_complex_tap_adjoint(gr, gi, kr, ki):
    """Adjoint of :func:`apply_complex_tap` for packed real-imag pairs."""
    return kr * gr + ki * gi, -ki * gr + kr * gi
