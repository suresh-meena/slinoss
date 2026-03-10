"""Common helpers for CuTe ``v2x2ssd`` chunk-increment backward.

The backward stage consumes the staged forward summary:

``inc = A_main^T @ B_main + u_head ⊗ b_head``

and then backpropagates through the operand preparation algebra to produce
gradients for the original ``(U, M, K, B, B_prev, U_prev)`` inputs.

The large dense work stays in CuTe via the existing forward SGEMM kernel. The
remaining scan over per-chunk complex scalars is intentionally explicit Torch
code: the chunk axis is short, the math is easier to audit there, and it keeps
the CuTe surface limited to the parts that actually dominate runtime.
"""

from __future__ import annotations

import torch


def _scalar_grad_from_vec(base: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Reduce ``grad`` against complex-vector ``base`` to a complex scalar grad.

    For ``y = s * base`` with complex scalar ``s`` and complex-vector output
    ``y``, the real-valued gradient w.r.t. ``s`` is the packed-complex scalar
    induced by the underlying 2x2 real multiplication:

    - ``d_re = Σ (g_re * x_re + g_im * x_im)``
    - ``d_im = Σ (-g_re * x_im + g_im * x_re)``
    """

    return torch.complex(
        (grad.real * base.real + grad.imag * base.imag).sum(dim=-1),
        (-grad.real * base.imag + grad.imag * base.real).sum(dim=-1),
    )
