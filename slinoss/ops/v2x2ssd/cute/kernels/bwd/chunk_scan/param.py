"""Parameter-side backward slice for ``chunk_scan`` gradients into ``M``.

This slice stays on the packed chunk contract and differentiates the dense
``Q/K`` algebra explicitly in Torch:

- exact packed ``dQ/dKprev/dKcurr`` from batched matrix products
- exact cumulative ``dlogprefix_half`` from the same packed contract
- short SO(2) reverse scan back to per-step ``M``

The chunk axis is intentionally kept explicit and small here. That keeps the
phase-sensitive path numerically transparent and avoids feeding approximate
packed ``dQ/dK`` slices into the final ``M`` reduction.
"""

from __future__ import annotations

import torch

from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_increment.common import (
    _scalar_grad_from_vec,
)
from slinoss.ops.v2x2ssd.reference import _as_complex_pairs


def _packed_causal_scales(logprefix_half: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the stable packed-contract scale tensors.

    ``logprefix_half`` stores half of the cumulative log-magnitude prefix. The
    packed dense path uses:

    - ``row_scale[t] = exp(2 * lp[t])`` for the off-term
    - ``scale[t, s] = exp(2 * (lp[t] - lp[s]))`` for the causal diagonal terms

    The explicit causal mask keeps the would-be undefined upper triangle out of
    the computation entirely instead of relying on later multiplication by zero.
    """

    if logprefix_half.ndim != 2:
        raise ValueError(
            f"logprefix_half must be rank-2 packed metadata. Got {tuple(logprefix_half.shape)}."
        )
    L = int(logprefix_half.shape[1])
    t_idx = torch.arange(L, device=logprefix_half.device).unsqueeze(1)
    s_idx = torch.arange(L, device=logprefix_half.device).unsqueeze(0)
    causal = (s_idx <= t_idx).unsqueeze(0)
    lp = logprefix_half.to(torch.float32)
    scale = torch.exp(2.0 * (lp.unsqueeze(-1) - lp.unsqueeze(1))).masked_fill(
        ~causal, 0.0
    )
    row_scale = torch.exp(2.0 * lp).unsqueeze(-1)
    return scale, row_scale


def _dlogprefix_half_packed(
    score_prev: torch.Tensor,
    score_curr: torch.Tensor,
    dSprev: torch.Tensor,
    dScurr: torch.Tensor,
    y_off: torch.Tensor,
    scale: torch.Tensor,
    d_out_flat: torch.Tensor,
) -> torch.Tensor:
    """Exact packed-contract gradient for cumulative ``logprefix_half``.

    For the diagonal terms, each ``lp[k]`` contributes with opposite signs to
    row ``k`` and column ``k`` of the stable segment-ratio matrix. Writing that
    contribution explicitly as row-sum minus column-sum avoids building an
    autograd graph for this short metadata path.
    """

    e_prev = dSprev * score_prev * scale
    e_curr = dScurr * score_curr * scale
    return (
        2.0 * (d_out_flat * y_off).sum(dim=-1)
        + 2.0 * (e_prev.sum(dim=2) - e_prev.sum(dim=1))
        + 2.0 * (e_curr.sum(dim=2) - e_curr.sum(dim=1))
    ).contiguous()


def _packed_phase_prefix(M_raw: torch.Tensor) -> torch.Tensor:
    """Build the unit-complex phase prefix from raw packed ``M``."""

    m_c = torch.view_as_complex(M_raw.contiguous())
    mag = m_c.abs().clamp_min(torch.finfo(torch.float32).tiny)
    unit = m_c / mag
    return torch.cumprod(unit, dim=1)


def chunk_scan_bwd_param_cute(
    Q: torch.Tensor,
    Kprev: torch.Tensor,
    Vprev: torch.Tensor,
    Kcurr: torch.Tensor,
    Vcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
    Z0: torch.Tensor,
    M_raw: torch.Tensor,
    d_out: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> torch.Tensor:
    """Compute ``dM`` for ``chunk_scan`` from cached packed forward tensors."""

    tensors = (
        ("Q", Q),
        ("Kprev", Kprev),
        ("Vprev", Vprev),
        ("Kcurr", Kcurr),
        ("Vcurr", Vcurr),
        ("logprefix_half", logprefix_half),
        ("Z0", Z0),
        ("M_raw", M_raw),
        ("d_out", d_out),
    )
    if any(t.device.type != "cuda" for _name, t in tensors):
        raise ValueError("CuTe chunk_scan backward requires CUDA tensors.")
    if any(not t.is_contiguous() for _name, t in tensors):
        raise ValueError(
            "chunk_scan backward cached operands and d_out must be contiguous."
        )
    if Q.ndim != 4 or Kprev.ndim != 4 or Kcurr.ndim != 4:
        raise ValueError("Q/K tensors must be rank-4 packed tensors.")
    if Q.shape != Kprev.shape or Q.shape != Kcurr.shape:
        raise ValueError(
            "Q, Kprev, and Kcurr must share the same packed D contract. Got "
            f"{tuple(Q.shape)}, {tuple(Kprev.shape)}, {tuple(Kcurr.shape)}."
        )
    if Vprev.shape != Vcurr.shape or Vprev.ndim != 4 or Vprev.shape[2] != 1:
        raise ValueError("Vprev/Vcurr must be packed as (BHC, L, 1, P).")
    if logprefix_half.shape != Q.shape[:2]:
        raise ValueError("logprefix_half must be (BHC, L) matching Q.")
    if Z0.ndim != 4 or Z0.shape[0] != Q.shape[0] or Z0.shape[2] != 1:
        raise ValueError("Z0 must be packed as (BHC, P, 1, D).")
    if M_raw.shape != (*Q.shape[:2], 2):
        raise ValueError(
            "M_raw must be (BHC, L, 2) matching Q. Got "
            f"{tuple(M_raw.shape)}."
        )
    if d_out.ndim != 4 or d_out.shape[:2] != (batch_size, n_heads) or int(d_out.shape[2]) != T:
        raise ValueError(
            "d_out must be (batch_size, n_heads, T, P). Got "
            f"{tuple(d_out.shape)}."
        )

    BHC, L, _, D = map(int, Q.shape)
    N = D // 2
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Q leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    if T > T_pad:
        raise ValueError(
            f"T={T} exceeds the cached padded length T_pad={T_pad} implied by Q."
        )

    P = int(Vprev.shape[-1])
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

    d_out_flat = d_out.reshape(BHC, L, P).to(torch.float32)
    Qf = Q.squeeze(2).to(torch.float32)
    Kprevf = Kprev.squeeze(2).to(torch.float32)
    Kcurrf = Kcurr.squeeze(2).to(torch.float32)
    Vprevf = Vprev.squeeze(2).to(torch.float32)
    Vcurrf = Vcurr.squeeze(2).to(torch.float32)
    Z0f = Z0.squeeze(2).to(torch.float32)

    scale, row_scale = _packed_causal_scales(logprefix_half)
    score_prev = torch.bmm(Qf, Kprevf.transpose(1, 2))
    score_curr = torch.bmm(Qf, Kcurrf.transpose(1, 2))
    dSprev = torch.bmm(d_out_flat, Vprevf.transpose(1, 2))
    dScurr = torch.bmm(d_out_flat, Vcurrf.transpose(1, 2))
    dScore_prev = dSprev * scale
    dScore_curr = dScurr * scale
    y_off = torch.bmm(Qf, Z0f.transpose(1, 2)) * row_scale

    dQ = (
        torch.bmm(d_out_flat * row_scale, Z0f)
        + torch.bmm(dScore_prev, Kprevf)
        + torch.bmm(dScore_curr, Kcurrf)
    )
    dKprev = torch.bmm(dScore_prev.transpose(1, 2), Qf)
    dKcurr = torch.bmm(dScore_curr.transpose(1, 2), Qf)
    d_logprefix_half = _dlogprefix_half_packed(
        score_prev,
        score_curr,
        dSprev,
        dScurr,
        y_off,
        scale,
        d_out_flat,
    )

    phase = _packed_phase_prefix(M_raw)
    phase_inv = torch.conj(phase).unsqueeze(-1)
    q_base = _as_complex_pairs(Qf, name="Q") * phase_inv
    kprev_base = _as_complex_pairs(Kprevf, name="Kprev") * phase_inv
    kcurr_base = _as_complex_pairs(Kcurrf, name="Kcurr") * phase_inv
    d_phase = (
        _scalar_grad_from_vec(
            q_base, torch.view_as_complex(dQ.reshape(BHC, L, N, 2).contiguous())
        )
        + _scalar_grad_from_vec(
            kprev_base,
            torch.view_as_complex(dKprev.reshape(BHC, L, N, 2).contiguous()),
        )
        + _scalar_grad_from_vec(
            kcurr_base,
            torch.view_as_complex(dKcurr.reshape(BHC, L, N, 2).contiguous()),
        )
    )

    phase_blk = phase.reshape(batch_size, n_heads, n_chunks, L)
    d_phase_blk = d_phase.reshape(batch_size, n_heads, n_chunks, L)
    m_c = torch.view_as_complex(M_raw.contiguous()).reshape(
        batch_size, n_heads, n_chunks, L
    )
    mag = m_c.abs().clamp_min(torch.finfo(torch.float32).tiny)
    unit = m_c / mag

    carry = torch.zeros(
        (batch_size, n_heads, n_chunks), device=M_raw.device, dtype=torch.complex64
    )
    d_unit = torch.zeros_like(d_phase_blk)
    for t in range(L - 1, -1, -1):
        total = d_phase_blk[..., t] + carry
        p_prev = torch.ones_like(total) if t == 0 else phase_blk[..., t - 1]
        d_unit[..., t] = _scalar_grad_from_vec(
            p_prev.unsqueeze(-1), total.unsqueeze(-1)
        )
        carry = _scalar_grad_from_vec(unit[..., t].unsqueeze(-1), total.unsqueeze(-1))

    # ``unit = m / |m|`` is the numerically dangerous-looking part of the SO(2)
    # map. The magnitude clamp above enforces the same strictly-nonzero
    # operating region as the forward path before dividing by ``|m|``.
    d_phase_m = (d_unit - unit * torch.real(torch.conj(unit) * d_unit)) / mag
    d_logr_half = torch.flip(
        torch.cumsum(
            torch.flip(
                d_logprefix_half.reshape(batch_size, n_heads, n_chunks, L),
                dims=[-1],
            ),
            dim=-1,
        ),
        dims=[-1],
    )
    d_mag_m = 0.5 * d_logr_half * m_c / (mag * mag)
    dM_c = (d_phase_m + d_mag_m).reshape(batch_size, n_heads, T_pad)
    return torch.view_as_real(dM_c).to(dtype=torch.float32)[:, :, :T, :].contiguous()


__all__ = ["chunk_scan_bwd_param_cute"]
