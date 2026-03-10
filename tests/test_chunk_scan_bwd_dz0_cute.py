from __future__ import annotations

import math

import pytest
import torch

from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan import chunk_scan_bwd_dz0_cute
from slinoss.ops.v2x2ssd.reference import (
    chunk_increment,
    chunk_scan,
    state_passing,
)


def _make_inputs(
    *,
    batch: int,
    heads: int,
    T: int,
    N: int,
    P: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    radius = 0.6 + 0.35 * torch.rand((batch, heads, T), device=device)
    angle = (2.0 * math.pi) * torch.rand((batch, heads, T), device=device) - math.pi
    M = torch.view_as_real(torch.polar(radius, angle)).to(torch.float32).contiguous()

    K_complex = (
        torch.randn((batch, heads, T, 2), device=device, dtype=torch.float32)
        + 1j * torch.randn((batch, heads, T, 2), device=device, dtype=torch.float32)
    ) * 0.1
    K = torch.view_as_real(K_complex).to(torch.float32).contiguous()

    U = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)
    B = torch.randn((batch, heads, T, 2 * N), device=device, dtype=torch.float32) * 0.1
    C = torch.randn((batch, heads, T, 2 * N), device=device, dtype=torch.float32) * 0.1
    B_prev = (
        torch.randn((batch, heads, 2 * N), device=device, dtype=torch.float32) * 0.1
    )
    U_prev = torch.randn((batch, heads, P), device=device, dtype=torch.float32)
    return U, M, K, B, C, B_prev, U_prev


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_dz0_cute_matches_autograd() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, B_prev, U_prev = _make_inputs(
        batch=2,
        heads=2,
        T=64,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    M.requires_grad_(True)
    K.requires_grad_(True)
    U.requires_grad_(True)
    B.requires_grad_(True)
    C.requires_grad_(True)
    B_prev.requires_grad_(True)
    U_prev.requires_grad_(True)

    inc, m_chunk = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=U.shape[2],
        chunk_size=32,
        compute_dtype=torch.float32,
    )
    chunk_starts, _ = state_passing(
        inc,
        m_chunk,
        initial_states=None,
        compute_dtype=torch.float32,
    )
    chunk_starts = chunk_starts.detach().requires_grad_(True)

    y = chunk_scan(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        B_prev=B_prev,
        U_prev=U_prev,
        T=U.shape[2],
        chunk_size=32,
        output_dtype=torch.float32,
        compute_dtype=torch.float32,
    )
    d_out = torch.randn_like(y)
    loss = (y * d_out).sum()

    (d_chunk_starts_ref,) = torch.autograd.grad(loss, (chunk_starts,))
    d_chunk_starts_cute = chunk_scan_bwd_dz0_cute(
        M.detach(),
        C.detach(),
        d_out.detach(),
        chunk_size=32,
        compute_dtype=torch.float32,
    )

    torch.testing.assert_close(
        d_chunk_starts_cute,
        d_chunk_starts_ref,
        atol=5e-5,
        rtol=0.0,
    )
