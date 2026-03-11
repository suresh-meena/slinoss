from __future__ import annotations

import math

import pytest
import torch

from slinoss.ops.v2x2ssd.cute.kernels.bwd.state_passing import (
    compile_state_passing_bwd_kernels,
    state_passing_bwd_m_cute,
    state_passing_bwd_state_cute,
    state_passing_bwd_cute,
)


def _as_complex_pairs(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"Expected even trailing dimension. Got {tuple(x.shape)}.")
    return torch.view_as_complex(
        x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2).contiguous()
    )


def _pack_complex_pairs(z: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(z).reshape(*z.shape[:-1], z.shape[-1] * 2).contiguous()


def _state_passing_autograd(
    inc: torch.Tensor,
    m_chunk: torch.Tensor,
    initial_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inc_c = _as_complex_pairs(inc)
    m_c = torch.view_as_complex(m_chunk.contiguous())
    z = _as_complex_pairs(initial_states)

    starts: list[torch.Tensor] = []
    for c in range(int(inc.shape[2])):
        starts.append(z)
        z = m_c[:, :, c].unsqueeze(-1).unsqueeze(-1) * z + inc_c[:, :, c]

    chunk_starts = _pack_complex_pairs(torch.stack(starts, dim=2))
    final_state = _pack_complex_pairs(z)
    return chunk_starts, final_state


def _make_inputs(
    *,
    batch: int,
    heads: int,
    chunks: int,
    N: int,
    P: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inc = torch.randn(
        (batch, heads, chunks, P, 2 * N), device=device, dtype=torch.float32
    )
    radius = 0.6 + 0.35 * torch.rand((batch, heads, chunks), device=device)
    angle = (2.0 * math.pi) * torch.rand(
        (batch, heads, chunks), device=device
    ) - math.pi
    m_chunk = (
        torch.view_as_real(torch.polar(radius, angle)).to(torch.float32).contiguous()
    )
    initial_states = torch.randn(
        (batch, heads, P, 2 * N), device=device, dtype=torch.float32
    )
    return inc, m_chunk, initial_states


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_state_passing_bwd_state_cute_matches_autograd() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    inc, m_chunk, initial = _make_inputs(
        batch=2,
        heads=2,
        chunks=12,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    inc.requires_grad_(True)
    m_chunk.requires_grad_(True)
    initial.requires_grad_(True)

    chunk_starts, final_state = _state_passing_autograd(inc, m_chunk, initial)
    d_chunk_starts = torch.randn_like(chunk_starts)
    d_final = torch.randn_like(final_state)
    loss = (chunk_starts * d_chunk_starts).sum() + (final_state * d_final).sum()

    d_inc_ref, _, d_initial_ref = torch.autograd.grad(loss, (inc, m_chunk, initial))
    d_inc_cute, d_initial_cute = state_passing_bwd_state_cute(
        d_chunk_starts.detach(),
        d_final.detach(),
        m_chunk.detach(),
    )

    torch.testing.assert_close(d_inc_cute, d_inc_ref, atol=2e-4, rtol=0.0)
    torch.testing.assert_close(d_initial_cute, d_initial_ref, atol=2e-4, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_state_passing_bwd_m_cute_matches_autograd() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    inc, m_chunk, initial = _make_inputs(
        batch=2,
        heads=2,
        chunks=12,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    inc.requires_grad_(True)
    m_chunk.requires_grad_(True)
    initial.requires_grad_(True)

    chunk_starts, final_state = _state_passing_autograd(inc, m_chunk, initial)
    d_chunk_starts = torch.randn_like(chunk_starts)
    d_final = torch.randn_like(final_state)
    loss = (chunk_starts * d_chunk_starts).sum() + (final_state * d_final).sum()

    _, d_m_ref, _ = torch.autograd.grad(loss, (inc, m_chunk, initial))
    d_inc_ref, _ = state_passing_bwd_state_cute(
        d_chunk_starts.detach(),
        d_final.detach(),
        m_chunk.detach(),
    )

    d_m_cute_f32 = state_passing_bwd_m_cute(
        chunk_starts.detach().to(dtype=torch.float32),
        d_inc_ref,
    )
    d_m_cute_f16 = state_passing_bwd_m_cute(
        chunk_starts.detach().to(dtype=torch.float16),
        d_inc_ref,
    )

    torch.testing.assert_close(d_m_cute_f32, d_m_ref, atol=2e-5, rtol=0.0)
    torch.testing.assert_close(d_m_cute_f16, d_m_ref, atol=2e-2, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compile_state_passing_bwd_kernels_matches_wrapper() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    inc, m_chunk, initial = _make_inputs(
        batch=2,
        heads=2,
        chunks=12,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )

    chunk_starts, final_state = _state_passing_autograd(inc, m_chunk, initial)
    d_chunk_starts = torch.randn_like(chunk_starts)
    d_final = torch.randn_like(final_state)

    d_inc_ref, d_m_ref, d_initial_ref = state_passing_bwd_cute(
        d_chunk_starts.detach(),
        d_final.detach(),
        chunk_starts.detach().to(dtype=torch.float16),
        m_chunk.detach(),
    )

    _, _, d_inc, d_m_chunk, d_initial, launch_pipeline = (
        compile_state_passing_bwd_kernels(
            chunk_starts.detach().to(dtype=torch.float16),
            m_chunk.detach(),
            d_chunk_starts=d_chunk_starts.detach(),
            d_final=d_final.detach(),
            return_launchers=True,
        )
    )
    launch_pipeline()

    torch.testing.assert_close(d_inc, d_inc_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(d_m_chunk, d_m_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(d_initial, d_initial_ref, atol=0.0, rtol=0.0)
