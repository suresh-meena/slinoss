from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from slinoss.ops.v2x2ssd.reference import v2x2ssm, v2x2ssd, v2x2ssd_ref


@dataclass(frozen=True)
class Inputs:
    U: torch.Tensor
    M: torch.Tensor
    K: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    initial_states: torch.Tensor | None
    B_prev: torch.Tensor | None
    U_prev: torch.Tensor | None


def _pack_complex_pairs(z: torch.Tensor, *, real_dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.view_as_real(z)
        .reshape(*z.shape[:-1], z.shape[-1] * 2)
        .to(dtype=real_dtype)
        .contiguous()
    )


def _make_inputs(
    *,
    b: int,
    h: int,
    T: int,
    N: int,
    P: int,
    dtype: torch.dtype,
    device: torch.device,
    streaming: bool,
    with_init: bool,
) -> Inputs:
    radius = 0.6 + 0.35 * torch.rand((b, h, T), device=device, dtype=torch.float32)
    angle = (2.0 * math.pi) * torch.rand(
        (b, h, T), device=device, dtype=torch.float32
    ) - math.pi
    M_complex = torch.polar(radius, angle)
    M = torch.view_as_real(M_complex).to(dtype=dtype).contiguous()

    K_complex = (
        torch.randn((b, h, T, 2), device=device, dtype=torch.float32)
        + 1j * torch.randn((b, h, T, 2), device=device, dtype=torch.float32)
    ) * 0.1
    K = torch.view_as_real(K_complex).to(dtype=dtype).contiguous()

    U = torch.randn((b, h, T, P), device=device, dtype=dtype)
    B = torch.randn((b, h, T, 2 * N), device=device, dtype=dtype) * 0.1
    C = torch.randn((b, h, T, 2 * N), device=device, dtype=dtype) * 0.1

    initial_states = None
    if with_init:
        z0 = (
            torch.randn((b, h, P, N), device=device, dtype=torch.float32)
            + 1j * torch.randn((b, h, P, N), device=device, dtype=torch.float32)
        ) * 0.1
        initial_states = _pack_complex_pairs(z0, real_dtype=dtype)

    B_prev = None
    U_prev = None
    if streaming:
        b_prev = (
            torch.randn((b, h, N), device=device, dtype=torch.float32)
            + 1j * torch.randn((b, h, N), device=device, dtype=torch.float32)
        ) * 0.1
        B_prev = _pack_complex_pairs(b_prev, real_dtype=dtype)
        U_prev = torch.randn((b, h, P), device=device, dtype=dtype)

    return Inputs(
        U=U,
        M=M,
        K=K,
        B=B,
        C=C,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
    )


def test_reference_stack_parity_float64() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    inputs = _make_inputs(
        b=2,
        h=3,
        T=9,
        N=4,
        P=5,
        dtype=torch.float32,
        device=device,
        streaming=True,
        with_init=True,
    )

    y_ssm, s_ssm, b_last_ssm, u_last_ssm = v2x2ssm(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        inputs.C,
        initial_states=inputs.initial_states,
        B_prev=inputs.B_prev,
        U_prev=inputs.U_prev,
        compute_dtype=torch.float64,
        output_dtype=torch.float64,
    )
    y_ref, s_ref, b_last_ref, u_last_ref = v2x2ssd_ref(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        inputs.C,
        chunk_size=4,
        initial_states=inputs.initial_states,
        B_prev=inputs.B_prev,
        U_prev=inputs.U_prev,
        compute_dtype=torch.float64,
        output_dtype=torch.float64,
    )
    y_fwd, s_fwd, b_last_fwd, u_last_fwd = v2x2ssd(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        inputs.C,
        chunk_size=4,
        initial_states=inputs.initial_states,
        B_prev=inputs.B_prev,
        U_prev=inputs.U_prev,
        compute_dtype=torch.float64,
        output_dtype=torch.float64,
    )

    assert torch.allclose(y_ssm, y_ref, atol=1e-10, rtol=0.0)
    assert torch.allclose(y_ssm, y_fwd, atol=1e-10, rtol=0.0)
    assert torch.allclose(s_ssm, s_ref, atol=1e-10, rtol=0.0)
    assert torch.allclose(s_ssm, s_fwd, atol=1e-10, rtol=0.0)
    assert torch.equal(b_last_ssm, b_last_ref)
    assert torch.equal(b_last_ssm, b_last_fwd)
    assert torch.equal(u_last_ssm, u_last_ref)
    assert torch.equal(u_last_ssm, u_last_fwd)


def test_reference_stack_parity_float32() -> None:
    torch.manual_seed(1)
    device = torch.device("cpu")
    inputs = _make_inputs(
        b=2,
        h=2,
        T=13,
        N=3,
        P=4,
        dtype=torch.float32,
        device=device,
        streaming=True,
        with_init=True,
    )

    y_ssm, s_ssm, _, _ = v2x2ssm(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        inputs.C,
        initial_states=inputs.initial_states,
        B_prev=inputs.B_prev,
        U_prev=inputs.U_prev,
    )
    y_ref, s_ref, _, _ = v2x2ssd_ref(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        inputs.C,
        chunk_size=8,
        initial_states=inputs.initial_states,
        B_prev=inputs.B_prev,
        U_prev=inputs.U_prev,
    )
    y_fwd, s_fwd, _, _ = v2x2ssd(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        inputs.C,
        chunk_size=8,
        initial_states=inputs.initial_states,
        B_prev=inputs.B_prev,
        U_prev=inputs.U_prev,
    )

    assert torch.allclose(y_ssm, y_ref, atol=2e-5, rtol=0.0)
    assert torch.allclose(y_ssm, y_fwd, atol=2e-5, rtol=0.0)
    assert torch.allclose(s_ssm, s_ref, atol=2e-5, rtol=0.0)
    assert torch.allclose(s_ssm, s_fwd, atol=2e-5, rtol=0.0)


def test_zero_length_returns_initial_and_prev() -> None:
    torch.manual_seed(2)
    device = torch.device("cpu")
    b, h, T, N, P = 2, 2, 0, 3, 4
    inputs = _make_inputs(
        b=b,
        h=h,
        T=T,
        N=N,
        P=P,
        dtype=torch.float32,
        device=device,
        streaming=True,
        with_init=True,
    )

    assert inputs.initial_states is not None
    assert inputs.B_prev is not None
    assert inputs.U_prev is not None

    y, final_state, b_last, u_last = v2x2ssd(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        inputs.C,
        chunk_size=4,
        initial_states=inputs.initial_states,
        B_prev=inputs.B_prev,
        U_prev=inputs.U_prev,
    )

    assert y.shape == (b, h, 0, P)
    assert torch.equal(final_state, inputs.initial_states)
    assert torch.equal(b_last, inputs.B_prev)
    assert torch.equal(u_last, inputs.U_prev)


def test_zero_length_defaults_to_zero_state_and_prev() -> None:
    device = torch.device("cpu")
    b, h, T, N, P = 1, 2, 0, 2, 3
    inputs = _make_inputs(
        b=b,
        h=h,
        T=T,
        N=N,
        P=P,
        dtype=torch.float32,
        device=device,
        streaming=False,
        with_init=False,
    )

    y, final_state, b_last, u_last = v2x2ssd_ref(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        inputs.C,
        chunk_size=8,
    )

    assert y.shape == (b, h, 0, P)
    assert torch.equal(final_state, torch.zeros((b, h, P, 2 * N)))
    assert torch.equal(b_last, torch.zeros((b, h, 2 * N)))
    assert torch.equal(u_last, torch.zeros((b, h, P)))


def test_chunked_paths_reject_exact_zero_transitions() -> None:
    torch.manual_seed(3)
    device = torch.device("cpu")
    inputs = _make_inputs(
        b=2,
        h=2,
        T=11,
        N=3,
        P=4,
        dtype=torch.float32,
        device=device,
        streaming=True,
        with_init=True,
    )

    M = inputs.M.clone()
    M[:, :, 2, :] = 0.0
    M[:, :, 7, :] = 0.0
    inputs = Inputs(
        U=inputs.U,
        M=M,
        K=inputs.K,
        B=inputs.B,
        C=inputs.C,
        initial_states=inputs.initial_states,
        B_prev=inputs.B_prev,
        U_prev=inputs.U_prev,
    )

    y_ssm, s_ssm, _, _ = v2x2ssm(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        inputs.C,
        initial_states=inputs.initial_states,
        B_prev=inputs.B_prev,
        U_prev=inputs.U_prev,
        compute_dtype=torch.float64,
        output_dtype=torch.float64,
    )

    assert torch.isfinite(y_ssm).all()
    assert torch.isfinite(s_ssm).all()

    with pytest.raises(ValueError, match="strictly nonzero"):
        v2x2ssd_ref(
            inputs.U,
            inputs.M,
            inputs.K,
            inputs.B,
            inputs.C,
            chunk_size=4,
            initial_states=inputs.initial_states,
            B_prev=inputs.B_prev,
            U_prev=inputs.U_prev,
            compute_dtype=torch.float64,
            output_dtype=torch.float64,
        )

    with pytest.raises(ValueError, match="strictly nonzero"):
        v2x2ssd(
            inputs.U,
            inputs.M,
            inputs.K,
            inputs.B,
            inputs.C,
            chunk_size=4,
            initial_states=inputs.initial_states,
            B_prev=inputs.B_prev,
            U_prev=inputs.U_prev,
            compute_dtype=torch.float64,
            output_dtype=torch.float64,
        )


def test_non_finite_inputs_raise_cleanly() -> None:
    torch.manual_seed(4)
    device = torch.device("cpu")
    inputs = _make_inputs(
        b=1,
        h=1,
        T=5,
        N=2,
        P=3,
        dtype=torch.float32,
        device=device,
        streaming=False,
        with_init=False,
    )

    M = inputs.M.clone()
    M[0, 0, 1, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        v2x2ssd_ref(
            inputs.U,
            M,
            inputs.K,
            inputs.B,
            inputs.C,
            chunk_size=4,
        )
