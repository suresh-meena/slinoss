"""CuTe backward kernels for the ``v2x2ssd`` chunk-increment stage."""

from __future__ import annotations

from typing import Callable

import torch

from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_increment import (
    _prepare_chunk_increment_operands,
)
from slinoss.ops.v2x2ssd.reference import _pack_complex_pairs

from .boundary import chunk_increment_bwd_boundary_cute
from .common import (
    prepare_chunk_increment_bwd_context,
    reshape_d_inc,
    validate_d_m_chunk,
    validate_prepared_state,
)
from .db import chunk_increment_bwd_db_cute
from .du import chunk_increment_bwd_du_cute
from .param_scan import chunk_increment_bwd_param_scan_cute


def compile_chunk_increment_bwd_prepared_kernels(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    A_main: torch.Tensor,
    B_main: torch.Tensor,
    u_head: torch.Tensor,
    b_head: torch.Tensor,
    d_inc: torch.Tensor,
    d_m_chunk: torch.Tensor,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
    return_launchers: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Callable[[], None],
    Callable[[], None],
]:
    """Compile the split chunk-increment backward pipeline from prepared state."""
    ctx = prepare_chunk_increment_bwd_context(
        U,
        M,
        K,
        B,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
    )
    validate_prepared_state(
        A_main=A_main,
        B_main=B_main,
        u_head=u_head,
        b_head=b_head,
        ctx=ctx,
    )
    d_inc_flat = reshape_d_inc(d_inc, ctx)
    d_m_chunk_flat = validate_d_m_chunk(d_m_chunk, ctx)

    dU = torch.empty((ctx.batch_size, ctx.n_heads, ctx.T, ctx.P), device=ctx.device, dtype=ctx.rdtype)
    dM = torch.empty((ctx.batch_size, ctx.n_heads, ctx.T, 2), device=ctx.device, dtype=ctx.rdtype)
    dK = torch.empty((ctx.batch_size, ctx.n_heads, ctx.T, 2, 2), device=ctx.device, dtype=ctx.rdtype)
    dB = torch.empty((ctx.batch_size, ctx.n_heads, ctx.T, ctx.D), device=ctx.device, dtype=ctx.rdtype)
    dB_prev_out = torch.empty((ctx.batch_size, ctx.n_heads, ctx.D), device=ctx.device, dtype=ctx.rdtype)
    dU_prev_out = torch.empty((ctx.batch_size, ctx.n_heads, ctx.P), device=ctx.device, dtype=ctx.rdtype)

    db_result = None
    du_result = None
    boundary_result = None
    param_result = None

    def _launch_db() -> None:
        nonlocal db_result
        db_result = chunk_increment_bwd_db_cute(
            A_main=A_main,
            d_inc_flat=d_inc_flat,
            ctx=ctx,
        )

    def _launch_du() -> None:
        nonlocal du_result
        du_result = chunk_increment_bwd_du_cute(
            B_main=B_main,
            b_head=b_head,
            d_inc_flat=d_inc_flat,
            ctx=ctx,
        )

    def _launch_boundary() -> None:
        nonlocal boundary_result
        boundary_result = chunk_increment_bwd_boundary_cute(
            u_head=u_head,
            d_inc_flat=d_inc_flat,
            ctx=ctx,
        )

    def _launch_scan() -> None:
        nonlocal param_result
        if db_result is None or boundary_result is None:
            raise RuntimeError("db and boundary must be launched before param_scan.")
        param_result = chunk_increment_bwd_param_scan_cute(
            d_alpha=db_result.d_alpha,
            d_boundary=boundary_result.d_boundary,
            d_m_chunk_flat=d_m_chunk_flat,
            ctx=ctx,
        )

    def _assemble_outputs() -> None:
        if db_result is None or du_result is None or boundary_result is None or param_result is None:
            raise RuntimeError("chunk_increment backward pipeline was not fully launched.")
        dB_blk = db_result.dB_blk.clone()
        if ctx.n_chunks > 1:
            dB_blk[:, :, :-1, -1, :] += boundary_result.d_b_prev_chunk0[:, :, 1:, :]

        dU.copy_(du_result.dU)
        dU_prev_out.copy_(du_result.dU_prev)
        dB_prev_out.copy_(boundary_result.dB_prev)
        dM.copy_(param_result.dM)
        dK.copy_(param_result.dK)
        dB.copy_(
            _pack_complex_pairs(
                dB_blk.reshape(ctx.batch_size, ctx.n_heads, ctx.T_pad, ctx.N),
                real_dtype=ctx.rdtype,
            )[:, :, : ctx.T, :].contiguous()
        )

    def launch_sequential() -> None:
        _launch_db()
        _launch_du()
        _launch_boundary()
        _launch_scan()
        _assemble_outputs()

    def launch_overlapped() -> None:
        stream_db = torch.cuda.Stream(device=U.device)
        stream_du = torch.cuda.Stream(device=U.device)
        stream_boundary = torch.cuda.Stream(device=U.device)
        ev_start = torch.cuda.Event(blocking=False, enable_timing=False)
        ev_db_done = torch.cuda.Event(blocking=False, enable_timing=False)
        ev_boundary_done = torch.cuda.Event(blocking=False, enable_timing=False)
        ev_du_done = torch.cuda.Event(blocking=False, enable_timing=False)

        current = torch.cuda.current_stream(device=U.device)
        current.record_event(ev_start)
        stream_db.wait_event(ev_start)
        stream_du.wait_event(ev_start)
        stream_boundary.wait_event(ev_start)

        with torch.cuda.stream(stream_db):
            _launch_db()
            stream_db.record_event(ev_db_done)
        with torch.cuda.stream(stream_du):
            _launch_du()
            stream_du.record_event(ev_du_done)
        with torch.cuda.stream(stream_boundary):
            _launch_boundary()
            stream_boundary.record_event(ev_boundary_done)

        current.wait_event(ev_db_done)
        current.wait_event(ev_boundary_done)
        _launch_scan()
        current.wait_event(ev_du_done)
        _assemble_outputs()

    if return_launchers:
        return dU, dM, dK, dB, dB_prev_out, dU_prev_out, launch_sequential, launch_overlapped

    launch_sequential()
    return dU, dM, dK, dB, dB_prev_out, dU_prev_out


def compile_chunk_increment_bwd_kernels(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    d_inc: torch.Tensor,
    d_m_chunk: torch.Tensor,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
    return_launchers: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Callable[[], None],
    Callable[[], None],
]:
    """Compile the split chunk-increment backward pipeline from the public contract."""
    A_main, B_main, u_head, b_head, _, _, _, _, _ = _prepare_chunk_increment_operands(
        U,
        M,
        K,
        B,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
    )
    return compile_chunk_increment_bwd_prepared_kernels(
        U,
        M,
        K,
        B,
        A_main=A_main,
        B_main=B_main,
        u_head=u_head,
        b_head=b_head,
        d_inc=d_inc,
        d_m_chunk=d_m_chunk,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
        return_launchers=return_launchers,
    )


def chunk_increment_bwd_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    d_inc: torch.Tensor,
    d_m_chunk: torch.Tensor,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Canonical backward entrypoint for the public chunk-increment contract."""
    return compile_chunk_increment_bwd_kernels(
        U,
        M,
        K,
        B,
        d_inc=d_inc,
        d_m_chunk=d_m_chunk,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
    )


def chunk_increment_bwd_prepared_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    A_main: torch.Tensor,
    B_main: torch.Tensor,
    u_head: torch.Tensor,
    b_head: torch.Tensor,
    d_inc: torch.Tensor,
    d_m_chunk: torch.Tensor,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Canonical backward entrypoint when forward-prepared operands are available."""
    return compile_chunk_increment_bwd_prepared_kernels(
        U,
        M,
        K,
        B,
        A_main=A_main,
        B_main=B_main,
        u_head=u_head,
        b_head=b_head,
        d_inc=d_inc,
        d_m_chunk=d_m_chunk,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
    )


__all__ = [
    "chunk_increment_bwd_cute",
    "chunk_increment_bwd_prepared_cute",
    "compile_chunk_increment_bwd_kernels",
    "compile_chunk_increment_bwd_prepared_kernels",
]
