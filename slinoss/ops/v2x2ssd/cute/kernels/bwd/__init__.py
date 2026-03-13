"""Thin CuTe host-JIT boundary for training-only ``v2x2ssd`` backward."""

from __future__ import annotations

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

from slinoss.perf import note_cache_event, record_region

from ..fwd.common import (
    _assumed_align,
    _pad_m_identity,
    _pad_zero_time,
    _tc_input_dtype,
    _torch_to_cutlass_dtype,
)
from .chunk_increment.boundary import ChunkIncrementBwdBoundaryAmpere
from .chunk_increment.db import ChunkIncrementBwdDBAmpere
from .chunk_increment.du import ChunkIncrementBwdDUAmpere
from .chunk_increment.param_scan import ChunkIncrementBwdParamScanAmpere
from .chunk_scan import (
    _fold_chunk_boundary_carries,
    _public_dk_from_parts,
    _public_from_chunked,
    _public_from_param_scan,
)
from .chunk_scan.db import ChunkScanBwdDBAmpere
from .chunk_scan.dcdr import ChunkScanBwdDCDRAmpere
from .chunk_scan.du import ChunkScanBwdDUAmpere
from .chunk_scan.dz0 import ChunkScanBwdDZ0Ampere
from .chunk_scan.param_scan import ChunkScanBwdParamScanAmpere
from .state_passing.common import _TileConfig, _choose_copy_bits_for_linear_tiles
from .state_passing.m import StatePassingBwdMAmpere
from .state_passing.state import StatePassingBwdStateAmpere


_BWD_HOST_CACHE: dict[tuple, object] = {}
_ZERO_PREV_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_ZERO_FINAL_GRAD_CACHE: dict[tuple, torch.Tensor] = {}
_BWD_WORKSPACE_CACHE: dict[tuple, tuple[torch.Tensor, ...]] = {}


def _get_zero_prev_tensors(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    heads: int,
    P: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (
        device.type,
        device.index if device.index is not None else -1,
        dtype,
        int(batch_size),
        int(heads),
        int(P),
        int(D),
    )
    cached = _ZERO_PREV_CACHE.get(key)
    if cached is None:
        cached = (
            torch.zeros((batch_size, heads, P), device=device, dtype=dtype),
            torch.zeros((batch_size, heads, D), device=device, dtype=dtype),
        )
        _ZERO_PREV_CACHE[key] = cached
    return cached


def _get_zero_final_grad(
    *,
    device: torch.device,
    batch_size: int,
    heads: int,
    P: int,
    D: int,
) -> torch.Tensor:
    key = (
        device.type,
        device.index if device.index is not None else -1,
        int(batch_size),
        int(heads),
        int(P),
        int(D),
    )
    cached = _ZERO_FINAL_GRAD_CACHE.get(key)
    if cached is None:
        cached = torch.zeros(
            (batch_size, heads, P, D), device=device, dtype=torch.float32
        )
        _ZERO_FINAL_GRAD_CACHE[key] = cached
    return cached


def _get_bwd_workspace(
    *,
    device: torch.device,
    tc_dtype: torch.dtype,
    batch_size: int,
    heads: int,
    n_chunks: int,
    chunk_size: int,
    P: int,
    D: int,
    n_d_tiles: int,
) -> tuple[torch.Tensor, ...]:
    key = (
        device.type,
        device.index if device.index is not None else -1,
        tc_dtype,
        int(batch_size),
        int(heads),
        int(n_chunks),
        int(chunk_size),
        int(P),
        int(D),
        int(n_d_tiles),
    )
    cached = _BWD_WORKSPACE_CACHE.get(key)
    if cached is not None:
        return cached

    BH = int(batch_size) * int(heads)
    BHC = BH * int(n_chunks)
    L = int(chunk_size)

    cached = (
        torch.empty((BH, n_chunks, P), device=device, dtype=tc_dtype),
        torch.empty((BH, n_chunks, D), device=device, dtype=tc_dtype),
        torch.empty((BHC, P, D), device=device, dtype=torch.float32),
        torch.empty(
            (batch_size, heads, n_chunks, P, D), device=device, dtype=torch.float32
        ),
        torch.empty((batch_size, heads, n_chunks, P, D), device=device, dtype=tc_dtype),
        torch.empty((batch_size, heads, P, D), device=device, dtype=torch.float32),
        torch.empty(
            (batch_size, heads, n_chunks, 2), device=device, dtype=torch.float32
        ),
        torch.empty((BHC, L, 1, P), device=device, dtype=tc_dtype),
        torch.empty((BHC, P), device=device, dtype=tc_dtype),
        torch.empty((BHC, L, 1, D), device=device, dtype=tc_dtype),
        torch.empty((BHC, D), device=device, dtype=tc_dtype),
        torch.empty((BHC, L, 1, D), device=device, dtype=tc_dtype),
        torch.empty((BHC, L, 1, P), device=device, dtype=tc_dtype),
        torch.empty((BHC, P), device=device, dtype=tc_dtype),
        torch.empty((BHC, L, 1, D), device=device, dtype=tc_dtype),
        torch.empty((BHC, D), device=device, dtype=tc_dtype),
        torch.empty((BHC, L), device=device, dtype=torch.float32),
        torch.empty((BHC, L, 4), device=device, dtype=torch.float32),
        torch.empty((BHC, L, 2), device=device, dtype=torch.float32),
        torch.empty((BHC, L, 2), device=device, dtype=torch.float32),
        torch.empty((BHC, 1, L, 2), device=device, dtype=torch.float32),
        torch.empty((BHC, 1, L, 2), device=device, dtype=torch.float32),
        torch.empty((BHC, 1, L, 2), device=device, dtype=torch.float32),
        torch.empty((BHC, L, D), device=device, dtype=tc_dtype),
        torch.empty((BHC, D), device=device, dtype=tc_dtype),
        torch.empty((BHC, L, P), device=device, dtype=tc_dtype),
        torch.empty((BHC, P), device=device, dtype=tc_dtype),
        torch.empty((2, L, n_d_tiles, BHC), device=device, dtype=torch.float32),
        torch.empty((2, BHC), device=device, dtype=torch.float32),
        torch.empty((2, L, BHC), device=device, dtype=torch.float32),
        torch.empty((2, L, BHC), device=device, dtype=torch.float32),
        torch.empty((2, L, BHC), device=device, dtype=torch.float32),
    )
    _BWD_WORKSPACE_CACHE[key] = cached
    return cached


def _make_row_major_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    stride = [1] * len(shape)
    running = 1
    for i in range(len(shape) - 1, -1, -1):
        stride[i] = running
        running *= int(shape[i])
    return tuple(stride)


def _make_ptr_arg(t: torch.Tensor) -> tuple[object, int]:
    align = _assumed_align(t)
    ptr = make_ptr(
        _torch_to_cutlass_dtype(t.dtype),
        t.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=align,
    )
    return ptr, align


def _make_cast_f32_to_tc(*, total_elems: int, num_threads: int = 256):
    grid_x = (int(total_elems) + int(num_threads) - 1) // int(num_threads)

    @cute.kernel
    def _cast_kernel(gSrc: cute.Tensor, gDst: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        idx = bidx * int(num_threads) + tidx
        if idx < int(total_elems):
            gDst[idx] = gSrc[idx].to(gDst.element_type)

    return _cast_kernel, int(grid_x), int(num_threads)


def _bwd_host_cache_key(
    *,
    device_index: int,
    tc_dtype: torch.dtype,
    U_shape: tuple[int, ...],
    M_shape: tuple[int, ...],
    K_shape: tuple[int, ...],
    B_shape: tuple[int, ...],
    C_shape: tuple[int, ...],
    m_chunk_shape: tuple[int, ...],
    chunk_starts_shape: tuple[int, ...],
    d_out_shape: tuple[int, ...],
    chunk_size: int,
    scan_num_threads_du: int,
    scan_num_threads_db: int,
    scan_num_threads_dc: int,
    scan_num_threads_param: int,
    state_num_threads: int,
    state_pairs_per_thread: int,
    state_copy_bits_state: int,
    state_copy_bits_out: int,
    n_d_tiles: int,
    alignments: tuple[int, ...],
) -> tuple:
    return (
        "v2x2ssd_bwd_host",
        device_index,
        tc_dtype,
        U_shape,
        M_shape,
        K_shape,
        B_shape,
        C_shape,
        m_chunk_shape,
        chunk_starts_shape,
        d_out_shape,
        int(chunk_size),
        int(scan_num_threads_du),
        int(scan_num_threads_db),
        int(scan_num_threads_dc),
        int(scan_num_threads_param),
        int(state_num_threads),
        int(state_pairs_per_thread),
        int(state_copy_bits_state),
        int(state_copy_bits_out),
        int(n_d_tiles),
        alignments,
    )


def _build_backward_args(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    m_chunk: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    scan_num_threads_du: int,
    scan_num_threads_db: int,
    scan_num_threads_dc: int,
    scan_num_threads_param: int,
    state_num_threads: int,
    state_pairs_per_thread: int,
) -> tuple[
    list[object],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[torch.Tensor, ...],
]:
    if U.device.type != "cuda":
        raise ValueError("CUDA tensor required.")
    if U.dtype != B.dtype or U.dtype != C.dtype:
        raise ValueError("U/B/C must share dtype.")
    if U.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("U/B/C must be float16/bfloat16/float32.")
    if M.dtype != torch.float32 or K.dtype != torch.float32:
        raise TypeError("M and K must be float32.")
    if d_out.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("d_out must be float16/bfloat16/float32.")
    if m_chunk.dtype != torch.float32 or chunk_starts.dtype != torch.float32:
        raise TypeError("m_chunk and chunk_starts must be float32.")

    Bsz, H, T, P = map(int, U.shape)
    D = int(B.shape[-1])
    if B.shape != (Bsz, H, T, D) or C.shape != (Bsz, H, T, D):
        raise ValueError("B/C must be (B,H,T,D) matching U.")
    if M.shape != (Bsz, H, T, 2):
        raise ValueError(f"M must be (B,H,T,2)={(Bsz, H, T, 2)}.")
    if K.shape != (Bsz, H, T, 2, 2):
        raise ValueError(f"K must be (B,H,T,2,2)={(Bsz, H, T, 2, 2)}.")
    if d_out.shape != (Bsz, H, T, P):
        raise ValueError("d_out must be (B,H,T,P) matching U.")
    if D % 2 != 0:
        raise ValueError("D must be divisible by 2.")

    L = int(chunk_size)
    if L <= 0:
        raise ValueError("chunk_size must be positive.")
    n_chunks = (T + L - 1) // L
    T_pad = n_chunks * L
    if tuple(m_chunk.shape) != (Bsz, H, n_chunks, 2):
        raise ValueError(
            f"m_chunk must be (B,H,C,2)={(Bsz, H, n_chunks, 2)}. "
            f"Got {tuple(m_chunk.shape)}."
        )
    if tuple(chunk_starts.shape) != (Bsz, H, n_chunks, P, D):
        raise ValueError(
            f"chunk_starts must be (B,H,C,P,D)={(Bsz, H, n_chunks, P, D)}. "
            f"Got {tuple(chunk_starts.shape)}."
        )

    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)

    U_tc = _pad_zero_time(U, T_pad=T_pad, dtype=tc_dtype)
    B_tc = _pad_zero_time(B, T_pad=T_pad, dtype=tc_dtype)
    C_tc = _pad_zero_time(C, T_pad=T_pad, dtype=tc_dtype)
    d_out_tc = _pad_zero_time(d_out, T_pad=T_pad, dtype=tc_dtype)
    M_f = _pad_m_identity(M, T_pad=T_pad)
    K_f = _pad_zero_time(K, T_pad=T_pad, dtype=torch.float32)
    m_chunk_f = m_chunk.to(dtype=torch.float32).contiguous()
    chunk_starts_f = chunk_starts.to(dtype=torch.float32).contiguous()

    U_prev0, B_prev0 = _get_zero_prev_tensors(
        device=U.device,
        dtype=tc_dtype,
        batch_size=Bsz,
        heads=H,
        P=P,
        D=D,
    )
    d_final = _get_zero_final_grad(
        device=U.device,
        batch_size=Bsz,
        heads=H,
        P=P,
        D=D,
    )

    BH = Bsz * H
    U_chunks = U_tc.reshape(BH, n_chunks, L, P)
    B_chunks = B_tc.reshape(BH, n_chunks, L, D)

    cutlass_dtype = _torch_to_cutlass_dtype(tc_dtype)
    inc_db_kernel = ChunkIncrementBwdDBAmpere(cutlass_dtype, chunk_size=L, D=D, P=P)
    n_d_tiles = (D + inc_db_kernel.bN - 1) // inc_db_kernel.bN

    (
        U_prev_chunks,
        B_prev_chunks,
        dZ0,
        d_inc,
        d_inc_tc,
        d_initial,
        d_m_chunk,
        dU_scan,
        dU_prev_scan,
        dB_scan,
        dB_prev_scan,
        dC_scan,
        dU_db_dummy,
        dU_prev_db_dummy,
        dB_du_dummy,
        dB_prev_du_dummy,
        dlogp,
        dR,
        dMp_scratch,
        dMc_scratch,
        dM_scan,
        dKprev_scan,
        dKcurr_scan,
        dB_inc,
        dB_prev_inc,
        dU_inc,
        dU_prev_inc,
        dMsum_part,
        dMp0,
        dM_inc,
        dKprev_inc,
        dKcurr_inc,
    ) = _get_bwd_workspace(
        device=U.device,
        tc_dtype=tc_dtype,
        batch_size=Bsz,
        heads=H,
        n_chunks=n_chunks,
        chunk_size=L,
        P=P,
        D=D,
        n_d_tiles=n_d_tiles,
    )

    U_prev_chunks[:, 0, :] = U_prev0.reshape(BH, P)
    B_prev_chunks[:, 0, :] = B_prev0.reshape(BH, D)
    if n_chunks > 1:
        U_prev_chunks[:, 1:, :] = U_chunks[:, :-1, -1, :]
        B_prev_chunks[:, 1:, :] = B_chunks[:, :-1, -1, :]

    state_cfg = _TileConfig(
        num_threads=int(state_num_threads),
        pairs_per_thread=int(state_pairs_per_thread),
    )
    state_copy_bits_state = _choose_copy_bits_for_linear_tiles(
        dZ0.reshape(Bsz, H, n_chunks, P, D),
        tile_stride_elems=P * D,
        elems_per_thread=state_cfg.elems_per_thread,
    )
    state_copy_bits_out = _choose_copy_bits_for_linear_tiles(
        d_inc,
        tile_stride_elems=P * D,
        elems_per_thread=state_cfg.elems_per_thread,
    )

    Kprev_view = K_f[:, :, :, 0, :]
    Kcurr_view = K_f[:, :, :, 1, :]

    ptr_tensors = [
        U_tc,
        B_tc,
        C_tc,
        M_f,
        K_f,
        Kprev_view,
        Kcurr_view,
        d_out_tc,
        m_chunk_f,
        chunk_starts_f,
        U_prev0,
        B_prev0,
        d_final,
        U_prev_chunks,
        B_prev_chunks,
        dZ0,
        d_inc,
        d_inc_tc,
        d_initial,
        d_m_chunk,
        dU_scan,
        dU_prev_scan,
        dB_scan,
        dB_prev_scan,
        dC_scan,
        dU_db_dummy,
        dU_prev_db_dummy,
        dB_du_dummy,
        dB_prev_du_dummy,
        dlogp,
        dR,
        dMp_scratch,
        dMc_scratch,
        dM_scan,
        dKprev_scan,
        dKcurr_scan,
        dB_inc,
        dB_prev_inc,
        dU_inc,
        dU_prev_inc,
        dMsum_part,
        dMp0,
        dM_inc,
        dKprev_inc,
        dKcurr_inc,
    ]
    dynamic_args: list[object] = []
    alignments_list: list[int] = []
    for tensor in ptr_tensors:
        ptr, align = _make_ptr_arg(tensor)
        dynamic_args.append(ptr)
        alignments_list.append(align)

    spec = (
        Bsz,
        H,
        T_pad,
        P,
        D,
        n_chunks,
        L,
        n_d_tiles,
    )
    cfg = (
        int(scan_num_threads_du),
        int(scan_num_threads_db),
        int(scan_num_threads_dc),
        int(scan_num_threads_param),
        int(state_num_threads),
        int(state_pairs_per_thread),
        int(state_copy_bits_state),
        int(state_copy_bits_out),
    )

    dU_scan_view = dU_scan.reshape(Bsz, H, n_chunks, L, P)
    dU_prev_scan_view = dU_prev_scan.reshape(Bsz, H, n_chunks, P)
    dB_scan_view = dB_scan.reshape(Bsz, H, n_chunks, L, D)
    dB_prev_scan_view = dB_prev_scan.reshape(Bsz, H, n_chunks, D)
    dC_scan_view = dC_scan.reshape(Bsz, H, n_chunks, L, D)
    dM_scan_view = dM_scan.reshape(Bsz, H, n_chunks, 1, L, 2)
    dKprev_scan_view = dKprev_scan.reshape(Bsz, H, n_chunks, 1, L, 2)
    dKcurr_scan_view = dKcurr_scan.reshape(Bsz, H, n_chunks, 1, L, 2)

    dU_inc_view = dU_inc.reshape(Bsz, H, n_chunks, L, P)
    dU_prev_inc_view = dU_prev_inc.reshape(Bsz, H, n_chunks, P)
    dB_inc_view = dB_inc.reshape(Bsz, H, n_chunks, L, D)
    dB_prev_inc_view = dB_prev_inc.reshape(Bsz, H, n_chunks, D)
    dM_inc_view = dM_inc.permute(2, 1, 0).reshape(Bsz, H, n_chunks, L, 2)
    dKprev_inc_view = dKprev_inc.permute(2, 1, 0).reshape(Bsz, H, n_chunks, L, 2)
    dKcurr_inc_view = dKcurr_inc.permute(2, 1, 0).reshape(Bsz, H, n_chunks, L, 2)

    keepalive = (
        U_tc,
        B_tc,
        C_tc,
        d_out_tc,
        M_f,
        K_f,
        m_chunk_f,
        chunk_starts_f,
        U_prev0,
        B_prev0,
        d_final,
        U_prev_chunks,
        B_prev_chunks,
        d_inc_tc,
    )
    return (
        dynamic_args,
        tuple(alignments_list),
        spec,
        cfg,
        (
            dU_scan_view,
            dU_prev_scan_view,
            dB_scan_view,
            dB_prev_scan_view,
            dC_scan_view,
            dM_scan_view,
            dKprev_scan_view,
            dKcurr_scan_view,
            dU_inc_view,
            dU_prev_inc_view,
            dB_inc_view,
            dB_prev_inc_view,
            dM_inc_view,
            dKprev_inc_view,
            dKcurr_inc_view,
            *keepalive,
        ),
    )


def _make_v2x2ssd_bwd_host_wrapper(
    *,
    spec: tuple[int, ...],
    cfg: tuple[int, ...],
):
    Bsz, H, T_pad, P, D, n_chunks, L, n_d_tiles = spec
    (
        scan_num_threads_du,
        scan_num_threads_db,
        scan_num_threads_dc,
        scan_num_threads_param,
        state_num_threads,
        state_pairs_per_thread,
        state_copy_bits_state,
        state_copy_bits_out,
    ) = cfg
    BH = Bsz * H
    BHC = BH * n_chunks

    def _make_tensor(
        ptr: cute.Pointer, shape: tuple[int, ...], stride: tuple[int, ...]
    ):
        return cute.make_tensor(ptr, cute.make_layout(shape, stride=stride))

    u_scan_shape = (BHC, L, 1, P)
    u_scan_stride = (L * P, P, P, 1)
    b_scan_shape = (BHC, L, 1, D)
    b_scan_stride = (L * D, D, D, 1)
    m_scan_shape = (BHC, L, 2)
    m_scan_stride = (L * 2, 2, 1)
    k_scan_shape = (BHC, L, 2, 2)
    k_scan_stride = (L * 4, 4, 2, 1)
    z0_shape = (BHC, P, D)
    z0_stride = (P * D, D, 1)
    u_prev0_shape = (BH, P)
    u_prev0_stride = (P, 1)
    b_prev0_shape = (BH, D)
    b_prev0_stride = (D, 1)
    dlogp_shape = (BHC, L)
    dlogp_stride = (L, 1)
    dm_scratch_shape = (BHC, L, 2)
    dm_scratch_stride = (L * 2, 2, 1)
    dr_shape = (BHC, L, 4)
    dr_stride = (L * 4, 4, 1)
    dparam_scan_shape = (BHC, 1, L, 2)
    dparam_scan_stride = (L * 2, L * 2, 2, 1)
    dlp_param_shape = (BHC, 1, L)
    dlp_param_stride = (L, L, 1)
    dr_param_shape = (BHC, 1, L, 4)
    dr_param_stride = (L * 4, L * 4, 4, 1)
    flat_state_shape = (BHC * P * D,)
    flat_state_stride = (1,)

    dz0_dout_shape = (P, T_pad, BH)
    dz0_dout_stride = (1, P, T_pad * P)
    dz0_c_shape = (D, T_pad, BH)
    dz0_c_stride = (1, D, T_pad * D)
    dz0_m_shape = (2, T_pad, BH)
    dz0_m_stride = (1, 2, T_pad * 2)
    dz0_out_shape = (P, D, BHC)
    dz0_out_stride = (D, 1, P * D)

    state_shape = (Bsz, H, n_chunks, P, D)
    state_stride = _make_row_major_stride(state_shape)
    m_chunk_shape = (Bsz, H, n_chunks, 2)
    m_chunk_stride = _make_row_major_stride(m_chunk_shape)
    final_shape = (Bsz, H, P, D)
    final_stride = _make_row_major_stride(final_shape)

    u_inc_shape = (L, P, BHC)
    u_inc_stride = (P, 1, L * P)
    du_out_shape = (P, L, BHC)
    du_out_stride = (1, P, L * P)
    b_inc_shape = (L, D, BHC)
    b_inc_stride = (D, 1, L * D)
    m_inc_shape = (2, L, BHC)
    m_inc_stride = (1, 2, L * 2)
    k_inc_shape = (2, L, BHC)
    k_inc_stride = (1, 4, L * 4)
    d_inc_shape = (P, D, BHC)
    d_inc_stride = (D, 1, P * D)
    d_inc_dp_shape = (D, P, BHC)
    d_inc_dp_stride = (1, D, P * D)
    d_inc_boundary_shape = (BHC, P, D)
    d_inc_boundary_stride = (P * D, D, 1)
    prev_u_chunks_shape = (P, BHC)
    prev_u_chunks_stride = (1, P)
    prev_b_chunks_shape = (D, BHC)
    prev_b_chunks_stride = (1, D)
    dmsum_part_shape = (2, L, n_d_tiles, BHC)
    dmsum_part_stride = _make_row_major_stride(dmsum_part_shape)
    dmp0_shape = (2, BHC)
    dmp0_stride = _make_row_major_stride(dmp0_shape)
    dmchunk_shape = (2, BHC)
    dmchunk_stride = (1, 2)
    dparam_inc_shape = (2, L, BHC)
    dparam_inc_stride = _make_row_major_stride(dparam_inc_shape)
    d_dummy_u_shape = (BHC, L, 1, P)
    d_dummy_u_stride = (L * P, P, P, 1)
    d_dummy_b_shape = (BHC, L, 1, D)
    d_dummy_b_stride = (L * D, D, D, 1)
    d_dummy_u_prev_shape = (BHC, P)
    d_dummy_u_prev_stride = (P, 1)
    d_dummy_b_prev_shape = (BHC, D)
    d_dummy_b_prev_stride = (D, 1)

    @cute.jit
    def _v2x2ssd_bwd_host_wrapper(
        U_ptr: cute.Pointer,
        B_ptr: cute.Pointer,
        C_ptr: cute.Pointer,
        M_ptr: cute.Pointer,
        K_ptr: cute.Pointer,
        Kprev_ptr: cute.Pointer,
        Kcurr_ptr: cute.Pointer,
        dOut_ptr: cute.Pointer,
        m_chunk_ptr: cute.Pointer,
        chunk_starts_ptr: cute.Pointer,
        U_prev0_ptr: cute.Pointer,
        B_prev0_ptr: cute.Pointer,
        d_final_ptr: cute.Pointer,
        U_prev_chunks_ptr: cute.Pointer,
        B_prev_chunks_ptr: cute.Pointer,
        dZ0_ptr: cute.Pointer,
        d_inc_ptr: cute.Pointer,
        d_inc_tc_ptr: cute.Pointer,
        d_initial_ptr: cute.Pointer,
        d_m_chunk_ptr: cute.Pointer,
        dU_scan_ptr: cute.Pointer,
        dU_prev_scan_ptr: cute.Pointer,
        dB_scan_ptr: cute.Pointer,
        dB_prev_scan_ptr: cute.Pointer,
        dC_scan_ptr: cute.Pointer,
        dU_db_dummy_ptr: cute.Pointer,
        dU_prev_db_dummy_ptr: cute.Pointer,
        dB_du_dummy_ptr: cute.Pointer,
        dB_prev_du_dummy_ptr: cute.Pointer,
        dlogp_ptr: cute.Pointer,
        dR_ptr: cute.Pointer,
        dMp_scratch_ptr: cute.Pointer,
        dMc_scratch_ptr: cute.Pointer,
        dM_scan_ptr: cute.Pointer,
        dKprev_scan_ptr: cute.Pointer,
        dKcurr_scan_ptr: cute.Pointer,
        dB_inc_ptr: cute.Pointer,
        dB_prev_inc_ptr: cute.Pointer,
        dU_inc_ptr: cute.Pointer,
        dU_prev_inc_ptr: cute.Pointer,
        dMsum_part_ptr: cute.Pointer,
        dMp0_ptr: cute.Pointer,
        dM_inc_ptr: cute.Pointer,
        dKprev_inc_ptr: cute.Pointer,
        dKcurr_inc_ptr: cute.Pointer,
    ):
        mU_scan = _make_tensor(U_ptr, u_scan_shape, u_scan_stride)
        mB_scan = _make_tensor(B_ptr, b_scan_shape, b_scan_stride)
        mC_scan = _make_tensor(C_ptr, b_scan_shape, b_scan_stride)
        mM_scan = _make_tensor(M_ptr, m_scan_shape, m_scan_stride)
        mK_scan = _make_tensor(K_ptr, k_scan_shape, k_scan_stride)
        mDOut_scan = _make_tensor(dOut_ptr, u_scan_shape, u_scan_stride)
        mZ0_scan = _make_tensor(chunk_starts_ptr, z0_shape, z0_stride)
        mU_prev0_scan = _make_tensor(U_prev0_ptr, u_prev0_shape, u_prev0_stride)
        mB_prev0_scan = _make_tensor(B_prev0_ptr, b_prev0_shape, b_prev0_stride)
        mDU_scan = _make_tensor(dU_scan_ptr, u_scan_shape, u_scan_stride)
        mDB_scan = _make_tensor(dB_scan_ptr, b_scan_shape, b_scan_stride)
        mDU_prev_scan = _make_tensor(
            dU_prev_scan_ptr, d_dummy_u_prev_shape, d_dummy_u_prev_stride
        )
        mDB_prev_scan = _make_tensor(
            dB_prev_scan_ptr, d_dummy_b_prev_shape, d_dummy_b_prev_stride
        )
        mDLogp = _make_tensor(dlogp_ptr, dlogp_shape, dlogp_stride)
        mDLogp_param = _make_tensor(dlogp_ptr, dlp_param_shape, dlp_param_stride)
        mDMprev_scan = _make_tensor(
            dMp_scratch_ptr, dm_scratch_shape, dm_scratch_stride
        )
        mDMcurr_scan = _make_tensor(
            dMc_scratch_ptr, dm_scratch_shape, dm_scratch_stride
        )
        mDMprev_param = _make_tensor(
            dMp_scratch_ptr, dparam_scan_shape, dparam_scan_stride
        )
        mDMcurr_param = _make_tensor(
            dMc_scratch_ptr, dparam_scan_shape, dparam_scan_stride
        )
        mDC_scan = _make_tensor(dC_scan_ptr, b_scan_shape, b_scan_stride)
        mDR_scan = _make_tensor(dR_ptr, dr_shape, dr_stride)
        mDR_param = _make_tensor(dR_ptr, dr_param_shape, dr_param_stride)
        mDM_scan = _make_tensor(dM_scan_ptr, dparam_scan_shape, dparam_scan_stride)
        mDKprev_scan = _make_tensor(
            dKprev_scan_ptr, dparam_scan_shape, dparam_scan_stride
        )
        mDKcurr_scan = _make_tensor(
            dKcurr_scan_ptr, dparam_scan_shape, dparam_scan_stride
        )

        mDOut_dz0 = _make_tensor(dOut_ptr, dz0_dout_shape, dz0_dout_stride)
        mC_dz0 = _make_tensor(C_ptr, dz0_c_shape, dz0_c_stride)
        mM_dz0 = _make_tensor(M_ptr, dz0_m_shape, dz0_m_stride)
        mDZ0 = _make_tensor(dZ0_ptr, dz0_out_shape, dz0_out_stride)

        mChunkStarts_state = _make_tensor(chunk_starts_ptr, state_shape, state_stride)
        mMchunk_state = _make_tensor(m_chunk_ptr, m_chunk_shape, m_chunk_stride)
        mDChunkStarts_state = _make_tensor(dZ0_ptr, state_shape, state_stride)
        mDFinal = _make_tensor(d_final_ptr, final_shape, final_stride)
        mDInc_state = _make_tensor(d_inc_ptr, state_shape, state_stride)
        mDInc_flat = _make_tensor(d_inc_ptr, flat_state_shape, flat_state_stride)
        mDInc_tc_flat = _make_tensor(d_inc_tc_ptr, flat_state_shape, flat_state_stride)
        mDInitial = _make_tensor(d_initial_ptr, final_shape, final_stride)
        mDMchunk_state = _make_tensor(d_m_chunk_ptr, m_chunk_shape, m_chunk_stride)

        mU_inc = _make_tensor(U_ptr, u_inc_shape, u_inc_stride)
        mB_inc = _make_tensor(B_ptr, b_inc_shape, b_inc_stride)
        mM_inc = _make_tensor(M_ptr, m_inc_shape, m_inc_stride)
        mKprev_inc = _make_tensor(Kprev_ptr, k_inc_shape, k_inc_stride)
        mKcurr_inc = _make_tensor(Kcurr_ptr, k_inc_shape, k_inc_stride)
        mDInc_DP = _make_tensor(d_inc_tc_ptr, d_inc_dp_shape, d_inc_dp_stride)
        mDInc = _make_tensor(d_inc_tc_ptr, d_inc_shape, d_inc_stride)
        mDInc_boundary = _make_tensor(
            d_inc_tc_ptr, d_inc_boundary_shape, d_inc_boundary_stride
        )
        mBPrev_chunks = _make_tensor(
            B_prev_chunks_ptr, prev_b_chunks_shape, prev_b_chunks_stride
        )
        mUPrev_chunks = _make_tensor(
            U_prev_chunks_ptr, prev_u_chunks_shape, prev_u_chunks_stride
        )
        mDB_inc = _make_tensor(dB_inc_ptr, b_inc_shape, b_inc_stride)
        mDU_inc = _make_tensor(dU_inc_ptr, du_out_shape, du_out_stride)
        mDBPrev_inc = _make_tensor(
            dB_prev_inc_ptr, prev_b_chunks_shape, prev_b_chunks_stride
        )
        mDUPrev_inc = _make_tensor(
            dU_prev_inc_ptr, prev_u_chunks_shape, prev_u_chunks_stride
        )
        mDMsum_part = _make_tensor(dMsum_part_ptr, dmsum_part_shape, dmsum_part_stride)
        mDMp0 = _make_tensor(dMp0_ptr, dmp0_shape, dmp0_stride)
        mDMchunk_inc = _make_tensor(d_m_chunk_ptr, dmchunk_shape, dmchunk_stride)
        mDM_inc = _make_tensor(dM_inc_ptr, dparam_inc_shape, dparam_inc_stride)
        mDKprev_inc = _make_tensor(dKprev_inc_ptr, dparam_inc_shape, dparam_inc_stride)
        mDKcurr_inc = _make_tensor(dKcurr_inc_ptr, dparam_inc_shape, dparam_inc_stride)

        mDU_db_dummy = _make_tensor(dU_db_dummy_ptr, d_dummy_u_shape, d_dummy_u_stride)
        mDU_prev_db_dummy = _make_tensor(
            dU_prev_db_dummy_ptr, d_dummy_u_prev_shape, d_dummy_u_prev_stride
        )
        mDB_du_dummy = _make_tensor(dB_du_dummy_ptr, d_dummy_b_shape, d_dummy_b_stride)
        mDB_prev_du_dummy = _make_tensor(
            dB_prev_du_dummy_ptr, d_dummy_b_prev_shape, d_dummy_b_prev_stride
        )

        tc_dtype = U_ptr.value_type

        scan_db = ChunkScanBwdDBAmpere(
            tc_dtype,
            chunk_size=L,
            D=D,
            P=P,
            num_threads=scan_num_threads_db,
        )
        scan_dcdr = ChunkScanBwdDCDRAmpere(
            tc_dtype,
            chunk_size=L,
            D=D,
            P=P,
            num_threads=scan_num_threads_dc,
        )
        scan_param = ChunkScanBwdParamScanAmpere(
            chunk_size=L, num_threads=scan_num_threads_param
        )
        scan_du = ChunkScanBwdDUAmpere(
            tc_dtype,
            chunk_size=L,
            D=D,
            P=P,
            num_threads=scan_num_threads_du,
        )
        scan_dz0 = ChunkScanBwdDZ0Ampere(tc_dtype, chunk_size=L)
        cast_d_inc, cast_grid_x, cast_num_threads = _make_cast_f32_to_tc(
            total_elems=BHC * P * D
        )

        state_cfg = _TileConfig(
            num_threads=state_num_threads,
            pairs_per_thread=state_pairs_per_thread,
        )
        state_bwd = StatePassingBwdStateAmpere(
            state_cfg,
            copy_bits_in=state_copy_bits_state,
            copy_bits_out=state_copy_bits_out,
        )
        state_m_bwd = StatePassingBwdMAmpere(
            state_cfg, copy_bits_in=state_copy_bits_state
        )

        inc_db = ChunkIncrementBwdDBAmpere(tc_dtype, chunk_size=L, D=D, P=P)
        inc_boundary = ChunkIncrementBwdBoundaryAmpere(tc_dtype, chunk_size=L, D=D, P=P)
        inc_param = ChunkIncrementBwdParamScanAmpere(chunk_size=L, nDtiles=n_d_tiles)
        inc_du = ChunkIncrementBwdDUAmpere(tc_dtype, chunk_size=L, D=D, P=P)

        scan_db(
            mU_scan,
            mB_scan,
            mC_scan,
            mM_scan,
            mK_scan,
            mDOut_scan,
            mU_prev0_scan,
            mB_prev0_scan,
            mDU_db_dummy,
            mDB_scan,
            mDU_prev_db_dummy,
            mDB_prev_scan,
            mDLogp,
            mDMprev_scan,
            mDMcurr_scan,
        )
        scan_dcdr(
            mU_scan,
            mB_scan,
            mC_scan,
            mM_scan,
            mK_scan,
            mDOut_scan,
            mU_prev0_scan,
            mB_prev0_scan,
            mZ0_scan,
            mDLogp,
            mDC_scan,
            mDR_scan,
        )
        scan_param(
            mM_scan,
            mK_scan,
            mDLogp_param,
            mDMprev_param,
            mDMcurr_param,
            mDR_param,
            mDM_scan,
            mDKprev_scan,
            mDKcurr_scan,
        )
        scan_du(
            mU_scan,
            mB_scan,
            mC_scan,
            mM_scan,
            mK_scan,
            mDOut_scan,
            mU_prev0_scan,
            mB_prev0_scan,
            mDU_scan,
            mDB_du_dummy,
            mDU_prev_scan,
            mDB_prev_du_dummy,
            mDLogp,
            mDMprev_scan,
            mDMcurr_scan,
        )
        scan_dz0(mDOut_dz0, mC_dz0, mM_dz0, mDZ0)

        state_bwd(
            mDChunkStarts_state,
            mDFinal,
            mMchunk_state,
            mDInc_state,
            mDInitial,
        )
        state_m_bwd(mChunkStarts_state, mDInc_state, mDMchunk_state)
        cast_d_inc(mDInc_flat, mDInc_tc_flat).launch(
            grid=[cast_grid_x, 1, 1],
            block=[cast_num_threads, 1, 1],
        )

        inc_db(
            mU_inc,
            mB_inc,
            mM_inc,
            mKprev_inc,
            mKcurr_inc,
            mDInc_DP,
            mDB_inc,
            mDMsum_part,
        )
        inc_boundary(
            mDInc_boundary,
            mBPrev_chunks,
            mUPrev_chunks,
            mM_inc,
            mKprev_inc,
            mDUPrev_inc,
            mDBPrev_inc,
            mDMp0,
        )
        inc_param(
            mM_inc,
            mKprev_inc,
            mKcurr_inc,
            mDMsum_part,
            mDMp0,
            mDMchunk_inc,
            mDM_inc,
            mDKprev_inc,
            mDKcurr_inc,
        )
        inc_du(
            mDInc,
            mB_inc,
            mM_inc,
            mKprev_inc,
            mKcurr_inc,
            mDU_inc,
        )

    return _v2x2ssd_bwd_host_wrapper


def compile_v2x2ssd_bwd_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    m_chunk: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    scan_num_threads_du: int = 128,
    scan_num_threads_db: int = 128,
    scan_num_threads_dc: int = 128,
    scan_num_threads_param: int = 32,
    state_num_threads: int = 128,
    state_pairs_per_thread: int = 8,
) -> object:
    dynamic_args, alignments, spec, cfg, _outputs = _build_backward_args(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        scan_num_threads_du=scan_num_threads_du,
        scan_num_threads_db=scan_num_threads_db,
        scan_num_threads_dc=scan_num_threads_dc,
        scan_num_threads_param=scan_num_threads_param,
        state_num_threads=state_num_threads,
        state_pairs_per_thread=state_pairs_per_thread,
    )
    cache_key = _bwd_host_cache_key(
        device_index=(U.device.index if U.device.index is not None else -1),
        tc_dtype=_tc_input_dtype(U.dtype, compute_dtype),
        U_shape=tuple(U.shape),
        M_shape=tuple(M.shape),
        K_shape=tuple(K.shape),
        B_shape=tuple(B.shape),
        C_shape=tuple(C.shape),
        m_chunk_shape=tuple(m_chunk.shape),
        chunk_starts_shape=tuple(chunk_starts.shape),
        d_out_shape=tuple(d_out.shape),
        chunk_size=int(chunk_size),
        scan_num_threads_du=int(scan_num_threads_du),
        scan_num_threads_db=int(scan_num_threads_db),
        scan_num_threads_dc=int(scan_num_threads_dc),
        scan_num_threads_param=int(scan_num_threads_param),
        state_num_threads=int(state_num_threads),
        state_pairs_per_thread=int(state_pairs_per_thread),
        state_copy_bits_state=int(cfg[6]),
        state_copy_bits_out=int(cfg[7]),
        n_d_tiles=int(spec[7]),
        alignments=alignments,
    )
    cached = _BWD_HOST_CACHE.get(cache_key)
    if cached is not None:
        note_cache_event("cache.v2x2ssd.backward.host_jit", hit=True)
        return cached

    note_cache_event("cache.v2x2ssd.backward.host_jit", hit=False)
    host_wrapper = _make_v2x2ssd_bwd_host_wrapper(spec=spec, cfg=cfg)
    compiled = cute.compile(host_wrapper, *dynamic_args)
    _BWD_HOST_CACHE[cache_key] = compiled
    return compiled


def v2x2ssd_bwd_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    m_chunk: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    scan_num_threads_du: int = 128,
    scan_num_threads_db: int = 128,
    scan_num_threads_dc: int = 128,
    scan_num_threads_param: int = 32,
    state_num_threads: int = 128,
    state_pairs_per_thread: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with record_region("backward.v2x2ssd.total"):
        dynamic_args, alignments, spec, cfg, outputs = _build_backward_args(
            U,
            M,
            K,
            B,
            C,
            m_chunk,
            chunk_starts,
            d_out,
            chunk_size=chunk_size,
            compute_dtype=compute_dtype,
            scan_num_threads_du=scan_num_threads_du,
            scan_num_threads_db=scan_num_threads_db,
            scan_num_threads_dc=scan_num_threads_dc,
            scan_num_threads_param=scan_num_threads_param,
            state_num_threads=state_num_threads,
            state_pairs_per_thread=state_pairs_per_thread,
        )
        cache_key = _bwd_host_cache_key(
            device_index=(U.device.index if U.device.index is not None else -1),
            tc_dtype=_tc_input_dtype(U.dtype, compute_dtype),
            U_shape=tuple(U.shape),
            M_shape=tuple(M.shape),
            K_shape=tuple(K.shape),
            B_shape=tuple(B.shape),
            C_shape=tuple(C.shape),
            m_chunk_shape=tuple(m_chunk.shape),
            chunk_starts_shape=tuple(chunk_starts.shape),
            d_out_shape=tuple(d_out.shape),
            chunk_size=int(chunk_size),
            scan_num_threads_du=int(scan_num_threads_du),
            scan_num_threads_db=int(scan_num_threads_db),
            scan_num_threads_dc=int(scan_num_threads_dc),
            scan_num_threads_param=int(scan_num_threads_param),
            state_num_threads=int(state_num_threads),
            state_pairs_per_thread=int(state_pairs_per_thread),
            state_copy_bits_state=int(cfg[6]),
            state_copy_bits_out=int(cfg[7]),
            n_d_tiles=int(spec[7]),
            alignments=alignments,
        )
        compiled = _BWD_HOST_CACHE.get(cache_key)
        if compiled is None:
            note_cache_event("cache.v2x2ssd.backward.host_jit", hit=False)
            host_wrapper = _make_v2x2ssd_bwd_host_wrapper(spec=spec, cfg=cfg)
            compiled = cute.compile(host_wrapper, *dynamic_args)
            _BWD_HOST_CACHE[cache_key] = compiled
        else:
            note_cache_event("cache.v2x2ssd.backward.host_jit", hit=True)

        compiled(*dynamic_args)

        (
            dU_scan,
            dU_prev_scan,
            dB_scan,
            dB_prev_scan,
            dC_scan,
            dM_scan,
            dKprev_scan,
            dKcurr_scan,
            dU_inc,
            dU_prev_inc,
            dB_inc,
            dB_prev_inc,
            dM_inc,
            dKprev_inc,
            dKcurr_inc,
            *_keepalive,
        ) = outputs

        dU_scan.add_(dU_inc)
        dU_prev_scan.add_(dU_prev_inc)
        dB_scan.add_(dB_inc)
        dB_prev_scan.add_(dB_prev_inc)
        dM_scan[:, :, :, 0, :, :].add_(dM_inc)
        dKprev_scan[:, :, :, 0, :, :].add_(dKprev_inc)
        dKcurr_scan[:, :, :, 0, :, :].add_(dKcurr_inc)

        dU_public = _fold_chunk_boundary_carries(dU_scan, dU_prev_scan)
        dB_public = _fold_chunk_boundary_carries(dB_scan, dB_prev_scan)

        return (
            _public_from_chunked(dU_public, T=U.shape[2], dtype=U.dtype),
            _public_from_param_scan(dM_scan, T=U.shape[2]),
            _public_dk_from_parts(dKprev_scan, dKcurr_scan, T=U.shape[2]),
            _public_from_chunked(dB_public, T=U.shape[2], dtype=B.dtype),
            _public_from_chunked(dC_scan, T=U.shape[2], dtype=C.dtype),
        )


__all__ = [
    "compile_v2x2ssd_bwd_cute",
    "v2x2ssd_bwd_cute",
]
