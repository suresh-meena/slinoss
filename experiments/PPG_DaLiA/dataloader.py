#!/usr/bin/env python3
"""Optimized PPG-DaLiA data preparation and dataloaders for SLinOSS experiments."""

from __future__ import annotations

import hashlib
import pickle
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PPG_DALIA_URL = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class PPGSplits:
    """Tensor splits for training and evaluation."""
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    num_features: int


class WindowDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Simple fixed-window dataset for regression."""
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _download_with_progress(url: str, dst: Path) -> None:
    progress = tqdm(total=0, unit="B", unit_scale=True, unit_divisor=1024, desc=f"download {dst.name}")
    def _hook(block_count: int, block_size: int, total_size: int) -> None:
        if total_size > 0 and progress.total != total_size:
            progress.total = total_size
        progress.update(block_count * block_size - progress.n)
    try:
        urllib.request.urlretrieve(url, dst, reporthook=_hook)
    finally:
        progress.close()


def _ensure_raw_data(data_root: Path) -> Path:
    raw_dir = data_root / "raw"
    extracted_dir = data_root / "extracted"
    ppg_root = extracted_dir / "PPG_FieldStudy"
    if (ppg_root / "S1" / "S1.pkl").exists():
        return ppg_root

    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    outer_zip = raw_dir / "ppg_dalia.zip"
    if not (outer_zip.exists() and zipfile.is_zipfile(outer_zip)):
        _download_with_progress(PPG_DALIA_URL, outer_zip)

    outer_unpack = extracted_dir / "outer"
    outer_unpack.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outer_zip, "r") as zf:
        zf.extractall(outer_unpack)

    inner_zip = outer_unpack / "data.zip"
    if not inner_zip.exists() or not zipfile.is_zipfile(inner_zip):
        raise RuntimeError(
            f"Expected inner archive at {inner_zip}, but it was not found after extraction."
        )
    with zipfile.ZipFile(inner_zip, "r") as zf:
        zf.extractall(extracted_dir)
    
    return ppg_root


def _minmax_scale_signed(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    if abs(hi - lo) < 1e-8: return np.zeros_like(x, dtype=np.float32)
    return (2.0 * (x - lo) / (hi - lo) - 1.0).astype(np.float32)


def _as_feature_matrix(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {arr.shape}.")
    return arr


def _subject_features_labels(subject_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    with subject_file.open("rb") as f:
        data = pickle.load(f, encoding="latin1")
    wrist = data["signal"]["wrist"]
    acc = np.repeat(_as_feature_matrix(wrist["ACC"], name="ACC"), 2, axis=0)
    bvp = _as_feature_matrix(wrist["BVP"], name="BVP")
    eda = np.repeat(_as_feature_matrix(wrist["EDA"], name="EDA"), 16, axis=0)
    temp = np.repeat(_as_feature_matrix(wrist["TEMP"], name="TEMP"), 16, axis=0)
    n = min(acc.shape[0], bvp.shape[0], eda.shape[0], temp.shape[0])
    x = np.concatenate(
        [
            _minmax_scale_signed(acc[:n]),
            _minmax_scale_signed(bvp[:n]),
            _minmax_scale_signed(eda[:n]),
            _minmax_scale_signed(temp[:n]),
        ],
        axis=1,
    )
    y = np.asarray(data["label"], dtype=np.float32).reshape(-1)
    y = np.concatenate([[y[0]] * 3, y], axis=0).astype(np.float32)
    return x, y


def _window_pair(x: np.ndarray, y: np.ndarray, x_window: int, x_hop: int, target_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if x_window > x.shape[0]:
        raise ValueError(
            f"x_window={x_window} exceeds available sequence length {x.shape[0]}."
        )
    ratio = x.shape[0] / y.shape[0]
    y_window, y_hop = max(1, int(round(x_window / ratio))), max(1, int(round(x_hop / ratio)))
    if y_window > y.shape[0]:
        raise ValueError(
            f"Derived y_window={y_window} exceeds available label length {y.shape[0]}."
        )
    xw = sliding_window_view(x, window_shape=x_window, axis=0)[::x_hop]
    xw = np.swapaxes(xw, 1, 2)
    yw = sliding_window_view(y, window_shape=y_window, axis=0)[::y_hop]
    n = min(xw.shape[0], yw.shape[0])
    if n == 0:
        raise ValueError(
            "Windowing produced zero aligned samples. "
            f"Got x_windows={xw.shape[0]} and y_windows={yw.shape[0]}."
        )
    xw, yw = xw[:n], yw[:n]
    if target_mode == "mean":
        target = yw.mean(axis=1, keepdims=True)
    elif target_mode == "center":
        target = yw[:, y_window // 2 : y_window // 2 + 1]
    elif target_mode == "sequence":
        target = yw
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    return xw.astype(np.float32), target.astype(np.float32)


def _concat_windows(label: str, chunks: list[np.ndarray]) -> torch.Tensor:
    if not chunks:
        raise ValueError(f"No windows were generated for {label}; check x_window/x_hop.")
    return torch.from_numpy(np.concatenate(chunks, axis=0))


def load_ppg_splits(config: dict) -> PPGSplits:
    data_root = Path(config["data_root"])
    cache_key = f"s{config['seed']}_t{config.get('include_time', False)}_w{config['x_window']}_h{config['x_hop']}_m{config['target_mode']}"
    digest = hashlib.sha1(cache_key.encode()).hexdigest()[:12]
    cache_file = data_root / "processed" / f"splits_{digest}.pt"
    
    if cache_file.exists():
        try:
            return torch.load(cache_file, weights_only=False)
        except Exception: pass

    ppg_root = _ensure_raw_data(data_root)
    subject_files = sorted(ppg_root.glob("S*/S*.pkl"), key=lambda p: int(p.stem[1:]))
    rng = np.random.default_rng(config["seed"])
    
    splits_data = {"tr_x": [], "tr_y": [], "va_x": [], "va_y": [], "te_x": [], "te_y": []}
    
    for sf in tqdm(subject_files, desc="Processing PPG subjects"):
        x, y = _subject_features_labels(sf)
        xw, yw = _window_pair(x, y, config["x_window"], config["x_hop"], config["target_mode"])
        if config.get("include_time", False):
            t = np.linspace(0, 1, xw.shape[1], dtype=np.float32).reshape(1, -1, 1)
            xw = np.concatenate([np.broadcast_to(t, (xw.shape[0], xw.shape[1], 1)), xw], axis=-1)
        
        n = xw.shape[0]
        v = rng.integers(0, 6)
        b = [int(p * n) for p in [0.15, 0.30, 0.70, 0.85]]
        slices = {0: (slice(0, b[2]), slice(b[2], b[3]), slice(b[3], n)),
                  1: (slice(0, b[2]), slice(b[3], n), slice(b[2], b[3])),
                  2: (slice(b[0], b[3]), slice(0, b[0]), slice(b[3], n)),
                  3: (slice(b[0], b[3]), slice(b[3], n), slice(0, b[0])),
                  4: (slice(b[1], n), slice(0, b[0]), slice(b[0], b[1])),
                  5: (slice(b[1], n), slice(b[0], b[1]), slice(0, b[0]))}[v]
        
        for i, (x_key, y_key) in enumerate([("tr_x", "tr_y"), ("va_x", "va_y"), ("te_x", "te_y")]):
            x_chunk = xw[slices[i]]
            y_chunk = yw[slices[i]]
            if x_chunk.shape[0] == 0 or y_chunk.shape[0] == 0:
                continue
            splits_data[x_key].append(x_chunk)
            splits_data[y_key].append(y_chunk)

    train_x = _concat_windows("train_x", splits_data["tr_x"])
    train_y = _concat_windows("train_y", splits_data["tr_y"])
    val_x = _concat_windows("val_x", splits_data["va_x"])
    val_y = _concat_windows("val_y", splits_data["va_y"])
    test_x = _concat_windows("test_x", splits_data["te_x"])
    test_y = _concat_windows("test_y", splits_data["te_y"])

    splits = PPGSplits(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        num_features=train_x.shape[-1],
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(splits, cache_file)
    return splits


def create_dataloaders(config: dict) -> Tuple[Dict[str, DataLoader], PPGSplits]:
    splits = load_ppg_splits(config)
    num_workers = int(config["num_workers"])
    common = dict(
        batch_size=config["batch_size"],
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return {
        "train": DataLoader(WindowDataset(splits.train_x, splits.train_y), shuffle=True, **common),
        "val": DataLoader(WindowDataset(splits.val_x, splits.val_y), shuffle=False, **common),
        "test": DataLoader(WindowDataset(splits.test_x, splits.test_y), shuffle=False, **common),
    }, splits
