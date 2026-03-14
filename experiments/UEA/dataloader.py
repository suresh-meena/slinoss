#!/usr/bin/env python3
"""Optimized UEA dataset utilities for SLinOSS experiments.

This module downloads UEA multivariate classification data, parses ``.ts``
files, and builds PyTorch dataloaders with padded batches and caching.
"""

from __future__ import annotations

import random
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

UEA_ARCHIVE_URLS = (
    "https://www.timeseriesclassification.com/aeon-toolkit/Multivariate2018_ts.zip",
    "http://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_arff.zip",
)
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class UEASplits:
    """Data tensors for train/val/test splits."""
    train_x: torch.Tensor
    train_lengths: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_lengths: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_lengths: torch.Tensor
    test_y: torch.Tensor
    num_features: int
    num_classes: int


class PaddedTimeSeriesDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Simple tensor dataset returning (x, length, y)."""
    def __init__(self, x: torch.Tensor, lengths: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.lengths = lengths
        self.y = y

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.lengths[idx], self.y[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    archive_path = raw_dir / "Multivariate2018.zip"

    if not (extracted_dir.exists() and any(extracted_dir.iterdir())):
        raw_dir.mkdir(parents=True, exist_ok=True)
        if not (archive_path.exists() and zipfile.is_zipfile(archive_path)):
            for url in UEA_ARCHIVE_URLS:
                try:
                    _download_with_progress(url, archive_path)
                    if zipfile.is_zipfile(archive_path): break
                except Exception: continue
        
        if not zipfile.is_zipfile(archive_path):
            raise RuntimeError("Failed to download UEA archive.")

        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extracted_dir)

    ts_root = extracted_dir / "Multivariate_ts"
    if ts_root.exists(): return ts_root
    arff_root = extracted_dir / "Multivariate_arff"
    if arff_root.exists(): return arff_root
    return extracted_dir


def _parse_ts_file(path: Path) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    samples: List[List[np.ndarray]] = []
    labels: List[str] = []
    in_data = False
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            if not in_data:
                if line.lower().startswith("@data"): in_data = True
                continue
            
            parts = line.split(":")
            labels.append(parts[-1].strip())
            dims = [np.asarray([float(v) if v not in {"", "?", "NaN", "nan"} else np.nan 
                               for v in chunk.strip().split(",")], dtype=np.float32) 
                    for chunk in parts[:-1]]
            
            # Ensure consistent lengths within a sample
            min_len = min(len(d) for d in dims)
            samples.append([d[:min_len] for d in dims])

    n_samples = len(samples)
    n_dims = len(samples[0])
    lengths = torch.tensor([len(s[0]) for s in samples], dtype=torch.long)
    max_len = int(lengths.max())

    data = np.zeros((n_samples, max_len, n_dims), dtype=np.float32)
    for i, dims in enumerate(samples):
        for d, series in enumerate(dims):
            data[i, :len(series), d] = np.nan_to_num(series, nan=0.0)

    return torch.from_numpy(data), lengths, labels


def _encode_labels(train_labels: List[str], test_labels: List[str]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    label_names = sorted(set(train_labels) | set(test_labels))
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    return (torch.tensor([label_to_idx[n] for n in train_labels], dtype=torch.long),
            torch.tensor([label_to_idx[n] for n in test_labels], dtype=torch.long),
            len(label_names))


def load_uea_splits(
    dataset_name: str,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    val_fraction: float = 0.15,
    seed: int = 7,
    normalize: bool = True,
    include_time: bool = False,
) -> UEASplits:
    cache_path = data_root / "cache" / f"{dataset_name}_v{val_fraction}_s{seed}_t{include_time}.pt"
    if cache_path.exists():
        try:
            return torch.load(cache_path, weights_only=False)
        except Exception: pass

    ts_root = _ensure_raw_data(data_root)
    dataset_dir = ts_root / dataset_name
    
    train_x, train_lengths, train_labels = _parse_ts_file(dataset_dir / f"{dataset_name}_TRAIN.ts")
    test_x, test_lengths, test_labels = _parse_ts_file(dataset_dir / f"{dataset_name}_TEST.ts")
    train_y_full, test_y, num_classes = _encode_labels(train_labels, test_labels)

    # Split train/val
    n = len(train_y_full)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    val_x, val_lengths, val_y = train_x[val_idx], train_lengths[val_idx], train_y_full[val_idx]
    train_x, train_lengths, train_y = train_x[train_idx], train_lengths[train_idx], train_y_full[train_idx]

    if normalize:
        mean, std = train_x.mean(dim=(0, 1), keepdim=True), train_x.std(dim=(0, 1), keepdim=True).clamp_min(1e-5)
        train_x = (train_x - mean) / std
        val_x = (val_x - mean) / std
        test_x = (test_x - mean) / std

    if include_time:
        def _add_time(x):
            t = torch.linspace(0, 1, x.shape[1], dtype=x.dtype).view(1, -1, 1).expand(x.shape[0], -1, 1)
            return torch.cat([t, x], dim=-1)
        train_x, val_x, test_x = _add_time(train_x), _add_time(val_x), _add_time(test_x)

    splits = UEASplits(train_x, train_lengths, train_y, val_x, val_lengths, val_y, 
                       test_x, test_lengths, test_y, int(train_x.shape[-1]), num_classes)
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(splits, cache_path)
    return splits


def create_dataloaders(config: dict) -> Tuple[Dict[str, DataLoader], UEASplits]:
    splits = load_uea_splits(
        config["dataset"],
        data_root=Path(config["data_root"]),
        val_fraction=config["val_fraction"],
        seed=config["seed"],
        normalize=config.get("normalize", True),
        include_time=config.get("include_time", False),
    )

    common = dict(batch_size=config["batch_size"], num_workers=config["num_workers"], pin_memory=True)
    return {
        "train": DataLoader(PaddedTimeSeriesDataset(splits.train_x, splits.train_lengths, splits.train_y), shuffle=True, **common),
        "val": DataLoader(PaddedTimeSeriesDataset(splits.val_x, splits.val_lengths, splits.val_y), shuffle=False, **common),
        "test": DataLoader(PaddedTimeSeriesDataset(splits.test_x, splits.test_lengths, splits.test_y), shuffle=False, **common),
    }, splits
