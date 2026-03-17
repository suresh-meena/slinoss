#!/usr/bin/env python3
"""Optimized UEA dataset utilities for SLinOSS experiments.

This module downloads UEA multivariate classification data, parses ``.ts``
files, and builds PyTorch dataloaders with padded batches and caching.
"""

from __future__ import annotations

import shutil
import random
import csv
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

UEA_ARCHIVE_URLS = (
    "https://www.timeseriesclassification.com/aeon-toolkit/Multivariate2018_ts.zip",
    "http://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_arff.zip",
    "https://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip",
    "http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip",
    "https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip",
    "http://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip",
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

    ts_root = extracted_dir / "Multivariate_ts"
    arff_root = extracted_dir / "Multivariate_arff"

    if ts_root.exists():
        return ts_root
    if arff_root.exists():
        return arff_root

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

    if ts_root.exists():
        return ts_root
    if arff_root.exists():
        return arff_root

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

            lengths = {len(dim) for dim in dims}
            if len(lengths) != 1:
                raise ValueError(
                    f"Inconsistent per-dimension lengths in {path.name}: {sorted(lengths)}. "
                    "This loader expects synchronized multivariate series."
                )
            samples.append(dims)

    n_samples = len(samples)
    n_dims = len(samples[0])
    lengths = torch.tensor([len(s[0]) for s in samples], dtype=torch.long)
    max_len = int(lengths.max())

    data = np.zeros((n_samples, max_len, n_dims), dtype=np.float32)
    for i, dims in enumerate(samples):
        for d, series in enumerate(dims):
            data[i, :len(series), d] = np.nan_to_num(series, nan=0.0)

    return torch.from_numpy(data), lengths, labels


def _parse_arff_file(path: Path) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    samples: List[List[np.ndarray]] = []
    labels: List[str] = []
    in_data = False
    
    with path.open("r", encoding="utf-8") as f:
        lines = []
        for line in f:
            if not in_data:
                if line.strip().lower().startswith("@data"):
                    in_data = True
                continue
            if line.strip() and not line.strip().startswith('%'):
                lines.append(line.strip())
                
    reader = csv.reader(lines, delimiter=',', quotechar='"', skipinitialspace=True)
    
    for row in reader:
        if not row:
            continue
        label = row[-1].strip().strip("'\"")
            
        dims = []
        is_flat_univariate = len(row) > 10 and all(
            " " not in c and "\n" not in c and "\\n" not in c for c in row[:-1]
        )
        if is_flat_univariate:
            dim_arr = np.asarray([
                float(v.strip().strip("'\"")) if v.strip().strip("'\"") not in {"", "?", "NaN", "nan"} else np.nan 
                for v in row[:-1]
            ], dtype=np.float32)
            dims.append(dim_arr)
        else:
            for dim_str in row[:-1]:
                dim_str = dim_str.strip().strip("'\"")
                dim_str = dim_str.replace('\\n', ' ').replace('\n', ' ').replace(',', ' ')
                vals = [v.strip().strip("'\"") for v in dim_str.split() if v.strip()]
                dim_arr = np.asarray([
                    float(v) if v not in {"", "?", "NaN", "nan"} else np.nan 
                    for v in vals
                ], dtype=np.float32)
                dims.append(dim_arr)
        samples.append(dims)
        labels.append(label)

    n_samples = len(samples)
    if not n_samples:
        return torch.empty(0, 0, 0, dtype=torch.float32), torch.empty(0, dtype=torch.long), []

    n_dims = len(samples[0])

    sample_lengths = []
    for s in samples:
        # For each sample, find the maximum length across all its dimensions.
        # This becomes the effective length of the multivariate series for this sample,
        # and shorter dimensions will be implicitly right-padded with zeros.
        sample_lengths.append(max(len(dim) for dim in s) if s else 0)

    lengths = torch.tensor(sample_lengths, dtype=torch.long)
    max_len = lengths.max().item() if sample_lengths else 0

    data = np.zeros((n_samples, max_len, n_dims), dtype=np.float32)
    for i, dims in enumerate(samples):
        for d, series in enumerate(dims):
            data[i, :len(series), d] = np.nan_to_num(series, nan=0.0)

    return torch.from_numpy(data), lengths, labels


def _length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return steps < lengths.unsqueeze(1)


def _normalize_valid_timesteps(
    train_x: torch.Tensor,
    train_lengths: torch.Tensor,
    *others: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    mask = _length_mask(train_lengths, train_x.shape[1]).unsqueeze(-1).to(train_x.dtype)
    count = mask.sum(dim=(0, 1), keepdim=True).clamp_min(1.0)
    mean = (train_x * mask).sum(dim=(0, 1), keepdim=True) / count
    centered = (train_x - mean) * mask
    std = torch.sqrt(centered.square().sum(dim=(0, 1), keepdim=True) / count).clamp_min(1e-5)

    normalized: list[torch.Tensor] = []
    for x, lengths in ((train_x, train_lengths),) + others:
        x_mask = _length_mask(lengths, x.shape[1]).unsqueeze(-1).to(x.dtype)
        normalized.append(((x - mean) / std) * x_mask)
    return tuple(normalized)


def _append_time_feature(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    time_feature = torch.linspace(0, 1, x.shape[1], dtype=x.dtype).view(1, -1, 1)
    time_feature = time_feature.expand(x.shape[0], -1, 1)
    x = torch.cat([time_feature, x], dim=-1)
    mask = _length_mask(lengths, x.shape[1]).unsqueeze(-1).to(x.dtype)
    return x * mask


def _encode_labels(train_labels: List[str], test_labels: List[str]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    label_names = sorted(set(train_labels) | set(test_labels))
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    return (torch.tensor([label_to_idx[n] for n in train_labels], dtype=torch.long),
            torch.tensor([label_to_idx[n] for n in test_labels], dtype=torch.long),
            len(label_names))


def _deduplicate_samples(
    x: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_x = x.cpu().numpy().reshape(x.shape[0], -1)
    lengths_np = lengths.cpu().numpy().reshape(-1, 1)
    y_np = y.cpu().numpy().reshape(-1, 1)
    identity = np.concatenate([flat_x, lengths_np, y_np], axis=1)
    _, keep_indices = np.unique(identity, axis=0, return_index=True)
    keep_indices = np.sort(keep_indices)
    keep = torch.as_tensor(keep_indices, dtype=torch.long)
    return x[keep], lengths[keep], y[keep]


def load_uea_splits(
    dataset_name: str,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    val_fraction: float = 0.15,
    seed: int = 7,
    normalize: bool = True,
    include_time: bool = False,
    split_strategy: str = "official",
) -> UEASplits:
    cache_path = data_root / "cache" / (
        f"{dataset_name}_v{val_fraction}_s{seed}_n{int(normalize)}_t{int(include_time)}_split_{split_strategy}.pt"
    )
    if cache_path.exists():
        try:
            return torch.load(cache_path, weights_only=False)
        except Exception: pass

    ts_root = _ensure_raw_data(data_root)
    dataset_dir = ts_root / dataset_name
    
    train_ts = dataset_dir / f"{dataset_name}_TRAIN.ts"
    test_ts = dataset_dir / f"{dataset_name}_TEST.ts"
    train_arff = dataset_dir / f"{dataset_name}_TRAIN.arff"
    test_arff = dataset_dir / f"{dataset_name}_TEST.arff"
    
    if train_ts.exists() and test_ts.exists():
        train_x, train_lengths, train_labels = _parse_ts_file(train_ts)
        test_x, test_lengths, test_labels = _parse_ts_file(test_ts)
    elif train_arff.exists() and test_arff.exists():
        train_x, train_lengths, train_labels = _parse_arff_file(train_arff)
        test_x, test_lengths, test_labels = _parse_arff_file(test_arff)
    else:
        raise FileNotFoundError(
            f"Expected TS or ARFF files for dataset {dataset_name} in {dataset_dir}, "
            "but one or both split files are missing."
        )

    if split_strategy == "official":
        train_y_full, test_y, num_classes = _encode_labels(train_labels, test_labels)

        # Split train/val from official train set only.
        n = len(train_y_full)
        indices = list(range(n))
        random.Random(seed).shuffle(indices)
        n_val = min(n - 1, max(1, int(n * val_fraction)))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        val_x, val_lengths, val_y = train_x[val_idx], train_lengths[val_idx], train_y_full[val_idx]
        train_x, train_lengths, train_y = train_x[train_idx], train_lengths[train_idx], train_y_full[train_idx]
    elif split_strategy == "linoss_701515":
        all_x = torch.cat([train_x, test_x], dim=0)
        all_lengths = torch.cat([train_lengths, test_lengths], dim=0)
        all_labels = train_labels + test_labels
        label_names = sorted(set(all_labels))
        label_to_idx = {name: idx for idx, name in enumerate(label_names)}
        all_y = torch.tensor([label_to_idx[n] for n in all_labels], dtype=torch.long)
        all_x, all_lengths, all_y = _deduplicate_samples(all_x, all_lengths, all_y)
        num_classes = len(label_names)

        n = len(all_y)
        indices = list(range(n))
        random.Random(seed).shuffle(indices)
        bound1 = int(n * 0.7)
        bound2 = int(n * 0.85)
        train_idx = indices[:bound1]
        val_idx = indices[bound1:bound2]
        test_idx = indices[bound2:]

        train_x, train_lengths, train_y = all_x[train_idx], all_lengths[train_idx], all_y[train_idx]
        val_x, val_lengths, val_y = all_x[val_idx], all_lengths[val_idx], all_y[val_idx]
        test_x, test_lengths, test_y = all_x[test_idx], all_lengths[test_idx], all_y[test_idx]
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")

    if normalize:
        train_x, val_x, test_x = _normalize_valid_timesteps(
            train_x,
            train_lengths,
            (val_x, val_lengths),
            (test_x, test_lengths),
        )

    if include_time:
        train_x = _append_time_feature(train_x, train_lengths)
        val_x = _append_time_feature(val_x, val_lengths)
        test_x = _append_time_feature(test_x, test_lengths)

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
        split_strategy=str(config.get("split_strategy", "official")),
    )

    num_workers = int(config["num_workers"])
    common = dict(
        batch_size=config["batch_size"],
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return {
        "train": DataLoader(PaddedTimeSeriesDataset(splits.train_x, splits.train_lengths, splits.train_y), shuffle=True, **common),
        "val": DataLoader(PaddedTimeSeriesDataset(splits.val_x, splits.val_lengths, splits.val_y), shuffle=False, **common),
        "test": DataLoader(PaddedTimeSeriesDataset(splits.test_x, splits.test_lengths, splits.test_y), shuffle=False, **common),
    }, splits
