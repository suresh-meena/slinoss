#!/usr/bin/env python3
"""Utilities for UEA experiments."""

from __future__ import annotations

import logging
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import yaml


_SECTION_FIELDS: dict[str, dict[str, str]] = {
    "experiment": {
        "dataset": "dataset",
        "runs_root": "runs_root",
    },
    "data": {
        "root": "data_root",
        "val_fraction": "val_fraction",
        "include_time": "include_time",
        "normalize": "normalize",
        "split_strategy": "split_strategy",
    },
    "training": {
        "seed": "seed",
        "seeds": "seeds",
        "epochs": "epochs",
        "num_steps": "num_steps",
        "print_steps": "print_steps",
        "early_stopping_evals": "early_stopping_evals",
        "mixed_precision": "mixed_precision",
        "torch_compile": "torch_compile",
        "torch_compile_mode": "torch_compile_mode",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "lr": "lr",
        "weight_decay": "weight_decay",
        "grad_clip": "grad_clip",
    },
    "model": {
        "d_model": "d_model",
        "n_layers": "n_layers",
        "norm_type": "norm_type",
        "ffn_activation": "ffn_activation",
        "ffn_mult": "ffn_mult",
        "d_state": "d_state",
        "expand": "expand",
        "d_head": "d_head",
        "d_conv": "d_conv",
        "chunk_size": "chunk_size",
        "dropout": "dropout",
        "dt_min": "dt_min",
        "dt_max": "dt_max",
        "dt_init_floor": "dt_init_floor",
        "r_min": "r_min",
        "r_max": "r_max",
        "theta_bound": "theta_bound",
        "k_max": "k_max",
        "eps": "eps",
        "normalize_bc": "normalize_bc",
    },
    "backend": {
        "scan_backend": "scan_backend",
    },
}
_KNOWN_SECTION_KEYS = set(_SECTION_FIELDS) | {"datasets", "sweep"}
_DEFAULT_RUNS_ROOT = "experiments/UEA/runs"
_DEFAULT_SCAN_BACKEND = "reference"
_VALID_SCAN_BACKENDS = {"auto", "cute", "reference"}


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with path.open("r") as f:
        config = yaml.safe_load(f)
    _require(isinstance(config, dict), f"Config at {path} must decode to a mapping.")
    return config


def _has_nested_layout(config: Mapping[str, Any]) -> bool:
    return any(key in config for key in _SECTION_FIELDS) or "datasets" in config


def _ensure_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    _require(isinstance(value, dict), f"{label} must be a mapping.")
    return dict(value)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _validate_top_level_config(config: Mapping[str, Any]) -> None:
    unknown = sorted(set(config) - _KNOWN_SECTION_KEYS)
    _require(
        not unknown,
        f"Unsupported top-level config keys: {', '.join(unknown)}.",
    )
    for section, fields in _SECTION_FIELDS.items():
        section_values = config.get(section)
        if section_values is None:
            continue
        _require(isinstance(section_values, dict), f"{section} must be a mapping.")

    datasets = _ensure_mapping(config.get("datasets"), label="datasets")
    for dataset_name, dataset_cfg in datasets.items():
        _require(
            isinstance(dataset_cfg, dict),
            f"datasets.{dataset_name} must be a mapping.",
        )
        unknown_sections = sorted(set(dataset_cfg) - (set(_SECTION_FIELDS) | {"sweep"}))
        _require(
            not unknown_sections,
            f"Unsupported sections in datasets.{dataset_name}: {', '.join(unknown_sections)}.",
        )
        for section, fields in _SECTION_FIELDS.items():
            section_values = dataset_cfg.get(section)
            if section_values is None:
                continue
            _require(
                isinstance(section_values, dict),
                f"datasets.{dataset_name}.{section} must be a mapping.",
            )


def _flatten_sections(config: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for section, fields in _SECTION_FIELDS.items():
        values = _ensure_mapping(config.get(section), label=section)
        # Add all fields as-is from the section
        for k, v in values.items():
            flat[k] = deepcopy(v)
        # Apply specific renames (e.g., 'root' -> 'data_root')
        for source_key, dest_key in fields.items():
            if source_key in values:
                flat[dest_key] = deepcopy(values[source_key])
    return flat


def _runtime_defaults() -> dict[str, Any]:
    return {
        "runs_root": _DEFAULT_RUNS_ROOT,
        "scan_backend": _DEFAULT_SCAN_BACKEND,
        "normalize": True,
        "split_strategy": "official",
        "print_steps": 1000,
        "early_stopping_evals": 10,
        "mixed_precision": "bf16" if torch.cuda.is_available() else "no",
        "torch_compile": False,
        "torch_compile_mode": "default",
    }


def _validate_runtime_config(config: Mapping[str, Any]) -> None:
    required = [
        "dataset",
        "data_root",
        "val_fraction",
        "seed",
        "batch_size",
        "num_workers",
        "lr",
        "weight_decay",
        "grad_clip",
        "d_model",
        "n_layers",
        "d_state",
        "expand",
        "d_head",
        "d_conv",
        "chunk_size",
        "dropout",
        "scan_backend",
        "runs_root",
    ]
    missing = [key for key in required if key not in config]
    _require(not missing, f"Missing required config keys: {', '.join(missing)}.")

    _require(0.0 < float(config["val_fraction"]) < 1.0, "val_fraction must be in (0, 1).")
    _require(int(config["seed"]) >= 0, "seed must be non-negative.")
    if "epochs" in config:
        _require(int(config["epochs"]) > 0, "epochs must be positive.")
    if "num_steps" in config:
        _require(int(config["num_steps"]) > 0, "num_steps must be positive.")
    if "print_steps" in config:
        _require(int(config["print_steps"]) > 0, "print_steps must be positive.")
    _require(
        "epochs" in config or "num_steps" in config,
        "Either epochs or num_steps must be provided.",
    )
    _require(int(config.get("early_stopping_evals", 10)) >= 0, "early_stopping_evals must be non-negative.")
    _require(int(config["batch_size"]) > 0, "batch_size must be positive.")
    _require(int(config["num_workers"]) >= 0, "num_workers must be non-negative.")
    _require(float(config["lr"]) > 0.0, "lr must be positive.")
    _require(float(config["weight_decay"]) >= 0.0, "weight_decay must be non-negative.")
    _require(float(config["grad_clip"]) >= 0.0, "grad_clip must be non-negative.")
    _require(int(config["d_model"]) > 0, "d_model must be positive.")
    _require(int(config["n_layers"]) > 0, "n_layers must be positive.")
    _require(int(config["d_state"]) > 0, "d_state must be positive.")
    _require(int(config["expand"]) > 0, "expand must be positive.")
    _require(int(config["d_head"]) > 0, "d_head must be positive.")
    _require(int(config["d_conv"]) > 0, "d_conv must be positive.")
    _require(int(config["chunk_size"]) > 0, "chunk_size must be positive.")
    _require(0.0 <= float(config["dropout"]) < 1.0, "dropout must be in [0, 1).")
    _require(
        int(config["expand"]) * int(config["d_model"]) % int(config["d_head"]) == 0,
        "expand * d_model must be divisible by d_head.",
    )
    _require(
        config["scan_backend"] in _VALID_SCAN_BACKENDS,
        f"scan_backend must be one of {sorted(_VALID_SCAN_BACKENDS)}.",
    )
    _require(
        str(config.get("split_strategy", "official")) in {"official", "linoss_701515"},
        "split_strategy must be one of ['official', 'linoss_701515'].",
    )
    seeds = config.get("seeds")
    if seeds is not None:
        _require(isinstance(seeds, list) and len(seeds) > 0, "seeds must be a non-empty list.")
        for s in seeds:
            _require(int(s) >= 0, "all seeds must be non-negative.")


def get_available_datasets(config: Mapping[str, Any]) -> list[str]:
    """Return dataset names declared in the config."""
    if _has_nested_layout(config):
        datasets = _ensure_mapping(config.get("datasets"), label="datasets")
        ordered = list(datasets)
        default_dataset = _ensure_mapping(config.get("experiment"), label="experiment").get("dataset")
        if default_dataset and default_dataset not in ordered:
            ordered.insert(0, str(default_dataset))
        return ordered

    datasets = list(_ensure_mapping(config.get("dataset_overrides"), label="dataset_overrides"))
    default_dataset = config.get("dataset")
    if default_dataset and default_dataset not in datasets:
        datasets.insert(0, str(default_dataset))
    return datasets


def specialize_config(config: Mapping[str, Any], dataset: Optional[str] = None) -> dict[str, Any]:
    """Apply dataset-specific overrides and return a flat runtime config."""
    if not _has_nested_layout(config):
        spec_config = dict(config)
        target_dataset = dataset or spec_config.get("dataset")
        if target_dataset:
            overrides = spec_config.pop("dataset_overrides", {})
            _require(isinstance(overrides, dict), "dataset_overrides must be a mapping.")
            if target_dataset in overrides:
                spec_config.update(deepcopy(overrides[target_dataset]))
            spec_config["dataset"] = target_dataset
        spec_config.setdefault("runs_root", _DEFAULT_RUNS_ROOT)
        spec_config.setdefault("scan_backend", _DEFAULT_SCAN_BACKEND)
        spec_config.setdefault("normalize", True)
        _validate_runtime_config(spec_config)
        return spec_config

    _validate_top_level_config(config)
    experiment = _ensure_mapping(config.get("experiment"), label="experiment")
    target_dataset = dataset or experiment.get("dataset")
    _require(target_dataset, "A dataset must be provided in config or via --dataset.")

    merged = {
        section: _ensure_mapping(config.get(section), label=section)
        for section in _SECTION_FIELDS
    }
    dataset_cfg = _ensure_mapping(
        _ensure_mapping(config.get("datasets"), label="datasets").get(str(target_dataset)),
        label=f"datasets.{target_dataset}",
    )
    for section in _SECTION_FIELDS:
        merged[section] = _deep_merge(merged[section], _ensure_mapping(dataset_cfg.get(section), label=section))

    flat_config = _runtime_defaults()
    flat_config.update(_flatten_sections(merged))
    flat_config["dataset"] = str(target_dataset)
    _validate_runtime_config(flat_config)
    return flat_config


def resolve_sweep_config(config: Mapping[str, Any], dataset: Optional[str] = None) -> dict[str, Any]:
    """Return the merged sweep configuration for a dataset."""
    if not _has_nested_layout(config):
        sweep_cfg = deepcopy(_ensure_mapping(config.get("sweep"), label="sweep"))
        sweep_cfg.setdefault("metric", "val_acc")
        sweep_cfg.setdefault("goal", "max")
        sweep_cfg.setdefault("grid", {})
        sweep_cfg.setdefault("fixed", {})
        _require(isinstance(sweep_cfg["fixed"], dict), "sweep.fixed must be a mapping.")
        _require(isinstance(sweep_cfg["grid"], dict), "sweep.grid must be a mapping.")
        for key, values in sweep_cfg["grid"].items():
            _require(
                isinstance(values, list) and values,
                f"sweep.grid.{key} must be a non-empty list.",
            )
        _require(
            sweep_cfg["goal"] in {"max", "min"},
            "sweep.goal must be either 'max' or 'min'.",
        )
        return sweep_cfg

    _validate_top_level_config(config)
    experiment = _ensure_mapping(config.get("experiment"), label="experiment")
    target_dataset = dataset or experiment.get("dataset")
    sweep_cfg = deepcopy(_ensure_mapping(config.get("sweep"), label="sweep"))

    if target_dataset:
        datasets = _ensure_mapping(config.get("datasets"), label="datasets")
        dataset_cfg = _ensure_mapping(datasets.get(str(target_dataset)), label=f"datasets.{target_dataset}")
        sweep_cfg = _deep_merge(sweep_cfg, _ensure_mapping(dataset_cfg.get("sweep"), label="sweep"))

    sweep_cfg.setdefault("metric", "val_acc")
    sweep_cfg.setdefault("goal", "max")
    sweep_cfg.setdefault("fixed", {})
    sweep_cfg.setdefault("grid", {})

    _require(isinstance(sweep_cfg["fixed"], dict), "sweep.fixed must be a mapping.")
    _require(isinstance(sweep_cfg["grid"], dict), "sweep.grid must be a mapping.")
    for key, values in sweep_cfg["grid"].items():
        _require(
            isinstance(values, list) and values,
            f"sweep.grid.{key} must be a non-empty list.",
        )
    _require(
        sweep_cfg["goal"] in {"max", "min"},
        "sweep.goal must be either 'max' or 'min'.",
    )
    return sweep_cfg


def configure_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Match the parameter-grouping convention used in the reference example."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "bias" not in name and "norm" not in name.lower():
            decay.append(param)
        else:
            no_decay.append(param)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    use_fused = any(param.is_cuda for param in decay) or any(param.is_cuda for param in no_decay)
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=use_fused)


def resolve_mixed_precision(value: Any) -> str:
    """Normalize config mixed precision values for Accelerate."""
    if isinstance(value, bool):
        return "bf16" if value and torch.cuda.is_available() else "no"

    if value is None:
        return "bf16" if torch.cuda.is_available() else "no"

    normalized = str(value).strip().lower()
    if normalized in {"false", "no", "off", "none"}:
        return "no"
    if normalized in {"true", "on"}:
        return "bf16" if torch.cuda.is_available() else "no"
    if normalized in {"no", "fp8", "fp16", "bf16"}:
        return normalized
    raise ValueError("mixed_precision must be one of ['no', 'fp8', 'fp16', 'bf16'].")


def setup_logging(log_dir: Path, run_name: str) -> logging.Logger:
    """Configure logging to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"

    logger = logging.getLogger("UEA")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def get_run_dir(base_dir: Path, dataset: str, run_name: Optional[str] = None) -> Path:
    """Generate a unique run directory."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = run_name if run_name else stamp
    run_dir = base_dir / dataset / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
