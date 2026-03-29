#!/usr/bin/env python3
"""Utilities for PPG experiments."""

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
        "runs_root": "runs_root",
    },
    "data": {
        "root": "data_root",
        "include_time": "include_time",
        "x_window": "x_window",
        "x_hop": "x_hop",
        "target_mode": "target_mode",
    },
    "training": {
        "seed": "seed",
        "epochs": "epochs",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "lr": "lr",
        "weight_decay": "weight_decay",
        "grad_clip": "grad_clip",
    },
    "model": {
        "d_model": "d_model",
        "n_layers": "n_layers",
        "d_state": "d_state",
        "expand": "expand",
        "d_head": "d_head",
        "d_conv": "d_conv",
        "chunk_size": "chunk_size",
        "dropout": "dropout",
    },
    "backend": {
        "scan_backend": "scan_backend",
    },
}
_KNOWN_SECTION_KEYS = set(_SECTION_FIELDS) | {"sweep"}
_DEFAULT_RUNS_ROOT = "experiments/PPG_DaLiA/runs"
_DEFAULT_SCAN_BACKEND = "auto"
_VALID_SCAN_BACKENDS = {"auto", "cute", "reference"}


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        config = yaml.safe_load(f)
    _require(isinstance(config, dict), f"Config at {path} must decode to a mapping.")
    return config


def _has_nested_layout(config: Mapping[str, Any]) -> bool:
    return any(key in config for key in _SECTION_FIELDS)


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
        unknown_fields = sorted(set(section_values) - set(fields))
        _require(
            not unknown_fields,
            f"Unsupported keys in {section}: {', '.join(unknown_fields)}.",
        )


def _flatten_sections(config: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for section, fields in _SECTION_FIELDS.items():
        values = _ensure_mapping(config.get(section), label=section)
        for source_key, dest_key in fields.items():
            if source_key in values:
                flat[dest_key] = deepcopy(values[source_key])
    return flat


def _runtime_defaults() -> dict[str, Any]:
    return {
        "runs_root": _DEFAULT_RUNS_ROOT,
        "scan_backend": _DEFAULT_SCAN_BACKEND,
    }


def _normalize_target_mode(target_mode: str) -> str:
    if target_mode == "full":
        return "sequence"
    return target_mode


def _validate_runtime_config(config: Mapping[str, Any]) -> None:
    required = [
        "data_root",
        "include_time",
        "x_window",
        "x_hop",
        "target_mode",
        "seed",
        "epochs",
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

    _require(int(config["x_window"]) > 0, "x_window must be positive.")
    _require(int(config["x_hop"]) > 0, "x_hop must be positive.")
    _require(int(config["seed"]) >= 0, "seed must be non-negative.")
    _require(int(config["epochs"]) > 0, "epochs must be positive.")
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
        config["target_mode"] in {"mean", "center", "sequence"},
        "target_mode must be one of ['mean', 'center', 'sequence'].",
    )


def specialize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not _has_nested_layout(config):
        runtime = dict(config)
        runtime.setdefault("runs_root", _DEFAULT_RUNS_ROOT)
        runtime.setdefault("scan_backend", _DEFAULT_SCAN_BACKEND)
        runtime["target_mode"] = _normalize_target_mode(str(runtime["target_mode"]))
        _validate_runtime_config(runtime)
        return runtime

    _validate_top_level_config(config)
    runtime = _runtime_defaults()
    runtime.update(_flatten_sections(config))
    runtime["target_mode"] = _normalize_target_mode(str(runtime["target_mode"]))
    _validate_runtime_config(runtime)
    return runtime


def resolve_sweep_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not _has_nested_layout(config):
        sweep_cfg = deepcopy(_ensure_mapping(config.get("sweep"), label="sweep"))
        sweep_cfg.setdefault("metric", "val_mae")
        sweep_cfg.setdefault("goal", "min")
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

    _validate_top_level_config(config)
    sweep_cfg = deepcopy(_ensure_mapping(config.get("sweep"), label="sweep"))
    sweep_cfg.setdefault("metric", "val_mae")
    sweep_cfg.setdefault("goal", "min")
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


def setup_logging(log_dir: Path, run_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"
    logger = logging.getLogger("PPG")
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


def get_run_dir(base_dir: Path, run_name: Optional[str] = None) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = run_name if run_name else stamp
    run_dir = base_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
