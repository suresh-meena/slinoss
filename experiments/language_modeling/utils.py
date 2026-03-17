#!/usr/bin/env python3
"""Utilities for FineWeb-Edu language-modeling experiments."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


class TrainState:
    """Small checkpointable container for training progress."""

    def __init__(self) -> None:
        self.global_step = 0
        self.seen_batches = 0
        self.tokens_seen = 0
        self.best_val_nll = float("inf")
        self.best_val_ppl = float("inf")
        self.last_train_loss = float("nan")
        self.elapsed_s = 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "global_step": int(self.global_step),
            "seen_batches": int(self.seen_batches),
            "tokens_seen": int(self.tokens_seen),
            "best_val_nll": float(self.best_val_nll),
            "best_val_ppl": float(self.best_val_ppl),
            "last_train_loss": float(self.last_train_loss),
            "elapsed_s": float(self.elapsed_s),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.global_step = int(state_dict.get("global_step", 0))
        self.seen_batches = int(state_dict.get("seen_batches", 0))
        self.tokens_seen = int(state_dict.get("tokens_seen", 0))
        self.best_val_nll = float(state_dict.get("best_val_nll", float("inf")))
        self.best_val_ppl = float(state_dict.get("best_val_ppl", float("inf")))
        self.last_train_loss = float(state_dict.get("last_train_loss", float("nan")))
        self.elapsed_s = float(state_dict.get("elapsed_s", 0.0))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _require(isinstance(config, dict), f"Config at {path} must decode to a mapping.")
    return dict(config)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_config(config: Mapping[str, Any], *, preset: str | None = None) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    requested = preset
    if requested is None:
        requested = str(resolved.get("experiment", {}).get("preset", "")).strip() or None
    presets = resolved.get("presets", {})
    if requested is not None:
        _require(
            isinstance(presets, Mapping) and requested in presets,
            f"Unknown preset '{requested}'. Available presets: {sorted(presets)}.",
        )
        resolved = deep_merge(resolved, presets[requested])
        resolved.setdefault("experiment", {})
        resolved["experiment"]["preset"] = requested
    return resolved


def set_nested_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    _require(parts, f"Invalid override key: {dotted_key!r}.")
    for part in parts[:-1]:
        current = cursor.get(part)
        if current is None:
            current = {}
            cursor[part] = current
        _require(
            isinstance(current, dict),
            f"Override path '{dotted_key}' crosses non-mapping key '{part}'.",
        )
        cursor = current
    cursor[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    merged = deepcopy(config)
    for override in overrides:
        key, sep, raw_value = override.partition("=")
        _require(
            sep == "=" and key,
            f"Overrides must look like section.key=value. Got {override!r}.",
        )
        set_nested_key(merged, key.strip(), yaml.safe_load(raw_value))
    return merged


def validate_config(config: Mapping[str, Any]) -> None:
    required_sections = ("experiment", "data", "training", "model", "backend")
    for section in required_sections:
        _require(section in config, f"Missing required config section: {section}.")
        _require(isinstance(config[section], Mapping), f"{section} must be a mapping.")

    experiment = dict(config["experiment"])
    data = dict(config["data"])
    training = dict(config["training"])
    model = dict(config["model"])
    backend = dict(config["backend"])

    _require(str(experiment.get("name", "")).strip(), "experiment.name is required.")
    _require(str(experiment.get("output_root", "")).strip(), "experiment.output_root is required.")
    _require(int(experiment.get("seed", -1)) >= 0, "experiment.seed must be non-negative.")

    _require(str(data.get("dataset_name", "")).strip(), "data.dataset_name is required.")
    _require(str(data.get("dataset_split", "")).strip(), "data.dataset_split is required.")
    _require(str(data.get("text_field", "")).strip(), "data.text_field is required.")
    _require(str(data.get("tokenizer_name", "")).strip(), "data.tokenizer_name is required.")
    _require(int(data.get("shuffle_buffer_size", 0)) >= 0, "data.shuffle_buffer_size must be non-negative.")
    _require(int(data.get("validation_docs", 0)) > 0, "data.validation_docs must be positive.")
    _require(int(data.get("tokenizer_batch_size", 0)) > 0, "data.tokenizer_batch_size must be positive.")

    _require(int(training.get("seq_len", 0)) > 0, "training.seq_len must be positive.")
    _require(int(training.get("micro_batch_size", 0)) > 0, "training.micro_batch_size must be positive.")
    _require(int(training.get("eval_batch_size", 0)) > 0, "training.eval_batch_size must be positive.")
    _require(
        int(training.get("gradient_accumulation_steps", 0)) > 0,
        "training.gradient_accumulation_steps must be positive.",
    )
    train_tokens = training.get("train_tokens")
    max_steps = training.get("max_steps")
    _require(
        train_tokens is not None or max_steps is not None,
        "Provide training.train_tokens or training.max_steps.",
    )
    if train_tokens is not None:
        _require(int(train_tokens) > 0, "training.train_tokens must be positive when set.")
    if max_steps is not None:
        _require(int(max_steps) > 0, "training.max_steps must be positive when set.")
    _require(int(training.get("warmup_steps", 0)) >= 0, "training.warmup_steps must be non-negative.")
    _require(float(training.get("learning_rate", 0.0)) > 0.0, "training.learning_rate must be positive.")
    _require(
        0.0 <= float(training.get("min_lr_ratio", -1.0)) <= 1.0,
        "training.min_lr_ratio must be in [0, 1].",
    )
    _require(float(training.get("weight_decay", -1.0)) >= 0.0, "training.weight_decay must be non-negative.")
    _require(float(training.get("adam_beta1", -1.0)) < 1.0, "training.adam_beta1 must be < 1.")
    _require(float(training.get("adam_beta2", -1.0)) < 1.0, "training.adam_beta2 must be < 1.")
    _require(float(training.get("adam_eps", 0.0)) > 0.0, "training.adam_eps must be positive.")
    _require(float(training.get("grad_clip", -1.0)) >= 0.0, "training.grad_clip must be non-negative.")
    _require(int(training.get("log_interval", 0)) > 0, "training.log_interval must be positive.")
    _require(int(training.get("eval_interval", 0)) > 0, "training.eval_interval must be positive.")
    _require(int(training.get("save_interval", 0)) > 0, "training.save_interval must be positive.")
    _require(int(training.get("max_eval_batches", 0)) > 0, "training.max_eval_batches must be positive.")
    _require(int(training.get("num_workers", -1)) >= 0, "training.num_workers must be non-negative.")

    _require(str(model.get("type", "")).strip() in {"slinoss", "mamba2"}, "model.type must be 'slinoss' or 'mamba2'.")
    _require(int(model.get("target_params", 0)) > 0, "model.target_params must be positive.")
    _require(int(model.get("d_model", 0)) > 0, "model.d_model must be positive.")
    _require(int(model.get("n_layers", 0)) > 0, "model.n_layers must be positive.")
    _require(int(model.get("d_state", 0)) > 0, "model.d_state must be positive.")
    _require(int(model.get("expand", 0)) > 0, "model.expand must be positive.")
    _require(int(model.get("d_head", 0)) > 0, "model.d_head must be positive.")
    _require(int(model.get("d_conv", 0)) > 0, "model.d_conv must be positive.")
    _require(int(model.get("chunk_size", 0)) > 0, "model.chunk_size must be positive.")
    _require(int(model.get("ffn_hidden_dim", 0)) > 0, "model.ffn_hidden_dim must be positive.")
    _require(
        int(model["expand"]) * int(model["d_model"]) % int(model["d_head"]) == 0,
        "model.expand * model.d_model must be divisible by model.d_head.",
    )
    _require(
        int(model["chunk_size"]) <= int(training["seq_len"]),
        "model.chunk_size must be <= training.seq_len.",
    )
    _require(
        str(backend.get("scan_backend", "")).strip() in {"auto", "reference", "cute"},
        "backend.scan_backend must be one of ['auto', 'reference', 'cute'].",
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_run_name(config: Mapping[str, Any], run_name: str | None = None) -> str:
    if run_name is not None:
        return run_name
    experiment = config["experiment"]
    preset = str(experiment.get("preset", config["model"]["type"]))
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return f"{experiment['name']}-{preset}-{stamp}"


def setup_logging(run_dir: Path, *, main_process: bool) -> logging.Logger:
    logger = logging.getLogger(f"slinoss.language_modeling.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    if not main_process:
        logger.addHandler(logging.NullHandler())
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def flatten_config(config: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in config.items():
        current = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_config(value, prefix=current))
        else:
            flat[current] = value
    return flat


def count_parameters(model: torch.nn.Module) -> tuple[int, float]:
    total = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    return total, total / 1e6


def configure_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and "norm" not in name.lower() and "bias" not in name.lower():
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    use_fused = any(parameter.is_cuda for parameter in decay + no_decay)
    return torch.optim.AdamW(
        groups,
        lr=lr,
        betas=betas,
        eps=eps,
        fused=use_fused,
    )


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step + 1, 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def derive_training_steps(
    config: Mapping[str, Any],
    *,
    world_size: int,
) -> dict[str, int]:
    training = dict(config["training"])
    tokens_per_microbatch = (
        int(training["micro_batch_size"]) * int(training["seq_len"]) * int(world_size)
    )
    tokens_per_step = tokens_per_microbatch * int(training["gradient_accumulation_steps"])
    derived = {
        "tokens_per_microbatch": int(tokens_per_microbatch),
        "tokens_per_step": int(tokens_per_step),
    }
    max_steps = training.get("max_steps")
    train_tokens = training.get("train_tokens")
    if max_steps is None:
        _require(train_tokens is not None, "train_tokens is required when max_steps is unset.")
        derived["max_steps"] = int(math.ceil(int(train_tokens) / float(tokens_per_step)))
    else:
        derived["max_steps"] = int(max_steps)
    return derived
