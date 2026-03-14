"""Utilities for the synthetic throughput comparison experiment."""

from __future__ import annotations

import json
import platform
import socket
import statistics
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected {label} to be a mapping, got {type(value).__name__}.")
    return {str(key): item for key, item in value.items()}


def _as_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Expected {label} to be a list, got {type(value).__name__}.")
    return list(value)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    suite: str
    description: str
    batch_size: int
    seq_len: int
    input_dim: int
    output_dim: int
    hidden_dim: int
    layers: int
    state_dim: int
    expand: int
    d_conv: int
    d_head: int
    chunk_size: int
    sweep_key: str | None = None
    sweep_value: int | float | str | None = None

    @property
    def sequences_per_step(self) -> int:
        return int(self.batch_size)

    @property
    def timesteps_per_step(self) -> int:
        return int(self.batch_size * self.seq_len)

    @property
    def payload(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = _as_mapping(payload, label="config")
    experiment = _as_mapping(config.get("experiment"), label="experiment")
    defaults = _as_mapping(config.get("defaults"), label="defaults")
    models = _as_mapping(config.get("models"), label="models")
    raw_cases = _as_list(config.get("cases"), label="cases")

    _require(models, "At least one model must be configured.")
    cases = [_build_case(defaults=defaults, raw_case=case) for case in raw_cases]
    return {
        "experiment": deepcopy(experiment),
        "defaults": deepcopy(defaults),
        "models": deepcopy(models),
        "cases": cases,
    }


def _build_case(*, defaults: dict[str, Any], raw_case: Any) -> CaseSpec:
    case_mapping = _as_mapping(raw_case, label="case")
    merged = deepcopy(defaults)
    merged.update(case_mapping)

    batch_size = int(merged["batch_size"])
    seq_len = int(merged["seq_len"])
    input_dim = int(merged["input_dim"])
    output_dim = int(merged["output_dim"])
    hidden_dim = int(merged["hidden_dim"])
    layers = int(merged["layers"])
    state_dim = int(merged["state_dim"])
    expand = int(merged["expand"])
    d_conv = int(merged["d_conv"])
    d_head = int(merged["d_head"])
    chunk_size = int(merged["chunk_size"])

    _require(batch_size > 0, f"{merged['name']}: batch_size must be positive.")
    _require(seq_len > 0, f"{merged['name']}: seq_len must be positive.")
    _require(input_dim > 0, f"{merged['name']}: input_dim must be positive.")
    _require(output_dim > 0, f"{merged['name']}: output_dim must be positive.")
    _require(hidden_dim > 0, f"{merged['name']}: hidden_dim must be positive.")
    _require(layers > 0, f"{merged['name']}: layers must be positive.")
    _require(state_dim > 0, f"{merged['name']}: state_dim must be positive.")
    _require(expand > 0, f"{merged['name']}: expand must be positive.")
    _require(d_conv > 0, f"{merged['name']}: d_conv must be positive.")
    _require(d_head > 0, f"{merged['name']}: d_head must be positive.")
    _require(chunk_size > 0, f"{merged['name']}: chunk_size must be positive.")
    _require(
        chunk_size <= seq_len,
        f"{merged['name']}: chunk_size={chunk_size} must be <= seq_len={seq_len}.",
    )
    _require(
        (hidden_dim * expand) % d_head == 0,
        (
            f"{merged['name']}: hidden_dim * expand = {hidden_dim * expand} "
            f"must be divisible by d_head = {d_head}."
        ),
    )

    return CaseSpec(
        name=str(merged["name"]),
        suite=str(merged["suite"]),
        description=str(merged.get("description", "")),
        batch_size=batch_size,
        seq_len=seq_len,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        layers=layers,
        state_dim=state_dim,
        expand=expand,
        d_conv=d_conv,
        d_head=d_head,
        chunk_size=chunk_size,
        sweep_key=(
            None if merged.get("sweep_key") is None else str(merged.get("sweep_key"))
        ),
        sweep_value=merged.get("sweep_value"),
    )


def filter_cases(cases: list[CaseSpec], names: set[str] | None) -> list[CaseSpec]:
    if not names:
        return list(cases)
    return [case for case in cases if case.name in names]


def filter_models(
    models: dict[str, dict[str, Any]],
    names: set[str] | None,
) -> dict[str, dict[str, Any]]:
    if not names:
        return deepcopy(models)
    return {
        model_name: deepcopy(model_cfg)
        for model_name, model_cfg in models.items()
        if model_name in names
    }


def summary_stats(samples: list[float]) -> dict[str, float]:
    _require(samples, "Cannot summarize an empty sample list.")
    return {
        "num_samples": float(len(samples)),
        "mean": float(statistics.fmean(samples)),
        "median": float(statistics.median(samples)),
        "min": float(min(samples)),
        "max": float(max(samples)),
        "stdev": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
    }


def timing_payload(
    step_ms: list[float],
    *,
    sequences_per_step: int,
    timesteps_per_step: int,
    cold_ms: float,
) -> dict[str, Any]:
    sequences_per_s = [
        1000.0 * sequences_per_step / ms if ms > 0.0 else 0.0 for ms in step_ms
    ]
    timesteps_per_s = [
        1000.0 * timesteps_per_step / ms if ms > 0.0 else 0.0 for ms in step_ms
    ]
    return {
        "cold_ms": float(cold_ms),
        "warm_step_ms": summary_stats(step_ms),
        "warm_sequences_per_s": summary_stats(sequences_per_s),
        "warm_timesteps_per_s": summary_stats(timesteps_per_s),
    }


def run_metadata() -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def make_run_dir(root: Path, *, run_name: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = run_name or stamp
    path = root / label
    path.mkdir(parents=True, exist_ok=False)
    return path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, CaseSpec):
        return value.payload
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
