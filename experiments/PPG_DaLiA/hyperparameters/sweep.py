#!/usr/bin/env python3
"""Hyperparameter sweep script for PPG-DaLiA SLinOSS experiments."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataloader import create_dataloaders
from model import PPGRegressor
from trainer import Trainer
from utils import (
    configure_optimizer,
    get_run_dir,
    load_config,
    resolve_sweep_config,
    set_seed,
    specialize_config,
)


def setup_sweep_logging(sweep_dir: Path) -> logging.Logger:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("PPG_SWEEP")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(sweep_dir / "sweep.log")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(ch)

    return logger


def _build_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys, values = zip(*grid.items())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _is_better(candidate: float, best: float, goal: str) -> bool:
    if goal == "min":
        return candidate < best
    return candidate > best


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PPG-DaLiA SLinOSS hyperparameter sweep.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
    )
    parser.add_argument(
        "--sweep-name",
        type=str,
        default=f"sweep_{time.strftime('%Y%m%d-%H%M%S')}",
    )
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    base_runtime = specialize_config(base_config)
    sweep_cfg = resolve_sweep_config(base_config)

    sweep_dir = Path(base_runtime["runs_root"]) / "_sweeps" / args.sweep_name
    logger = setup_sweep_logging(sweep_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fixed = dict(sweep_cfg["fixed"])
    combinations = _build_combinations(dict(sweep_cfg["grid"]))
    logger.info(
        "Starting PPG sweep '%s' on %s with %d combinations.",
        args.sweep_name,
        device,
        len(combinations),
    )
    logger.info(
        "Metric=%s goal=%s fixed=%s",
        sweep_cfg["metric"],
        sweep_cfg["goal"],
        json.dumps(fixed, sort_keys=True),
    )

    results: list[dict[str, Any]] = []
    best_metric = float("inf") if sweep_cfg["goal"] == "min" else float("-inf")
    best_result: dict[str, Any] | None = None

    for index, combo in enumerate(combinations):
        config = dict(base_runtime)
        config.update(fixed)
        config.update(combo)
        config["run_name"] = f"c{index:03d}"

        run_dir = get_run_dir(sweep_dir, config["run_name"])
        logger.info("Run %d/%d: %s", index + 1, len(combinations), json.dumps(combo, sort_keys=True))
        set_seed(config["seed"])

        try:
            loaders, splits = create_dataloaders(config)
            out_dim = int(splits.train_y.shape[-1])
            model = PPGRegressor(
                input_dim=splits.num_features,
                out_dim=out_dim,
                d_model=config["d_model"],
                n_layers=config["n_layers"],
                d_state=config["d_state"],
                expand=config["expand"],
                d_head=config["d_head"],
                d_conv=config["d_conv"],
                chunk_size=config["chunk_size"],
                dropout=config["dropout"],
                scan_backend=config["scan_backend"],
            ).to(device)

            optimizer = configure_optimizer(
                model,
                lr=config["lr"],
                weight_decay=config["weight_decay"],
            )
            trainer = Trainer(model, optimizer, device, logger, grad_clip=config["grad_clip"])

            best_val_mae = float("inf")
            for epoch in range(1, config["epochs"] + 1):
                trainer.train_epoch(loaders["train"], epoch)
                _, val_mae, _ = trainer.evaluate(loaders["val"], desc=f"Val Epoch {epoch}")
                best_val_mae = min(best_val_mae, val_mae)

            result = {
                "run_name": config["run_name"],
                "metric": sweep_cfg["metric"],
                "goal": sweep_cfg["goal"],
                "best_val_mae": best_val_mae,
                "config": config,
                "combo": combo,
                "fixed": fixed,
            }
            results.append(result)
            with (run_dir / "summary.json").open("w") as f:
                json.dump(result, f, indent=2)

            logger.info("Done. best_val_mae=%.4f", best_val_mae)
            if _is_better(best_val_mae, best_metric, sweep_cfg["goal"]):
                best_metric = best_val_mae
                best_result = result
        except Exception as exc:
            logger.error("Error in run %d: %s", index, exc)
            continue

    if best_result is not None:
        logger.info(
            "Best result: %s=%.4f combo=%s",
            sweep_cfg["metric"],
            best_metric,
            json.dumps(best_result["combo"], sort_keys=True),
        )

    with (sweep_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Sweep complete. Results saved to %s", sweep_dir / "results.json")


if __name__ == "__main__":
    main()
