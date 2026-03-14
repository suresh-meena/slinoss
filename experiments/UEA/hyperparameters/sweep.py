#!/usr/bin/env python3
"""Hyperparameter sweep script for UEA SLinOSS experiments."""

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
from model import UEAClassifier
from trainer import Trainer
from utils import (
    configure_optimizer,
    get_available_datasets,
    get_run_dir,
    load_config,
    resolve_sweep_config,
    set_seed,
    specialize_config,
)


def setup_sweep_logging(sweep_dir: Path) -> logging.Logger:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("UEA_SWEEP")
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


def _select_datasets(base_config: dict[str, Any], cli_datasets: list[str] | None) -> list[str]:
    if cli_datasets:
        return cli_datasets
    configured = get_available_datasets(base_config)
    if configured:
        return configured[:1]
    return [specialize_config(base_config)["dataset"]]


def _is_better(candidate: float, best: float, goal: str) -> bool:
    if goal == "min":
        return candidate < best
    return candidate > best


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UEA SLinOSS hyperparameter sweep.")
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
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    datasets = _select_datasets(base_config, args.datasets)
    if not datasets:
        raise ValueError("No datasets provided for the sweep.")

    first_runtime = specialize_config(base_config, dataset=datasets[0])
    sweep_dir = Path(first_runtime["runs_root"]) / "_sweeps" / args.sweep_name
    logger = setup_sweep_logging(sweep_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting sweep '%s' on %s.", args.sweep_name, ", ".join(datasets))
    logger.info("Using device: %s", device)

    results: list[dict[str, Any]] = []

    for dataset in datasets:
        logger.info("=== Dataset: %s ===", dataset)
        base_runtime = specialize_config(base_config, dataset=dataset)
        sweep_cfg = resolve_sweep_config(base_config, dataset=dataset)
        fixed = dict(sweep_cfg["fixed"])
        combinations = _build_combinations(dict(sweep_cfg["grid"]))
        logger.info(
            "Metric=%s goal=%s combinations=%d fixed=%s",
            sweep_cfg["metric"],
            sweep_cfg["goal"],
            len(combinations),
            json.dumps(fixed, sort_keys=True),
        )

        best_metric = float("inf") if sweep_cfg["goal"] == "min" else float("-inf")
        best_result: dict[str, Any] | None = None

        for index, combo in enumerate(combinations):
            config = dict(base_runtime)
            config.update(fixed)
            config.update(combo)
            config["run_name"] = f"c{index:03d}_{dataset}"

            run_dir = get_run_dir(sweep_dir, dataset, config["run_name"])
            logger.info("Run %d/%d: %s", index + 1, len(combinations), json.dumps(combo, sort_keys=True))

            set_seed(config["seed"])
            try:
                loaders, splits = create_dataloaders(config)
                model = UEAClassifier(
                    input_dim=splits.num_features,
                    num_classes=splits.num_classes,
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

                best_val_acc = float("-inf")
                for epoch in range(1, config["epochs"] + 1):
                    trainer.train_epoch(loaders["train"], epoch)
                    _, val_acc = trainer.evaluate(loaders["val"], desc=f"Val Epoch {epoch}")
                    best_val_acc = max(best_val_acc, val_acc)

                result = {
                    "dataset": dataset,
                    "run_name": config["run_name"],
                    "metric": sweep_cfg["metric"],
                    "goal": sweep_cfg["goal"],
                    "best_val_acc": best_val_acc,
                    "config": config,
                    "combo": combo,
                    "fixed": fixed,
                }
                results.append(result)
                with (run_dir / "summary.json").open("w") as f:
                    json.dump(result, f, indent=2)

                logger.info("Done. best_val_acc=%.4f", best_val_acc)
                if _is_better(best_val_acc, best_metric, sweep_cfg["goal"]):
                    best_metric = best_val_acc
                    best_result = result
            except Exception as exc:
                logger.error("Error in run %d for dataset %s: %s", index, dataset, exc)
                continue

        if best_result is not None:
            logger.info(
                "Best result for %s: %s=%.4f combo=%s",
                dataset,
                sweep_cfg["metric"],
                best_metric,
                json.dumps(best_result["combo"], sort_keys=True),
            )

    with (sweep_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Sweep complete. Results saved to %s", sweep_dir / "results.json")


if __name__ == "__main__":
    main()
