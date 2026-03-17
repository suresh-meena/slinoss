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
from accelerate import Accelerator

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataloader import create_dataloaders
from model import UEAClassifier
from trainer import Trainer
from utils import (
    configure_optimizer,
    get_available_datasets,
    get_run_dir,
    load_config,
    resolve_mixed_precision,
    resolve_sweep_config,
    set_seed,
    specialize_config,
)

from analysis import analyze_sweep_results


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
        return configured
    return [specialize_config(base_config)["dataset"]]


def _is_better(candidate: float, best: float, goal: str) -> bool:
    if goal == "min":
        return candidate < best
    return candidate > best


def _metric_value(result: dict[str, Any], metric: str) -> float:
    if metric not in result:
        raise KeyError(f"Sweep metric '{metric}' was not found in result keys: {sorted(result)}")
    return float(result[metric])


def _checkpoint_path(sweep_dir: Path, dataset: str, rank: int) -> Path:
    return sweep_dir / dataset / "_partials" / f"rank_{rank:02d}.checkpoint.json"


def _load_checkpoint(path: Path) -> tuple[set[int], list[dict[str, Any]]]:
    if not path.exists():
        return set(), []
    try:
        with path.open("r") as f:
            data = json.load(f)
        completed = set(int(x) for x in data.get("completed_indices", []))
        results = data.get("results", [])
        if not isinstance(results, list):
            return completed, []
        return completed, results
    except Exception:
        return set(), []


def _save_checkpoint(path: Path, completed_indices: set[int], results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_indices": sorted(completed_indices),
        "results": results,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def _run_combo_seed(config: dict[str, Any], device: torch.device, logger: logging.Logger) -> tuple[float, float]:
    loaders, splits = create_dataloaders(config)
    model = UEAClassifier(
        input_dim=splits.num_features,
        num_classes=splits.num_classes,
        **config,
    ).to(device)
    if bool(config.get("torch_compile", False)):
        compile_mode = str(config.get("torch_compile_mode", "default"))
        model = torch.compile(model, mode=compile_mode)
    optimizer = configure_optimizer(
        model,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    prepared_loaders = {"train": loaders["train"], "val": loaders["val"], "test": loaders["test"]}
    trainer = Trainer(model, optimizer, device, logger, grad_clip=config["grad_clip"], accelerator=None)

    best_val_acc = float("-inf")
    best_state = None

    if "num_steps" in config:
        num_steps = int(config["num_steps"])
        print_steps = int(config.get("print_steps", 1000))
        patience = int(config.get("early_stopping_evals", 10))
        no_improvement = 0
        train_iter = iter(prepared_loaders["train"])
        for step in range(1, num_steps + 1):
            try:
                x, lengths, y = next(train_iter)
            except StopIteration:
                train_iter = iter(prepared_loaders["train"])
                x, lengths, y = next(train_iter)
            trainer.train_step(x, lengths, y)
            if step % print_steps != 0:
                continue
            _, val_acc = trainer.evaluate(prepared_loaders["val"], desc=f"Val @ step {step}")
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                no_improvement = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improvement += 1
                if patience > 0 and no_improvement >= patience:
                    break
    else:
        for epoch in range(1, config["epochs"] + 1):
            trainer.train_epoch(prepared_loaders["train"], epoch)
            _, val_acc = trainer.evaluate(prepared_loaders["val"], desc=f"Val Epoch {epoch}")
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_acc = trainer.evaluate(prepared_loaders["test"], desc="Test")
    return float(best_val_acc), float(test_acc)


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
    parser.add_argument("--num-seeds", type=int, default=None, help="Optional cap on number of seeds per combo.")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from rank checkpoints and skip completed assigned combos.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints and rerun assigned combos.",
    )
    args = parser.parse_args()

    if args.no_resume:
        args.resume = False

    base_config = load_config(args.base_config)
    datasets = _select_datasets(base_config, args.datasets)
    if not datasets:
        raise ValueError("No datasets provided for the sweep.")

    first_runtime = specialize_config(base_config, dataset=datasets[0])
    sweep_dir = Path(first_runtime["runs_root"]) / "_sweeps" / args.sweep_name
    accelerator = Accelerator(mixed_precision=resolve_mixed_precision(first_runtime.get("mixed_precision")))
    logger = setup_sweep_logging(sweep_dir) if accelerator.is_main_process else logging.getLogger("UEA_SWEEP.worker")
    if not accelerator.is_main_process:
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    logger.info("Starting sweep '%s' on %s.", args.sweep_name, ", ".join(datasets))
    logger.info("Using device: %s", device)
    if accelerator.is_main_process:
        logger.info("Process topology: world_size=%d", world_size)

    for dataset in datasets:
        dataset_results: list[dict[str, Any]] = []
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

        assigned = [(index, combo) for index, combo in enumerate(combinations) if index % world_size == rank]
        logger.info("Rank %d assigned %d/%d combos for dataset %s", rank, len(assigned), len(combinations), dataset)

        checkpoint_file = _checkpoint_path(sweep_dir, dataset, rank)
        completed_indices, checkpoint_results = _load_checkpoint(checkpoint_file) if args.resume else (set(), [])
        if checkpoint_results:
            dataset_results.extend(checkpoint_results)
        if args.resume:
            logger.info(
                "Rank %d resume state: %d completed combos recovered from %s",
                rank,
                len(completed_indices),
                checkpoint_file,
            )

        assigned_pending = [(idx, combo) for idx, combo in assigned if idx not in completed_indices]
        logger.info("Rank %d pending combos for %s: %d", rank, dataset, len(assigned_pending))

        for index, combo in assigned_pending:
            config = dict(base_runtime)
            config.update(fixed)
            config.update(combo)
            config["run_name"] = f"c{index:03d}_{dataset}"

            run_dir = get_run_dir(sweep_dir, dataset, config["run_name"])
            logger.info("Rank %d run %d/%d: %s", rank, index + 1, len(combinations), json.dumps(combo, sort_keys=True))

            try:
                seeds = [int(s) for s in config.get("seeds", [config["seed"]])]
                if args.num_seeds is not None:
                    seeds = seeds[: max(args.num_seeds, 1)]

                seed_vals: list[float] = []
                seed_tests: list[float] = []
                for seed in seeds:
                    seed_cfg = dict(config)
                    seed_cfg["seed"] = seed
                    set_seed(seed)
                    best_val_acc, test_acc = _run_combo_seed(seed_cfg, device, logger)
                    seed_vals.append(best_val_acc)
                    seed_tests.append(test_acc)

                mean_best_val = sum(seed_vals) / len(seed_vals)
                mean_test = sum(seed_tests) / len(seed_tests)

                result = {
                    "dataset": dataset,
                    "run_name": config["run_name"],
                    "metric": sweep_cfg["metric"],
                    "goal": sweep_cfg["goal"],
                    "best_val_acc": mean_best_val,
                    "mean_test_acc": mean_test,
                    "seeds": seeds,
                    "config": config,
                    "combo": combo,
                    "fixed": fixed,
                }
                result["metric_value"] = _metric_value(result, str(sweep_cfg["metric"]))
                dataset_results.append(result)
                with (run_dir / "summary.json").open("w") as f:
                    json.dump(result, f, indent=2)
                completed_indices.add(index)
                _save_checkpoint(checkpoint_file, completed_indices, dataset_results)

                logger.info("Done. mean_best_val_acc=%.4f mean_test_acc=%.4f", mean_best_val, mean_test)
            except Exception as exc:
                logger.error("Error in run %d for dataset %s: %s", index, dataset, exc)
                continue

        partial_dir = sweep_dir / dataset / "_partials"
        partial_dir.mkdir(parents=True, exist_ok=True)
        partial_path = partial_dir / f"rank_{rank:02d}.json"
        with partial_path.open("w") as f:
            json.dump(dataset_results, f, indent=2)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            merged_results: list[dict[str, Any]] = []
            for worker_rank in range(world_size):
                worker_path = partial_dir / f"rank_{worker_rank:02d}.json"
                if not worker_path.exists():
                    continue
                with worker_path.open("r") as f:
                    worker_results = json.load(f)
                if isinstance(worker_results, list):
                    merged_results.extend(worker_results)

            best_metric = float("inf") if sweep_cfg["goal"] == "min" else float("-inf")
            best_result: dict[str, Any] | None = None
            for result in merged_results:
                metric_value = _metric_value(result, str(sweep_cfg["metric"]))
                if _is_better(metric_value, best_metric, sweep_cfg["goal"]):
                    best_metric = metric_value
                    best_result = result

            dataset_results_path = sweep_dir / dataset / "results.json"
            with dataset_results_path.open("w") as f:
                json.dump(merged_results, f, indent=2)

            analysis_dir = sweep_dir / dataset / "analysis"
            analyze_sweep_results(
                merged_results,
                analysis_dir,
                metric=str(sweep_cfg["metric"]),
                goal=str(sweep_cfg.get("goal", "max")),
            )

            if best_result is not None:
                logger.info(
                    "Best result for %s: %s=%.4f combo=%s",
                    dataset,
                    sweep_cfg["metric"],
                    best_metric,
                    json.dumps(best_result["combo"], sort_keys=True),
                )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Also emit a top-level merged file across all dataset results for convenience.
        all_results: list[dict[str, Any]] = []
        for dataset in datasets:
            dataset_results_path = sweep_dir / dataset / "results.json"
            if not dataset_results_path.exists():
                continue
            with dataset_results_path.open("r") as f:
                ds_results = json.load(f)
            if isinstance(ds_results, list):
                all_results.extend(ds_results)
        with (sweep_dir / "results.json").open("w") as f:
            json.dump(all_results, f, indent=2)
        analyze_sweep_results(all_results, sweep_dir / "analysis", metric="best_val_acc", goal="max")
        logger.info("Sweep complete. Results saved to %s", sweep_dir / "results.json")


if __name__ == "__main__":
    main()
