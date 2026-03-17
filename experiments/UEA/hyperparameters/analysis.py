#!/usr/bin/env python3
"""Post-process and summarize UEA hyperparameter sweep results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _metric_value(result: dict[str, Any], metric: str) -> float:
    value = result.get(metric)
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def _sorted_results(results: list[dict[str, Any]], metric: str, goal: str) -> list[dict[str, Any]]:
    reverse = goal == "max"
    return sorted(results, key=lambda item: _metric_value(item, metric), reverse=reverse)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyze_sweep_results(
    results: list[dict[str, Any]],
    out_dir: Path,
    *,
    metric: str = "best_val_acc",
    goal: str = "max",
) -> None:
    """Write summary artifacts for a merged sweep result list."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        (out_dir / "analysis.md").write_text("# Sweep Analysis\n\nNo completed results were found.\n")
        return

    ranked = _sorted_results(results, metric, goal)

    long_rows: list[dict[str, Any]] = []
    for result in results:
        row = {
            "dataset": result.get("dataset", "unknown"),
            "run_name": result.get("run_name", "unknown"),
            "metric": metric,
            "metric_value": _metric_value(result, metric),
            "mean_test_acc": result.get("mean_test_acc", ""),
            "seeds": ",".join(str(s) for s in result.get("seeds", [])),
            "combo": json.dumps(result.get("combo", {}), sort_keys=True),
        }
        long_rows.append(row)

    _write_csv(
        out_dir / "results_long.csv",
        long_rows,
        ["dataset", "run_name", "metric", "metric_value", "mean_test_acc", "seeds", "combo"],
    )

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for result in ranked:
        by_dataset.setdefault(str(result.get("dataset", "unknown")), []).append(result)

    leaderboard_rows: list[dict[str, Any]] = []
    dataset_best: dict[str, Any] = {}
    for dataset, items in by_dataset.items():
        best = items[0]
        dataset_best[dataset] = best
        leaderboard_rows.append(
            {
                "dataset": dataset,
                "best_run_name": best.get("run_name", "unknown"),
                "metric": metric,
                "metric_value": _metric_value(best, metric),
                "mean_test_acc": best.get("mean_test_acc", ""),
                "combo": json.dumps(best.get("combo", {}), sort_keys=True),
                "seeds": ",".join(str(s) for s in best.get("seeds", [])),
            }
        )

    _write_csv(
        out_dir / "leaderboard.csv",
        leaderboard_rows,
        ["dataset", "best_run_name", "metric", "metric_value", "mean_test_acc", "combo", "seeds"],
    )

    with (out_dir / "best_per_dataset.json").open("w") as f:
        json.dump(dataset_best, f, indent=2)

    lines: list[str] = []
    lines.append("# Sweep Analysis")
    lines.append("")
    lines.append(f"- Total completed runs: {len(results)}")
    lines.append(f"- Metric: `{metric}` (goal: `{goal}`)")
    lines.append("")
    lines.append("## Best per Dataset")
    lines.append("")
    for row in leaderboard_rows:
        lines.append(
            f"- {row['dataset']}: {row['metric']}={float(row['metric_value']):.4f}, "
            f"test={row['mean_test_acc']}, run={row['best_run_name']}, combo={row['combo']}"
        )

    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `results_long.csv`: one row per completed run")
    lines.append("- `leaderboard.csv`: best run per dataset")
    lines.append("- `best_per_dataset.json`: machine-readable best configs")
    (out_dir / "analysis.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze merged UEA sweep results.")
    parser.add_argument("results_json", type=Path, help="Path to merged results.json")
    parser.add_argument("--metric", type=str, default="best_val_acc")
    parser.add_argument("--goal", type=str, choices=["max", "min"], default="max")
    args = parser.parse_args()

    with args.results_json.open("r") as f:
        results = json.load(f)
    if not isinstance(results, list):
        raise ValueError("results_json must contain a list of results.")

    analyze_sweep_results(results, args.results_json.parent, metric=args.metric, goal=args.goal)


if __name__ == "__main__":
    main()
