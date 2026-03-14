#!/usr/bin/env python3
"""Generate paper-style summaries for throughput_comparison benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <results.parent>/analysis.",
    )
    parser.add_argument(
        "--baseline",
        default="slinoss",
        help="Model key used for relative speedup tables.",
    )
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_results(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("kind") == "throughput_comparison", "Unexpected payload kind.")
    return payload


def _flatten_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_name, case_payload in payload["cases"].items():
        case = dict(case_payload["case"])
        for model_name, model_payload in case_payload["models"].items():
            params = dict(model_payload["params"])
            for measure_name, measure_payload in model_payload["measurements"].items():
                rows.append(
                    {
                        "case_name": case_name,
                        "suite": case["suite"],
                        "sweep_key": case.get("sweep_key"),
                        "sweep_value": case.get("sweep_value"),
                        "description": case["description"],
                        "batch_size": case["batch_size"],
                        "seq_len": case["seq_len"],
                        "hidden_dim": case["hidden_dim"],
                        "layers": case["layers"],
                        "model_name": model_name,
                        "framework": model_payload["framework"],
                        "family": model_payload["family"],
                        "dtype": model_payload["dtype"],
                        "device": model_payload["device"],
                        "measure": measure_name,
                        "cold_ms": float(measure_payload["cold_ms"]),
                        "warm_step_mean_ms": float(
                            measure_payload["warm_step_ms"]["mean"]
                        ),
                        "warm_step_stdev_ms": float(
                            measure_payload["warm_step_ms"]["stdev"]
                        ),
                        "warm_sequences_per_s": float(
                            measure_payload["warm_sequences_per_s"]["mean"]
                        ),
                        "warm_timesteps_per_s": float(
                            measure_payload["warm_timesteps_per_s"]["mean"]
                        ),
                        "params_m": float(params["millions"]),
                        "params_mb": float(params["megabytes"]),
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(rows, "No rows available for CSV output.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    sweep_value = row["sweep_value"]
    if sweep_value is None:
        sweep_value = -1
    return (row["case_name"], row["model_name"], row["measure"], sweep_value)


def _default_case_rows(rows: list[dict[str, Any]], *, measure: str) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in rows
            if row["case_name"] == "default" and row["measure"] == measure
        ],
        key=lambda row: row["model_name"],
    )


def _markdown_table(headers: list[str], values: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in values)
    return "\n".join(lines)


def _speedup_map(
    rows: list[dict[str, Any]],
    *,
    baseline: str,
    measure: str,
) -> dict[str, float]:
    selected = _default_case_rows(rows, measure=measure)
    baseline_row = next((row for row in selected if row["model_name"] == baseline), None)
    if baseline_row is None:
        return {}
    baseline_tps = float(baseline_row["warm_timesteps_per_s"])
    if baseline_tps <= 0.0:
        return {}
    return {
        row["model_name"]: float(row["warm_timesteps_per_s"]) / baseline_tps
        for row in selected
    }


def _write_report(path: Path, rows: list[dict[str, Any]], *, baseline: str) -> None:
    forward_rows = _default_case_rows(rows, measure="forward")
    train_rows = _default_case_rows(rows, measure="train_step")
    backward_rows = _default_case_rows(rows, measure="backward")

    sections: list[str] = ["# Throughput Comparison Report"]

    for title, selected_rows in (
        ("Forward Throughput", forward_rows),
        ("Backward Throughput", backward_rows),
        ("Train-Step Throughput", train_rows),
    ):
        if not selected_rows:
            continue
        headers = [
            "Model",
            "Framework",
            "Warm ms",
            "Warm timesteps/s",
            "Cold ms",
            "Params (M)",
        ]
        values = [
            [
                row["model_name"],
                row["framework"],
                f"{row['warm_step_mean_ms']:.3f}",
                f"{row['warm_timesteps_per_s']:.2f}",
                f"{row['cold_ms']:.3f}",
                f"{row['params_m']:.3f}",
            ]
            for row in selected_rows
        ]
        sections.append(f"## {title}")
        sections.append(_markdown_table(headers, values))

    speedup_headers = ["Model", "Forward x", "Train-step x"]
    forward_speedups = _speedup_map(rows, baseline=baseline, measure="forward")
    train_speedups = _speedup_map(rows, baseline=baseline, measure="train_step")
    speedup_models = sorted(set(forward_speedups) | set(train_speedups))
    if speedup_models:
        sections.append(f"## Relative Speedup vs `{baseline}`")
        sections.append(
            _markdown_table(
                speedup_headers,
                [
                    [
                        model_name,
                        f"{forward_speedups.get(model_name, float('nan')):.2f}",
                        f"{train_speedups.get(model_name, float('nan')):.2f}",
                    ]
                    for model_name in speedup_models
                ],
            )
        )

    sections.append("## Suggested Paper Assets")
    sections.append(
        "- Main table: warm steady-state `forward`, `backward`, and `train_step` throughput on `default`."
    )
    sections.append("- Figure: throughput scaling with sequence length.")
    sections.append("- Figure: throughput scaling with batch size.")
    sections.append("- Figure: cold-start latency versus warm steady-state latency.")
    sections.append("- Figure: throughput versus parameter count on the default case.")

    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def _configure_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk", palette="deep")


def _plot_default_bars(
    rows: list[dict[str, Any]],
    *,
    measure: str,
    field: str,
    ylabel: str,
    output_path: Path,
) -> None:
    selected = _default_case_rows(rows, measure=measure)
    if not selected:
        return
    x_labels = [row["model_name"] for row in selected]
    y_values = [float(row[field]) for row in selected]

    _configure_theme()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=x_labels, y=y_values, ax=ax, palette="deep")
    ax.set_ylabel(ylabel)
    ax.set_title(f"default: {measure} {ylabel}")
    ax.set_xlabel("Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_suite_lines(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    measure: str,
    output_path: Path,
) -> None:
    selected = [
        row
        for row in rows
        if row["suite"] == suite and row["measure"] == measure and row["sweep_key"] is not None
    ]
    if not selected:
        return

    x_label = str(selected[0]["sweep_key"])
    _configure_theme()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_name in sorted({row["model_name"] for row in selected}):
        model_rows = sorted(
            [row for row in selected if row["model_name"] == model_name],
            key=lambda row: float(row["sweep_value"]),
        )
        x_values = [float(row["sweep_value"]) for row in model_rows]
        y_values = [float(row["warm_timesteps_per_s"]) for row in model_rows]
        sns.lineplot(x=x_values, y=y_values, marker="o", label=model_name, ax=ax)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Warm timesteps/s")
    ax.set_title(f"{suite}: {measure} throughput scaling")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_param_scatter(rows: list[dict[str, Any]], *, output_path: Path) -> None:
    selected = _default_case_rows(rows, measure="train_step")
    if not selected:
        return

    _configure_theme()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for row in selected:
        sns.scatterplot(
            x=[float(row["params_m"])],
            y=[float(row["warm_timesteps_per_s"])],
            label=row["model_name"],
            ax=ax,
            s=80,
            legend=False,
        )
        ax.annotate(
            row["model_name"],
            (float(row["params_m"]), float(row["warm_timesteps_per_s"])),
            textcoords="offset points",
            xytext=(4, 4),
        )
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Warm train-step timesteps/s")
    ax.set_title("default: throughput versus parameter count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> int:
    args = _parse_args()
    payload = _load_results(args.results)
    rows = sorted(_flatten_rows(payload), key=_sort_key)
    output_dir = (
        args.output_dir if args.output_dir is not None else args.results.parent / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "summary.csv", rows)
    _write_report(output_dir / "report.md", rows, baseline=args.baseline)
    _plot_default_bars(
        rows,
        measure="forward",
        field="warm_timesteps_per_s",
        ylabel="Warm timesteps/s",
        output_path=output_dir / "default_forward_timesteps_per_s.png",
    )
    _plot_default_bars(
        rows,
        measure="train_step",
        field="warm_timesteps_per_s",
        ylabel="Warm timesteps/s",
        output_path=output_dir / "default_train_step_timesteps_per_s.png",
    )
    _plot_default_bars(
        rows,
        measure="train_step",
        field="cold_ms",
        ylabel="Cold ms",
        output_path=output_dir / "default_train_step_cold_ms.png",
    )
    for suite in ("sequence_length", "batch_size", "hidden_dim", "layers"):
        _plot_suite_lines(
            rows,
            suite=suite,
            measure="train_step",
            output_path=output_dir / f"{suite}_train_step_scaling.png",
        )
    _plot_param_scatter(rows, output_path=output_dir / "default_param_scatter.png")

    print(f"[done] analysis={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
