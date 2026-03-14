#!/usr/bin/env python3
"""Analysis script for UEA SLinOSS experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _summary_kind(summary: Any) -> str | None:
    if not isinstance(summary, dict):
        return None
    if isinstance(summary.get("history"), list):
        return "run"
    if "best_val_acc" in summary:
        return "sweep_run"
    return None


def _configure_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk", palette="deep")


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except Exception:
        return path


def _format_relative(root: Path, target: Path) -> str:
    try:
        return str(target.relative_to(root))
    except ValueError:
        try:
            return str(target.resolve())
        except Exception:
            return str(target)


def _best_epoch_row(history: pd.DataFrame) -> pd.Series:
    return history.loc[history["val_acc"].idxmax()]


def _append_config_block(report: list[str], title: str, config: dict[str, Any], keys: list[str]) -> None:
    report.append(title)
    for key in keys:
        if key in config:
            report.append(f"  {key:15}: {config[key]}")


def plot_run_metrics(summary: dict[str, Any], out_dir: Path) -> None:
    """Plot loss and accuracy with the best validation epoch highlighted."""
    history = pd.DataFrame(summary["history"])
    if history.empty:
        print("No history found; skipping run plot.")
        return

    best = _best_epoch_row(history)
    _configure_theme()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    config = summary["config"]
    title = f"{config['dataset']} | {config['run_name']}"

    sns.lineplot(data=history, x="epoch", y="train_loss", ax=axes[0], label="Train", linewidth=2.0)
    sns.lineplot(data=history, x="epoch", y="val_loss", ax=axes[0], label="Val", linewidth=2.0)
    axes[0].axvline(best["epoch"], color="black", linestyle="--", linewidth=1.2, label="Best epoch")
    axes[0].axhline(summary["test_loss"], color="tab:red", linestyle=":", linewidth=1.2, label="Test")
    axes[0].scatter([best["epoch"]], [best["val_loss"]], color="black", s=70, zorder=5)
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    sns.lineplot(data=history, x="epoch", y="train_acc", ax=axes[1], label="Train", linewidth=2.0)
    sns.lineplot(data=history, x="epoch", y="val_acc", ax=axes[1], label="Val", linewidth=2.0)
    axes[1].axvline(best["epoch"], color="black", linestyle="--", linewidth=1.2, label="Best epoch")
    axes[1].axhline(summary["test_acc"], color="tab:red", linestyle=":", linewidth=1.2, label="Test")
    axes[1].scatter([best["epoch"]], [best["val_acc"]], color="black", s=70, zorder=5)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        (
            f"Best epoch {int(best['epoch'])}: val_acc={best['val_acc']:.4f}, "
            f"val_loss={best['val_loss']:.4f} | "
            f"test_acc={summary['test_acc']:.4f}, test_loss={summary['test_loss']:.4f}"
        ),
        ha="center",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.95))
    output_path = out_dir / "metrics_plot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to {output_path}")


def generate_run_report(summary: dict[str, Any], out_dir: Path) -> None:
    """Generate a concise text report for a single training run."""
    config = summary["config"]
    history = pd.DataFrame(summary["history"])
    if history.empty:
        raise ValueError("summary history is empty.")

    best = _best_epoch_row(history)
    last = history.iloc[-1]

    report: list[str] = []
    report.append("=" * 64)
    report.append("UEA SLinOSS RUN REPORT")
    report.append("=" * 64)
    report.append(f"Dataset:         {config['dataset']}")
    report.append(f"Run name:        {config['run_name']}")
    report.append(f"Epochs trained:  {int(last['epoch'])}")
    report.append(f"Best epoch:      {int(best['epoch'])}")
    report.append(f"Best val acc:    {best['val_acc']:.4f}")
    report.append(f"Best val loss:   {best['val_loss']:.4f}")
    report.append(f"Final val acc:   {last['val_acc']:.4f}")
    report.append(f"Final val loss:  {last['val_loss']:.4f}")
    report.append(f"Test acc:        {summary['test_acc']:.4f}")
    report.append(f"Test loss:       {summary['test_loss']:.4f}")
    report.append(f"Gap @ best:      {best['train_acc'] - best['val_acc']:+.4f}")
    report.append("-" * 64)

    _append_config_block(report, "Data:", config, ["data_root", "val_fraction", "include_time", "normalize"])
    _append_config_block(report, "Training:", config, ["seed", "epochs", "batch_size", "lr", "weight_decay", "grad_clip"])
    _append_config_block(report, "Model:", config, ["d_model", "n_layers", "d_state", "expand", "d_head", "d_conv", "chunk_size", "dropout"])
    _append_config_block(report, "Backend:", config, ["scan_backend"])

    report.append("-" * 64)
    report.append("Last 5 epochs:")
    for _, row in history.tail(5).iterrows():
        report.append(
            f"  Epoch {int(row['epoch']):03d}: "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f}"
        )
    report.append("=" * 64)

    report_text = "\n".join(report)
    output_path = out_dir / "report.txt"
    output_path.write_text(report_text)
    print(f"Report saved to {output_path}")
    print(report_text)


def _combo_text(result: dict[str, Any]) -> str:
    combo = result.get("combo")
    if isinstance(combo, dict) and combo:
        return json.dumps(combo, sort_keys=True)
    config = result.get("config")
    if isinstance(config, dict):
        keys = ["lr", "batch_size", "d_model", "n_layers", "d_state", "dropout"]
        compact = {key: config[key] for key in keys if key in config}
        if compact:
            return json.dumps(compact, sort_keys=True)
    return "{}"


def plot_sweep_results(results: list[dict[str, Any]], out_dir: Path) -> None:
    """Plot sweep winners and metric spread without overcrowding."""
    if not results:
        print("No sweep results found; skipping sweep plot.")
        return

    rows: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        rows.append(
            {
                "dataset": result.get("dataset", "unknown"),
                "run_name": result.get("run_name", f"run_{index:03d}"),
                "best_val_acc": result["best_val_acc"],
            }
        )
    df = pd.DataFrame(rows)
    _configure_theme()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    best_df = (
        df.sort_values(["dataset", "best_val_acc"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .head(1)
        .sort_values("best_val_acc", ascending=False)
    )
    sns.barplot(data=best_df, x="dataset", y="best_val_acc", ax=axes[0], color="tab:blue")
    axes[0].set_title("Best Validation Accuracy by Dataset")
    axes[0].set_xlabel("Dataset")
    axes[0].set_ylabel("Best val_acc")
    axes[0].tick_params(axis="x", rotation=35)
    upper = min(1.0, float(best_df["best_val_acc"].max()) + 0.05)
    axes[0].set_ylim(0.0, max(0.05, upper))

    sns.boxplot(data=df, x="dataset", y="best_val_acc", ax=axes[1], color="lightsteelblue", showfliers=False)
    sns.stripplot(data=df, x="dataset", y="best_val_acc", ax=axes[1], color="tab:blue", alpha=0.65, size=4)
    axes[1].set_title("Sweep Distribution by Dataset")
    axes[1].set_xlabel("Dataset")
    axes[1].set_ylabel("best_val_acc")
    axes[1].tick_params(axis="x", rotation=35)

    fig.suptitle("UEA Sweep Summary", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = out_dir / "sweep_plot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to {output_path}")


def generate_sweep_report(results: list[dict[str, Any]], out_dir: Path) -> None:
    """Generate a concise sweep summary."""
    if not results:
        raise ValueError("results.json does not contain any completed runs.")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result.get("dataset", "unknown"), []).append(result)

    report: list[str] = []
    report.append("=" * 64)
    report.append("UEA SLinOSS SWEEP REPORT")
    report.append("=" * 64)
    report.append(f"Completed runs:  {len(results)}")
    report.append(f"Datasets:        {len(grouped)}")
    report.append("-" * 64)

    for dataset in sorted(grouped):
        ranked = sorted(grouped[dataset], key=lambda item: item["best_val_acc"], reverse=True)
        best = ranked[0]
        report.append(f"Dataset:         {dataset}")
        report.append(f"Best val acc:    {best['best_val_acc']:.4f}")
        report.append(f"Best run:        {best.get('run_name', 'n/a')}")
        report.append(f"Best combo:      {_combo_text(best)}")
        report.append("Top 3:")
        for item in ranked[:3]:
            report.append(
                f"  {item.get('run_name', 'n/a'):20} "
                f"val_acc={item['best_val_acc']:.4f} combo={_combo_text(item)}"
            )
        report.append("-" * 64)

    output_path = out_dir / "sweep_report.txt"
    output_path.write_text("\n".join(report))
    print(f"Report saved to {output_path}")
    print("\n".join(report))


def generate_tree_report(path: Path) -> None:
    run_summaries: list[tuple[Path, dict[str, Any]]] = []
    sweep_runs: list[tuple[Path, dict[str, Any]]] = []
    sweep_roots: list[tuple[Path, list[dict[str, Any]]]] = []

    for summary_path in sorted(path.rglob("summary.json")):
        summary = _load_json(summary_path)
        kind = _summary_kind(summary)
        if kind == "run":
            run_summaries.append((summary_path.parent, summary))
        elif kind == "sweep_run":
            sweep_runs.append((summary_path.parent, summary))

    for results_path in sorted(path.rglob("results.json")):
        results = _load_json(results_path)
        if isinstance(results, list):
            sweep_roots.append((results_path.parent, results))

    root = _safe_resolve(path)
    report: list[str] = []
    report.append("=" * 64)
    report.append("UEA SLinOSS TREE REPORT")
    report.append("=" * 64)
    report.append(f"Root:            {root}")
    report.append(f"Run summaries:   {len(run_summaries)}")
    report.append(f"Sweep run dirs:  {len(sweep_runs)}")
    report.append(f"Sweep roots:     {len(sweep_roots)}")
    report.append("-" * 64)

    if run_summaries:
        best_run_dir, best_run = max(
            run_summaries,
            key=lambda item: float(item[1]["best_val_acc"]),
        )
        report.append(
            f"Best run:        {_format_relative(root, best_run_dir)} "
            f"(dataset={best_run['config']['dataset']}, "
            f"best_val_acc={best_run['best_val_acc']:.4f})"
        )
        report.append("Top runs:")
        ranked_runs = sorted(
            run_summaries,
            key=lambda item: float(item[1]["best_val_acc"]),
            reverse=True,
        )[:10]
        for run_dir, summary in ranked_runs:
            report.append(
                f"  {_format_relative(root, run_dir)} "
                f"best_val_acc={summary['best_val_acc']:.4f} "
                f"test_acc={summary['test_acc']:.4f}"
            )
        report.append("-" * 64)

    if sweep_roots:
        report.append("Sweep roots:")
        for sweep_dir, results in sweep_roots:
            if results:
                best = max(results, key=lambda item: float(item["best_val_acc"]))
                best_metric = f"{best['best_val_acc']:.4f}"
                best_name = best.get("run_name", "n/a")
            else:
                best_metric = "n/a"
                best_name = "n/a"
            report.append(
                f"  {_format_relative(root, sweep_dir)} "
                f"runs={len(results)} best_run={best_name} best_val_acc={best_metric}"
            )
        report.append("-" * 64)

    report.append("Analysis hint:")
    report.append("  Pass a run directory, a sweep directory, or a sweep child run directory.")
    report.append("=" * 64)

    output_path = path / "tree_report.txt"
    output_path.write_text("\n".join(report))
    print(f"Tree report saved to {output_path}")
    print("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a UEA run or sweep directory.")
    parser.add_argument("path", type=Path, help="Run directory with summary.json or sweep directory with results.json")
    args = parser.parse_args()
    path = _safe_resolve(args.path)
    summary_path = path / "summary.json"
    results_path = path / "results.json"

    if summary_path.exists():
        summary = _load_json(summary_path)
        kind = _summary_kind(summary)
        if kind == "run":
            plot_run_metrics(summary, path)
            generate_run_report(summary, path)
            return
        if kind == "sweep_run":
            plot_sweep_results([summary], path)
            generate_sweep_report([summary], path)
            return
        raise ValueError(f"Unsupported summary schema in {summary_path}.")

    if results_path.exists():
        results = _load_json(results_path)
        _require(isinstance(results, list), f"Expected {results_path} to contain a list.")
        plot_sweep_results(results, path)
        generate_sweep_report(results, path)
        return

    if any(path.rglob("summary.json")) or any(path.rglob("results.json")):
        generate_tree_report(path)
        return

    print(f"Error: neither summary.json nor results.json was found in {path}")


if __name__ == "__main__":
    main()
