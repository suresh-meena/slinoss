#!/usr/bin/env python3
"""Prebuild cached UEA dataset splits for configured datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataloader import create_dataloaders
from utils import get_available_datasets, load_config, specialize_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prebuild cached UEA dataset splits.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
        help="Path to the UEA config file.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset subset. Defaults to all datasets declared in the config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_config = load_config(args.config)
    datasets = args.datasets or get_available_datasets(base_config)
    if not datasets:
        raise ValueError("No datasets found in the config.")

    for dataset in datasets:
        cfg = specialize_config(base_config, dataset=dataset)
        print(f"Processing {dataset}...")
        loaders, splits = create_dataloaders(cfg)
        print(
            "  done:"
            f" train={len(loaders['train'].dataset)}"
            f" val={len(loaders['val'].dataset)}"
            f" test={len(loaders['test'].dataset)}"
            f" features={splits.num_features}"
            f" classes={splits.num_classes}"
        )


if __name__ == "__main__":
    main()
