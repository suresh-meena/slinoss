#!/usr/bin/env python3
"""Benchmark the canonical v2x2ssd kernels and report CuTe/reference speedups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    DEFAULT_BATCH,
    DEFAULT_CHUNK,
    DEFAULT_DTYPE,
    DEFAULT_HEADS,
    DEFAULT_N,
    DEFAULT_P,
    DEFAULT_T,
    DIRECTIONS,
    STAGES,
    PerfConfig,
    benchmark,
    build_callable,
    dtype_from_str,
    ensure_cuda,
    format_header,
    seed_all,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", *STAGES),
        default="all",
        help="Kernel stage to benchmark.",
    )
    parser.add_argument(
        "--direction",
        choices=("both", *DIRECTIONS),
        default="both",
        help="Benchmark forward, backward, or both.",
    )
    parser.add_argument(
        "--backend",
        choices=("reference", "cute", "both"),
        default="both",
        help="Backend(s) to benchmark.",
    )
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    parser.add_argument("--P", type=int, default=DEFAULT_P)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16", "fp32"),
        default=DEFAULT_DTYPE,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ensure_cuda(args.device)
    seed_all(args.seed)

    cfg = PerfConfig(
        batch=args.batch,
        heads=args.heads,
        T=args.T,
        N=args.N,
        P=args.P,
        chunk_size=args.chunk_size,
        dtype=dtype_from_str(args.dtype),
        device=args.device,
        seed=args.seed,
    )
    stages = STAGES if args.stage == "all" else (args.stage,)
    directions = DIRECTIONS if args.direction == "both" else (args.direction,)
    backends = ("reference", "cute") if args.backend == "both" else (args.backend,)

    rows: list[dict[str, float | str | list[float] | None]] = []
    for direction in directions:
        for stage in stages:
            by_backend: dict[str, dict[str, float | list[float]]] = {}
            for backend in backends:
                fn = build_callable(cfg, stage=stage, direction=direction, backend=backend)
                stats = benchmark(
                    fn,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeat=args.repeat,
                )
                by_backend[backend] = stats

            ref_mean = (
                float(by_backend["reference"]["mean_ms"])
                if "reference" in by_backend
                else None
            )
            cute_mean = (
                float(by_backend["cute"]["mean_ms"]) if "cute" in by_backend else None
            )
            speedup = (
                ref_mean / cute_mean
                if ref_mean is not None and cute_mean is not None and cute_mean > 0.0
                else None
            )

            for backend in backends:
                rows.append(
                    {
                        "direction": direction,
                        "stage": stage,
                        "backend": backend,
                        "mean_ms": float(by_backend[backend]["mean_ms"]),
                        "median_ms": float(by_backend[backend]["median_ms"]),
                        "stdev_ms": float(by_backend[backend]["stdev_ms"]),
                        "speedup_vs_reference": speedup if backend == "cute" else 1.0,
                        "samples_ms": by_backend[backend]["samples_ms"],
                    }
                )

    print(format_header(cfg))
    print("| direction | stage | backend | mean_ms | median_ms | stdev_ms | vs_ref |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        vs_ref = row["speedup_vs_reference"]
        vs_ref_str = "-" if vs_ref is None else f"{float(vs_ref):.3f}x"
        print(
            f"| {row['direction']} | {row['stage']} | {row['backend']} | "
            f"{float(row['mean_ms']):.6f} | {float(row['median_ms']):.6f} | "
            f"{float(row['stdev_ms']):.6f} | {vs_ref_str} |"
        )

    if args.json_out is not None:
        payload = {
            "header": format_header(cfg),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
            "rows": rows,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"json: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
