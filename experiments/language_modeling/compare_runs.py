#!/usr/bin/env python3
"""Compare language-modeling run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _resolve_summary_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "summary.json"
        if candidate.exists():
            return candidate
    return path


def _load_summary(path: Path) -> dict[str, Any]:
    summary_path = _resolve_summary_path(path)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    data["_path"] = str(summary_path)
    return data


def _get_nested(data: dict[str, Any], *keys: str) -> Any:
    cursor: Any = data
    for key in keys:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def _render_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "run",
        "model",
        "params_m",
        "steps",
        "tokens_seen",
        "best_val_ppl",
        "last_val_ppl",
        "path",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(header, "")) for header in headers)
            + " |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare language-modeling run summaries.")
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories or summary.json files.")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    loaded = [_load_summary(path) for path in args.runs]
    rows = [
        {
            "run": item.get("run_name", ""),
            "model": _get_nested(item, "model", "type") or "",
            "params_m": f"{float(item.get('parameter_count_m', 0.0)):.3f}",
            "steps": int(item.get("global_step", 0)),
            "tokens_seen": int(item.get("tokens_seen", 0)),
            "best_val_ppl": f"{float(_get_nested(item, 'best_eval', 'ppl') or float('nan')):.4f}",
            "last_val_ppl": f"{float(_get_nested(item, 'last_eval', 'ppl') or float('nan')):.4f}",
            "path": item["_path"],
        }
        for item in loaded
    ]
    print(_render_markdown(rows))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
