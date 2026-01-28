#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(val: Any) -> str:
    try:
        if isinstance(val, float):
            return f"{val:.4f}"
    except Exception:
        pass
    return str(val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a minimal Markdown results table.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Result JSON files.")
    parser.add_argument("--out", default=None, help="Optional output Markdown file.")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for path in args.inputs:
        if not os.path.exists(path):
            continue
        payload = load_json(path)
        results = payload.get("results", {})
        test = results.get("test_metrics", {})
        row = {
            "run_name": payload.get("run_name", os.path.basename(path)),
            "test_loss": results.get("test_loss"),
            "test_ap": test.get("ap"),
            "test_recall_at_k": test.get("recall_at_k"),
            "test_cost": test.get("cost"),
            "best_val_loss": results.get("best_val_loss"),
            "best_epoch": results.get("best_epoch"),
        }
        rows.append(row)

    headers = [
        "run_name",
        "test_loss",
        "test_ap",
        "test_recall_at_k",
        "test_cost",
        "best_val_loss",
        "best_epoch",
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h, "")) for h in headers) + " |")

    table = "\n".join(lines)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"Saved table to {args.out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
