#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from e

from training import CentralTrainer


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_nan(value: Any) -> bool:
    try:
        return value != value
    except Exception:
        return False


def _mean_std(values: List[float]) -> Dict[str, float]:
    values = [v for v in values if v is not None and not _is_nan(v)]
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {"mean": float(mean), "std": 0.0}
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return {"mean": float(mean), "std": float(var ** 0.5)}


def _fmt(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple seeds and summarize results.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Seed list.")
    parser.add_argument("--out_dir", type=str, default="runs", help="Output directory for run JSONs.")
    parser.add_argument("--name", type=str, default=None, help="Base run name override.")
    parser.add_argument("--table_out", type=str, default=None, help="Optional output Markdown file.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    out_paths: List[str] = []

    for seed in args.seeds:
        cfg = load_config(args.config)
        cfg["seed"] = int(seed)
        cfg.setdefault("data", {})["seed"] = int(seed)

        results_cfg = cfg.setdefault("results", {})
        base_name = args.name or results_cfg.get("name", "multiseed")
        run_name = f"{base_name}_seed{seed}"
        results_cfg["name"] = run_name
        results_cfg["dir"] = args.out_dir
        results_cfg["enabled"] = True

        trainer = CentralTrainer(cfg)
        results = trainer.train()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.out_dir, f"{run_name}_{ts}.json")
        payload = {
            "run_name": run_name,
            "seed": int(seed),
            "config_path": args.config,
            "results": results,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        out_paths.append(out_path)

        test = results.get("test_metrics", {}) if isinstance(results, dict) else {}
        rows.append(
            {
                "seed": int(seed),
                "run_name": run_name,
                "test_loss": results.get("test_loss"),
                "test_ap": test.get("ap"),
                "test_recall_at_k": test.get("recall_at_k"),
                "test_cost": test.get("cost"),
                "best_val_loss": results.get("best_val_loss"),
                "best_epoch": results.get("best_epoch"),
            }
        )

    # Build summary table
    headers = [
        "seed",
        "run_name",
        "test_loss",
        "test_ap",
        "test_recall_at_k",
        "test_cost",
        "best_val_loss",
        "best_epoch",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(h, "")) for h in headers) + " |")

    # Aggregate stats
    agg_keys = ["test_loss", "test_ap", "test_recall_at_k", "test_cost", "best_val_loss"]
    agg = {k: _mean_std([r.get(k) for r in rows]) for k in agg_keys}
    lines.append("| mean | - | " + " | ".join(_fmt(agg[k]["mean"]) for k in agg_keys) + " | - |")
    lines.append("| std | - | " + " | ".join(_fmt(agg[k]["std"]) for k in agg_keys) + " | - |")

    table = "\n".join(lines)
    table_out = args.table_out or os.path.join(args.out_dir, f"{(args.name or 'multiseed')}_summary.md")
    with open(table_out, "w", encoding="utf-8") as f:
        f.write(table + "\n")

    print(f"Saved runs: {len(out_paths)}")
    for path in out_paths:
        print(f"  {path}")
    print(f"Saved summary table to {table_out}")


if __name__ == "__main__":
    main()
