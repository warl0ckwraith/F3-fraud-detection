#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from e

from training import CentralTrainer


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train centralized F³ prototype.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/centralized_example.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="Optional path to save results JSON.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    trainer = CentralTrainer(cfg)
    results = trainer.train()

    rcfg = cfg.get("results", {})
    enabled = bool(rcfg.get("enabled", False)) or args.results_path is not None
    if enabled and results:
        out_path = args.results_path or rcfg.get("path")
        if not out_path:
            out_dir = rcfg.get("dir", "runs")
            name = rcfg.get("name", "central_run")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"{name}_{ts}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {
            "run_name": rcfg.get("name", "central_run"),
            "config_path": args.config,
            "results": results,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()

