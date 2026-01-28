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

from data.synthetic import generate_synthetic_transactions, save_transactions_jsonl


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and save synthetic dataset (JSONL).")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument("--out", type=str, default="data/synthetic/mvp_v1.jsonl", help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_users", type=int, default=300)
    parser.add_argument("--num_merchants", type=int, default=150)
    parser.add_argument("--num_devices", type=int, default=200)
    parser.add_argument("--num_transactions", type=int, default=8000)
    parser.add_argument("--fraud_ratio", type=float, default=0.03)
    parser.add_argument(
        "--base_timestamp", type=int, default=1700000000, help="Fixed timestamp for determinism."
    )
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(args.config)

    dcfg = cfg.get("data", {})
    seed = int(dcfg.get("seed", args.seed))
    num_users = int(dcfg.get("num_users", args.num_users))
    num_merchants = int(dcfg.get("num_merchants", args.num_merchants))
    num_devices = int(dcfg.get("num_devices", args.num_devices))
    num_transactions = int(dcfg.get("num_transactions", args.num_transactions))
    fraud_ratio = float(dcfg.get("fraud_ratio", args.fraud_ratio))
    base_timestamp = dcfg.get("base_timestamp", args.base_timestamp)
    profile = dcfg.get("synthetic_profile")

    transactions = generate_synthetic_transactions(
        num_users=num_users,
        num_merchants=num_merchants,
        num_devices=num_devices,
        num_transactions=num_transactions,
        fraud_ratio=fraud_ratio,
        seed=seed,
        base_timestamp=base_timestamp,
        profile=profile,
    )
    save_transactions_jsonl(transactions, args.out)

    meta = {
        "out": args.out,
        "seed": seed,
        "num_users": num_users,
        "num_merchants": num_merchants,
        "num_devices": num_devices,
        "num_transactions": num_transactions,
        "fraud_ratio": fraud_ratio,
        "base_timestamp": base_timestamp,
        "synthetic_profile": profile,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved dataset to {args.out}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
