#!/usr/bin/env bash
set -euo pipefail

python -m scripts.train_central --config configs/baseline_seq.yaml
python -m scripts.train_central --config configs/baseline_graph.yaml
python -m scripts.results_table --inputs runs/*.json --out runs/results.md
