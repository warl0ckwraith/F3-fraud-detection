#!/usr/bin/env bash
set -euo pipefail

python -m scripts.train_central --config configs/mvp_cpu.yaml
