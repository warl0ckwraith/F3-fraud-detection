# Fraud Detection Prototype (F3)

This repo is a research prototype for fraud detection. It combines transaction sequence modeling with a temporal heterogeneous graph encoder and can run a federated learning simulation to test non-IID client data. It is meant for experiments, not production.

What is implemented so far:
- Sequence-only baseline training
- Graph + sequence model (FAHGT-T + GRU) with fusion
- Risk-aware loss options (VaR/ES)
- FedAvg-style federated training loop
- Deterministic synthetic data generator
- IEEE-CIS dataset loader (minimal config wiring)

Setup and first run:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.train_central --config configs/review1_seq.yaml
```
