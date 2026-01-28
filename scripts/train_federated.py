#!/usr/bin/env python3
"""
Federated Learning training script for F³ prototype.

Simulates federated learning with non-IID data partitioning and FedAvg aggregation.
"""
import argparse
import os
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from e

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data import generate_synthetic_transactions, TransactionDataset
from data.loader import _split_indices, _split_indices_time
from data.synthetic import load_transactions_jsonl
from federated.client import FLClient
from federated.server import FedAvgServer
from federated.partition import non_iid_split
from losses import RiskAwareBCELoss
from models import FraudModel
from training import metrics as M


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def get_device(device_str: str) -> torch.device:
    """Get torch device from string."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def create_client_datasets(
    transactions: List[Dict[str, Any]],
    user_to_txn_indices: Dict[int, List[int]],
    client_user_map: Dict[int, List[int]],
    seq_len: int,
    use_graph: bool,
) -> Dict[int, TransactionDataset]:
    """Create per-client datasets based on user assignments."""
    client_datasets = {}
    for client_id, user_list in client_user_map.items():
        # Get all transaction indices for users assigned to this client
        client_txn_indices = []
        for user_id in user_list:
            if user_id in user_to_txn_indices:
                client_txn_indices.extend(user_to_txn_indices[user_id])
        
        if not client_txn_indices:
            continue
        
        # Create subset of transactions for this client
        client_transactions = [transactions[i] for i in client_txn_indices]
        client_datasets[client_id] = TransactionDataset(
            client_transactions, seq_len=seq_len, use_graph=use_graph
        )
    return client_datasets


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    recall_at: float,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Evaluate model on a loader and compute basic metrics."""
    model.eval()
    total_loss = 0.0
    n = 0
    scores_all = []
    labels_all = []
    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(device).float()
            label = batch["label"].to(device).float()
            batch_dict = {"sequence": seq, "label": label}
            if "graph_data" in batch and batch["graph_data"] is not None:
                graph_data = batch["graph_data"][0] if isinstance(batch["graph_data"], list) else batch["graph_data"]
                batch_dict["graph_data"] = graph_data.to(device) if hasattr(graph_data, "to") else graph_data
            if "graph_txn_idx" in batch and batch["graph_txn_idx"] is not None:
                graph_txn_idx = batch["graph_txn_idx"]
                if isinstance(graph_txn_idx, list):
                    graph_txn_idx = torch.tensor(
                        [int(x) if not isinstance(x, torch.Tensor) else int(x.item()) for x in graph_txn_idx],
                        device=device,
                    )
                else:
                    graph_txn_idx = graph_txn_idx.to(device)
                batch_dict["graph_txn_idx"] = graph_txn_idx

            logits = model(batch_dict)
            if criterion is not None:
                loss = criterion(logits, label)
                total_loss += float(loss.item()) * label.shape[0]
                n += label.shape[0]
            scores = torch.sigmoid(logits.view(-1))
            scores_all.append(scores.cpu())
            labels_all.append(label.view(-1).cpu())

    if scores_all:
        y_scores = torch.cat(scores_all)
        y_true = torch.cat(labels_all)
        rec = M.recall_at_k(y_true, y_scores, k=recall_at)
        ap = M.pr_auc(y_true, y_scores) or 0.0
    else:
        rec = 0.0
        ap = 0.0
    avg_loss = total_loss / max(1, n) if criterion is not None else float("nan")
    return {"loss": avg_loss, "recall_at_k": rec, "ap": ap}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train F³ prototype with federated learning.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/centralized_example.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of federated rounds (overrides config).",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Number of clients (overrides config).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device(str(cfg.get("device", "auto")))

    # FL configuration
    fl_cfg = cfg.get("fl", {})
    num_clients = args.num_clients or int(fl_cfg.get("num_clients", 3))
    num_rounds = args.rounds or int(fl_cfg.get("rounds", 10))
    local_epochs = int(fl_cfg.get("local_epochs", 1))

    # Data configuration
    dcfg = cfg.get("data", {})
    dataset_path = dcfg.get("dataset_path")
    if dataset_path:
        transactions = load_transactions_jsonl(str(dataset_path))
    else:
        transactions = generate_synthetic_transactions(
            num_users=int(dcfg.get("num_users", 1000)),
            num_merchants=int(dcfg.get("num_merchants", 500)),
            num_devices=int(dcfg.get("num_devices", 800)),
            num_transactions=int(dcfg.get("num_transactions", 50_000)),
            fraud_ratio=float(dcfg.get("fraud_ratio", 0.02)),
            seed=int(cfg.get("seed", 42)),
            base_timestamp=dcfg.get("base_timestamp"),
            profile=dcfg.get("synthetic_profile"),
        )

    # Train/val/test split (global)
    split_strategy = str(dcfg.get("split_strategy", "random")).lower()
    val_frac = float(dcfg.get("val_split", 0.15))
    test_frac = float(dcfg.get("test_split", 0.15))
    if split_strategy == "time":
        train_idx, val_idx, test_idx = _split_indices_time(transactions, val_frac, test_frac)
    else:
        train_idx, val_idx, test_idx = _split_indices(len(transactions), val_frac, test_frac, seed=int(cfg.get("seed", 42)))
    train_transactions = [transactions[i] for i in train_idx]
    val_transactions = [transactions[i] for i in val_idx]
    test_transactions = [transactions[i] for i in test_idx]

    # Build user-to-transaction mapping
    from collections import defaultdict
    user_to_txn_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, rec in enumerate(train_transactions):
        user_to_txn_indices[int(rec["user_id"])].append(idx)

    # Non-IID partitioning
    user_ids = [int(rec["user_id"]) for rec in train_transactions]
    client_user_map = non_iid_split(
        user_ids,
        num_clients=num_clients,
        min_users_per_client=int(fl_cfg.get("min_users_per_client", 50)),
        seed=int(cfg.get("seed", 42)),
    )

    # Create client datasets
    mcfg = cfg.get("model", {})
    arch = str(mcfg.get("architecture", "fraud_model")).lower()
    if arch in ("baseline_gnn", "baseline", "gnn"):
        raise ValueError("baseline_gnn is not supported in federated training. Use scripts.train_central.")
    use_graph = bool(mcfg.get("use_graph", False)) or arch in ("baseline_gnn", "baseline", "gnn")
    seq_len = int(dcfg.get("seq_len", 10))
    client_datasets = create_client_datasets(
        train_transactions, user_to_txn_indices, client_user_map, seq_len, use_graph
    )

    # Create client dataloaders
    batch_size = int(dcfg.get("batch_size", 256))
    client_loaders = {
        cid: DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
        for cid, ds in client_datasets.items()
    }

    # Create val/test loaders
    val_dataset = TransactionDataset(val_transactions, seq_len=seq_len, use_graph=use_graph)
    test_dataset = TransactionDataset(test_transactions, seq_len=seq_len, use_graph=use_graph)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Create global model
    model = FraudModel(
        use_graph=use_graph,
        seq_cfg=mcfg.get("sequence_encoder", {}),
        fusion_cfg=mcfg.get("fusion_head", {}),
        graph_cfg=mcfg.get("graph_encoder", {}),
    ).to(device)

    # Loss function
    lcfg = cfg.get("loss", {})
    criterion = RiskAwareBCELoss(
        alpha=float(lcfg.get("alpha", 0.95)),
        lambda_risk=float(lcfg.get("lambda_risk", 1.0)),
        weighting=str(lcfg.get("weighting", "var")),
    )

    # Server
    weighted_agg = fl_cfg.get("weighted_aggregation", True)
    server = FedAvgServer(weighted=weighted_agg)

    # Training loop
    ocfg = cfg.get("optim", {})
    lr = float(ocfg.get("lr", 1e-3))
    recall_at = float(cfg.get("metrics", {}).get("recall_at", 0.03))

    print(f"Starting federated training with {num_clients} clients for {num_rounds} rounds")
    print(f"Local epochs per round: {local_epochs}")

    for round_num in range(1, num_rounds + 1):
        # Client updates
        client_models = []
        client_losses = []

        for client_id, loader in client_loaders.items():
            # Create client model (copy of global)
            client_model = FraudModel(
                use_graph=use_graph,
                seq_cfg=mcfg.get("sequence_encoder", {}),
                fusion_cfg=mcfg.get("fusion_head", {}),
                graph_cfg=mcfg.get("graph_encoder", {}),
            ).to(device)
            client_model.load_state_dict(model.state_dict())

            # Local training
            client = FLClient(client_id)
            client_result = client.local_update(
                model=client_model,
                dataloader=loader,
                criterion=criterion,
                lr=lr,
                epochs=local_epochs,
                device=device,
            )

            client_models.append(client_model)
            client_losses.append(client_result["train_loss"])

        # Server aggregation
        if weighted_agg:
            client_data_sizes = [len(client_datasets.get(cid, [])) for cid in client_loaders.keys()]
            client_weights = server.get_client_weights(client_data_sizes)
        else:
            client_weights = None
        server.aggregate(model, client_models, client_weights=client_weights)

        # Global validation on held-out set
        val_metrics = evaluate_model(model, val_loader, device, recall_at, criterion=criterion)
        avg_client_loss = sum(client_losses) / len(client_losses) if client_losses else 0.0
        print(
            f"[Round {round_num:02d}] avg_client_loss={avg_client_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_recall@{recall_at:.2%}={val_metrics['recall_at_k']:.4f} "
            f"val_ap={val_metrics['ap']:.4f}"
        )

    test_metrics = evaluate_model(model, test_loader, device, recall_at, criterion=criterion)
    print(
        f"[Test] loss={test_metrics['loss']:.6f} "
        f"recall@{recall_at:.2%}={test_metrics['recall_at_k']:.4f} "
        f"ap={test_metrics['ap']:.4f}"
    )
    print("Federated training completed!")


if __name__ == "__main__":
    main()
