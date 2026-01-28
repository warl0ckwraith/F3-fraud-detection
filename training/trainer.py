from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.loader import get_dataloaders
from losses import RiskAwareBCELoss
from models import FraudModel
from . import metrics as M


def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CentralTrainer:
    """
    Basic centralized trainer for the F³ prototype.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        seed_everything(int(cfg.get("seed", 42)))
        self.device = _get_device(str(cfg.get("device", "auto")))

        # Data
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(cfg)

        # Model
        mcfg = cfg.get("model", {})
        arch = str(mcfg.get("architecture", "fraud_model")).lower()
        if arch in ("baseline_gnn", "baseline", "gnn"):
            from models import BaselineGraphModel

            bcfg = mcfg.get("baseline", {})
            self.model = BaselineGraphModel(
                gnn_type=str(bcfg.get("gnn_type", "GCN")),
                input_dim=int(bcfg.get("input_dim", 3)),
                hidden_dim=int(bcfg.get("hidden_dim", 64)),
                num_layers=int(bcfg.get("num_layers", 2)),
                output_dim=int(bcfg.get("output_dim", 64)),
                heads=int(bcfg.get("heads", 4)),
                dropout=float(bcfg.get("dropout", 0.1)),
                classifier_hidden_dims=bcfg.get("classifier_hidden_dims", (64,)),
                target_node_type=str(bcfg.get("target_node_type", "transaction")),
            ).to(self.device)
        else:
            self.model = FraudModel(
                use_graph=bool(mcfg.get("use_graph", False)),
                seq_cfg=mcfg.get("sequence_encoder", {}),
                fusion_cfg=mcfg.get("fusion_head", {}),
                graph_cfg=mcfg.get("graph_encoder", {}),
            ).to(self.device)

        # Initialize graph embeddings/projections before optimizer, if needed
        graph_data = getattr(self.train_loader.dataset, "_graph_data", None)
        if isinstance(graph_data, list) and graph_data:
            graph_data = graph_data[0]
        if graph_data is not None and hasattr(self.model, "prepare_for_training"):
            self.model.prepare_for_training(graph_data)

        # Loss
        lcfg = cfg.get("loss", {})
        self.criterion = RiskAwareBCELoss(
            alpha=float(lcfg.get("alpha", 0.95)),
            lambda_risk=float(lcfg.get("lambda_risk", 1.0)),
            weighting=str(lcfg.get("weighting", "var")),
        )

        # Optim
        ocfg = cfg.get("optim", {})
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(ocfg.get("lr", 1e-3)),
            weight_decay=float(ocfg.get("weight_decay", 1e-5)),
        )
        self.epochs = int(ocfg.get("epochs", 3))

        # Learning rate scheduler
        self.scheduler: Optional[ReduceLROnPlateau] = None
        if ocfg.get("lr_schedule", {}).get("enabled", False):
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=float(ocfg.get("lr_schedule", {}).get("factor", 0.5)),
                patience=int(ocfg.get("lr_schedule", {}).get("patience", 3)),
                verbose=True,
            )

        # Early stopping
        self.early_stopping: Optional[Dict[str, Any]] = None
        if ocfg.get("early_stopping", {}).get("enabled", False):
            self.early_stopping = {
                "patience": int(ocfg.get("early_stopping", {}).get("patience", 5)),
                "min_delta": float(ocfg.get("early_stopping", {}).get("min_delta", 1e-6)),
                "best_val": float("inf"),
                "patience_counter": 0,
            }

        # Metrics
        self.recall_at = float(cfg.get("metrics", {}).get("recall_at", 0.03))

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        seq = batch["sequence"].to(self.device).float()
        label = batch["label"].to(self.device).float()
        moved = {"sequence": seq, "label": label}
        
        # Handle graph data if present
        if "graph_data" in batch and batch["graph_data"] is not None:
            # Graph data is the same for all items in batch (shared graph)
            # Take the first one and move to device
            graph_data = batch["graph_data"][0] if isinstance(batch["graph_data"], list) else batch["graph_data"]
            moved["graph_data"] = graph_data.to(self.device) if hasattr(graph_data, "to") else graph_data
        
        if "graph_txn_idx" in batch and batch["graph_txn_idx"] is not None:
            # Stack transaction indices from batch
            graph_txn_idx = batch["graph_txn_idx"]
            if isinstance(graph_txn_idx, list):
                # Handle list of tensors/scalars
                graph_txn_idx = torch.tensor([int(x) if not isinstance(x, torch.Tensor) else int(x.item()) 
                                             for x in graph_txn_idx], device=self.device)
            else:
                graph_txn_idx = graph_txn_idx.to(self.device)
            moved["graph_txn_idx"] = graph_txn_idx
        
        return moved

    def train(self) -> Dict[str, Any]:
        best_val = float("inf")
        best_epoch = -1
        last_val_loss = float("inf")
        last_val_metrics: Dict[str, float] = {}
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running = 0.0
            n = 0
            for batch in self.train_loader:
                b = self._move_batch(batch)
                logits = self.model(b)  # (B, 1)
                loss = self.criterion(logits, b["label"])
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                running += float(loss.item()) * b["label"].shape[0]
                n += b["label"].shape[0]
            train_loss = running / max(1, n)

            val_loss, val_metrics = self.evaluate(self.val_loader)
            last_val_loss = val_loss
            last_val_metrics = val_metrics
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"[Epoch {epoch:02d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                      f"val_recall@{self.recall_at:.2%}={val_metrics.get('recall_at_k', float('nan')):.4f} "
                      f"val_ap={val_metrics.get('ap', float('nan')):.4f} lr={current_lr:.6f}")
            else:
                print(f"[Epoch {epoch:02d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                      f"val_recall@{self.recall_at:.2%}={val_metrics.get('recall_at_k', float('nan')):.4f} "
                      f"val_ap={val_metrics.get('ap', float('nan')):.4f}")

            # Early stopping check
            if self.early_stopping is not None:
                improvement = self.early_stopping["best_val"] - val_loss
                if improvement > self.early_stopping["min_delta"]:
                    self.early_stopping["best_val"] = val_loss
                    self.early_stopping["patience_counter"] = 0
                else:
                    self.early_stopping["patience_counter"] += 1
                    if self.early_stopping["patience_counter"] >= self.early_stopping["patience"]:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                # TODO: Save best model to disk if needed

        test_loss, test_metrics = self.evaluate(self.test_loader)
        print(f"[Test] loss={test_loss:.6f} recall@{self.recall_at:.2%}={test_metrics.get('recall_at_k', float('nan')):.4f} "
              f"ap={test_metrics.get('ap', float('nan')):.4f} cost={test_metrics.get('cost', float('nan')):.4f}")
        return {
            "best_val_loss": float(best_val),
            "best_epoch": int(best_epoch),
            "last_val_loss": float(last_val_loss),
            "last_val_metrics": last_val_metrics,
            "test_loss": float(test_loss),
            "test_metrics": test_metrics,
        }

    @torch.no_grad()
    def evaluate(self, loader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total = 0.0
        n = 0
        all_scores = []
        all_labels = []
        for batch in loader:
            b = self._move_batch(batch)
            logits = self.model(b)
            loss = self.criterion(logits, b["label"])
            total += float(loss.item()) * b["label"].shape[0]
            n += b["label"].shape[0]
            scores = torch.sigmoid(logits.view(-1))
            all_scores.append(scores.cpu())
            all_labels.append(b["label"].view(-1).cpu())
        mean_loss = total / max(1, n)
        y_scores = torch.cat(all_scores) if all_scores else torch.empty(0)
        y_true = torch.cat(all_labels) if all_labels else torch.empty(0)
        rec = M.recall_at_k(y_true, y_scores, k=self.recall_at) if y_true.numel() > 0 else 0.0
        ap = M.pr_auc(y_true, y_scores) if y_true.numel() > 0 else None
        if ap is None:
            ap = float("nan")
        cost = M.cost_metrics(y_true, y_scores).get("cost", float("nan")) if y_true.numel() > 0 else float("nan")
        return mean_loss, {"recall_at_k": rec, "ap": ap, "cost": cost}
