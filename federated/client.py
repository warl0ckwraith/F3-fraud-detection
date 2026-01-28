from typing import Any, Dict, Optional

import torch
from torch import nn, optim


class FLClient:
    """
    Federated Learning client with local training.

    Based on patterns from federated_fraud_detection.
    Supports local training with configurable optimizer and epochs.
    """

    def __init__(self, client_id: int) -> None:
        """
        Args:
            client_id: Unique client identifier
        """
        self.client_id = int(client_id)

    def local_update(
        self,
        model: nn.Module,
        dataloader,
        criterion: nn.Module,
        lr: float = 1e-3,
        epochs: int = 1,
        device: torch.device = torch.device("cpu"),
        optimizer_type: str = "Adam",
        weight_decay: float = 1e-5,
        max_grad_norm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform local training update.

        Args:
            model: Model to train
            dataloader: Local data loader
            criterion: Loss function
            lr: Learning rate
            epochs: Number of local epochs
            device: Training device
            optimizer_type: Optimizer type ("Adam" or "SGD")
            weight_decay: Weight decay for optimizer
            max_grad_norm: Optional gradient clipping

        Returns:
            Dictionary with training statistics
        """
        model.train()

        # Ensure graph embeddings are sized before optimizer
        graph_encoder = getattr(model, "graph_encoder", None)
        if graph_encoder is not None:
            graph_data = getattr(dataloader.dataset, "_graph_data", None)
            if isinstance(graph_data, list) and graph_data:
                graph_data = graph_data[0]
            if graph_data is not None:
                if hasattr(graph_encoder, "prepare_for_training"):
                    graph_encoder.prepare_for_training(graph_data)
                else:
                    graph_encoder.ensure_node_embeddings(graph_data)

        # Setup optimizer
        if optimizer_type.lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        total_loss = 0.0
        n_samples = 0
        n_batches = 0

        for epoch in range(epochs):
            for batch in dataloader:
                # Move batch to device
                seq = batch["sequence"].to(device).float()
                y = batch["label"].to(device).float()

                # Prepare batch dict
                batch_dict = {"sequence": seq, "label": y}

                # Handle graph data if present
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

                # Forward pass
                logits = model(batch_dict)
                loss = criterion(logits, y)

                # Backward pass
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Gradient clipping
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()

                # Statistics
                batch_size = y.shape[0]
                total_loss += float(loss.item()) * batch_size
                n_samples += batch_size
                n_batches += 1

        avg_loss = total_loss / max(1, n_samples)

        return {
            "client_id": self.client_id,
            "train_loss": avg_loss,
            "n_samples": n_samples,
            "n_batches": n_batches,
        }
