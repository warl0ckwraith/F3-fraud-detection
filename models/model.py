from typing import Any, Dict, Optional

import torch
from torch import nn

from .sequence_encoder import GRUSequenceEncoder
from .fusion_head import FusionHead


class FraudModel(nn.Module):
    """
    End-to-end fraud model that fuses sequence and (optional) graph embeddings.
    """

    def __init__(
        self,
        use_graph: bool,
        seq_cfg: Dict[str, Any],
        fusion_cfg: Dict[str, Any],
        graph_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.use_graph = bool(use_graph)
        # Sequence encoder
        self.seq_encoder = GRUSequenceEncoder(
            input_size=int(seq_cfg.get("input_size", 2)),
            hidden_size=int(seq_cfg.get("hidden_size", 64)),
            num_layers=int(seq_cfg.get("num_layers", 1)),
            bidirectional=bool(seq_cfg.get("bidirectional", False)),
            dropout=float(seq_cfg.get("dropout", 0.0)),
        )
        seq_out = self.seq_encoder.output_size

        # Optional graph encoder
        self.graph_encoder = None
        graph_out = None
        if self.use_graph:
            from .fahgt_t import FAHGTT
            gcfg = graph_cfg or {}
            graph_out = int(gcfg.get("out_dim", 64))
            self.graph_encoder = FAHGTT(
                out_dim=graph_out,
                hidden_dim=int(gcfg.get("hidden_dim", 64)),
                num_layers=int(gcfg.get("num_layers", 2)),
                heads=int(gcfg.get("heads", 4)),
                temporal_dim=int(gcfg.get("temporal_dim", 16)),
                dropout=float(gcfg.get("dropout", 0.1)),
                time_attr=str(gcfg.get("time_attr", "timestamp")),
            )

        # Fusion + classifier
        self.fusion = FusionHead(
            seq_dim=seq_out,
            graph_dim=graph_out,
            hidden_dims=fusion_cfg.get("hidden_dims", (128, 64)),
            dropout=float(fusion_cfg.get("dropout", 0.1)),
        )

    def prepare_for_training(self, graph_data: Any) -> None:
        """Initialize graph encoder parameters that depend on data shapes."""
        if self.use_graph and self.graph_encoder is not None and graph_data is not None:
            if hasattr(self.graph_encoder, "prepare_for_training"):
                self.graph_encoder.prepare_for_training(graph_data)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            batch: Dict with keys:
              - 'sequence': (B, T, F) float tensor
              - 'graph_data': Optional HeteroData instance (if use_graph=True)
              - 'graph_txn_idx': Optional tensor of transaction indices in graph (B,)
        Returns:
            logits: (B, 1)
        """
        seq = batch["sequence"]
        if seq.dim() == 2:  # (T, F) single sample
            seq = seq.unsqueeze(0)
        seq_emb = self.seq_encoder(seq)

        graph_emb = None
        if self.use_graph and self.graph_encoder is not None:
            graph_data = batch.get("graph_data", None)
            graph_txn_idx = batch.get("graph_txn_idx", None)

            if graph_data is not None:
                # Move graph to same device as model
                device = next(self.parameters()).device
                graph_data = graph_data.to(device)

                # Get transaction embeddings from graph
                if graph_txn_idx is not None:
                    # Select specific transaction nodes
                    graph_txn_idx = graph_txn_idx.to(device)
                    graph_emb = self.graph_encoder(graph_data, node_type="transaction", node_indices=graph_txn_idx)
                else:
                    # Get all transaction embeddings (will need to select later)
                    graph_emb = self.graph_encoder(graph_data, node_type="transaction")

                # Ensure graph_emb matches batch size
                if graph_emb.shape[0] != seq_emb.shape[0]:
                    # If we got all embeddings, we need to select the right ones
                    if graph_txn_idx is not None and graph_emb.shape[0] > seq_emb.shape[0]:
                        graph_emb = graph_emb[graph_txn_idx]

        logits = self.fusion(seq_emb, graph_emb)
        return logits
