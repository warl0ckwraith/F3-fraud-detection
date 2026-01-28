from typing import Iterable, List, Optional

import torch
from torch import nn


class FusionHead(nn.Module):
    """
    Simple fusion head that concatenates available embeddings and predicts fraud logit.

    If graph embedding is None, operates as a sequence-only MLP head.
    """

    def __init__(
        self,
        seq_dim: int,
        graph_dim: Optional[int] = None,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_dim = int(seq_dim)
        self.graph_dim = int(graph_dim) if graph_dim is not None else 0
        self.input_dim = self.seq_dim + self.graph_dim

        dims: List[int] = [self.input_dim] + [int(h) for h in hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.out = nn.Linear(dims[-1], 1)

    def forward(self, seq_emb: torch.Tensor, graph_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            seq_emb: (B, D_s)
            graph_emb: (B, D_g) or None
        Returns:
            logits: (B, 1)
        """
        if graph_emb is None:
            x = seq_emb
        else:
            x = torch.cat([seq_emb, graph_emb], dim=-1)
        h = self.mlp(x)
        logits = self.out(h)
        return logits


