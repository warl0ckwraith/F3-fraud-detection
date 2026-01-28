from typing import Optional, Tuple

import torch
from torch import nn


class GRUSequenceEncoder(nn.Module):
    """
    GRU-based encoder for fixed-length sequences of transaction features.

    Expected input: Tensor of shape (batch, seq_len, input_size)
    Output: Tensor of shape (batch, hidden_out) where hidden_out depends on
            bidirectionality.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )

        self.output_size = self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) sequence batch
        Returns:
            (B, output_size) sequence embedding (last time step)
        """
        out, h_n = self.gru(x)  # out: (B, T, D), h_n: (num_layers * dir, B, H)
        # Use last time step output from GRU (already includes both directions if enabled)
        last = out[:, -1, :]  # (B, D)
        return last


