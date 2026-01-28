from typing import Dict, Iterable, List, Optional

import torch
from torch import nn


class FedAvgServer:
    """
    FedAvg server with weighted aggregation support.

    Based on best practices from federated_fraud_detection and FL literature.
    Supports uniform and weighted averaging (by data volume).
    """

    def __init__(self, weighted: bool = True) -> None:
        """
        Args:
            weighted: If True, weight aggregation by client data volume
        """
        self.weighted = weighted

    @torch.no_grad()
    def aggregate(
        self,
        global_model: nn.Module,
        client_models: Iterable[nn.Module],
        client_weights: Optional[List[float]] = None,
    ) -> None:
        """
        In-place FedAvg: average client weights into the global model.

        Args:
            global_model: Global model to update
            client_models: Iterable of client models
            client_weights: Optional weights for each client (by data volume).
                          If None and weighted=True, uses uniform weights.
        """
        client_models = list(client_models)
        if not client_models:
            return

        # Compute weights
        if self.weighted and client_weights is not None:
            weights = torch.tensor(client_weights, dtype=torch.float32)
            weights = weights / weights.sum()  # Normalize
        else:
            # Uniform weights
            weights = torch.ones(len(client_models)) / len(client_models)

        # Aggregate
        global_state = global_model.state_dict()
        agg = {k: torch.zeros_like(v) for k, v in global_state.items()}

        for i, cm in enumerate(client_models):
            sd = cm.state_dict()
            w = weights[i].item()
            for k in agg.keys():
                agg[k] += sd[k] * w

        global_state.update(agg)
        global_model.load_state_dict(global_state, strict=True)

    def get_client_weights(self, client_data_sizes: List[int]) -> List[float]:
        """
        Compute aggregation weights based on data sizes.

        Args:
            client_data_sizes: List of data sizes per client

        Returns:
            Normalized weights
        """
        total = sum(client_data_sizes)
        return [size / total for size in client_data_sizes]


