"""
Baseline GNN models for fraud detection comparison.

Based on patterns from arxyzan/fraud-detection-gnn and DGFraud.
Implements GCN, GAT, and GraphSAGE as baselines.
"""
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
    from torch_geometric.data import HeteroData, Data
except ImportError:
    raise ImportError("PyTorch Geometric is required for baseline models. Install with: pip install torch-geometric")


class GCNBaseline(nn.Module):
    """
    Graph Convolutional Network baseline for fraud detection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        else:
            self.convs.append(GCNConv(input_dim, output_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (N, input_dim)
            edge_index: Edge indices (2, E)
        Returns:
            Node embeddings (N, output_dim)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATBaseline(nn.Module):
    """
    Graph Attention Network baseline for fraud detection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True))
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout, concat=False))
        else:
            self.convs.append(GATConv(input_dim, output_dim, heads=1, dropout=dropout, concat=False))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (N, input_dim)
            edge_index: Edge indices (2, E)
        Returns:
            Node embeddings (N, output_dim)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphSAGEBaseline(nn.Module):
    """
    GraphSAGE baseline for fraud detection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim))
        else:
            self.convs.append(SAGEConv(input_dim, output_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (N, input_dim)
            edge_index: Edge indices (2, E)
        Returns:
            Node embeddings (N, output_dim)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class HeteroGNNBaseline(nn.Module):
    """
    Heterogeneous GNN baseline that converts hetero graph to homogeneous
    and applies standard GNN.

    This is a simple baseline for comparison with FAHGT-T.
    """

    def __init__(
        self,
        gnn_type: str = "GCN",
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gnn_type = gnn_type
        self.input_dim = int(input_dim)

        if gnn_type == "GCN":
            self.gnn = GCNBaseline(input_dim, hidden_dim, num_layers, output_dim, dropout)
        elif gnn_type == "GAT":
            self.gnn = GATBaseline(input_dim, hidden_dim, num_layers, heads, output_dim, dropout)
        elif gnn_type == "GraphSAGE":
            self.gnn = GraphSAGEBaseline(input_dim, hidden_dim, num_layers, output_dim, dropout)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

    def hetero_to_homo(self, hetero_data: HeteroData, target_node_type: str = "transaction") -> Data:
        """
        Convert heterogeneous graph to homogeneous by focusing on target node type.

        Args:
            hetero_data: Heterogeneous graph
            target_node_type: Node type to extract (default: "transaction")

        Returns:
            Homogeneous PyG Data object
        """
        # Ensure all node types have features with consistent dimension
        hetero = hetero_data.clone()
        for node_type in hetero.node_types:
            node_data = hetero[node_type]
            if hasattr(node_data, "x") and node_data.x is not None:
                x = node_data.x.float()
                if x.shape[-1] < self.input_dim:
                    pad = torch.zeros(
                        x.shape[0], self.input_dim - x.shape[-1], dtype=x.dtype, device=x.device
                    )
                    x = torch.cat([x, pad], dim=-1)
                elif x.shape[-1] > self.input_dim:
                    x = x[:, : self.input_dim]
                node_data.x = x
            else:
                device = node_data.edge_index.device if hasattr(node_data, "edge_index") else None
                node_data.x = torch.zeros(
                    node_data.num_nodes, self.input_dim, dtype=torch.float32, device=device
                )

        # Use PyG's homogeneous conversion to properly reindex node IDs across types
        homo = hetero.to_homogeneous()
        return homo

    def forward(
        self,
        hetero_data: HeteroData,
        node_type: str = "transaction",
        node_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hetero_data: Heterogeneous graph
            node_type: Target node type
            node_indices: Optional indices to select

        Returns:
            Node embeddings (N, output_dim)
        """
        # Convert to homogeneous graph
        homo_data = self.hetero_to_homo(hetero_data, node_type)
        device = next(self.parameters()).device
        homo_data = homo_data.to(device)

        # Apply GNN
        embeddings = self.gnn(homo_data.x, homo_data.edge_index)

        # Select target node type indices if requested (or return all target nodes)
        if node_type in hetero_data.node_types:
            type_idx = hetero_data.node_types.index(node_type)
            txn_global = (homo_data.node_type == type_idx).nonzero(as_tuple=False).view(-1)
            if node_indices is not None:
                txn_global = txn_global.index_select(0, node_indices.to(device))
            embeddings = embeddings.index_select(0, txn_global)

        return embeddings


class BaselineGraphModel(nn.Module):
    """
    Graph-only baseline classifier using a homogeneous GNN encoder.
    """

    def __init__(
        self,
        gnn_type: str = "GCN",
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        classifier_hidden_dims: Iterable[int] = (64,),
        target_node_type: str = "transaction",
    ) -> None:
        super().__init__()
        self.encoder = HeteroGNNBaseline(
            gnn_type=gnn_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            heads=heads,
            dropout=dropout,
        )
        self.target_node_type = target_node_type
        dims = [output_dim] + [int(h) for h in classifier_hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.out = nn.Linear(dims[-1], 1)

    def prepare_for_training(self, graph_data: HeteroData) -> None:
        """No-op hook for compatibility with trainer initialization."""
        _ = graph_data

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        graph_data = batch.get("graph_data", None)
        if graph_data is None:
            raise ValueError("BaselineGraphModel requires 'graph_data' in batch.")
        graph_txn_idx = batch.get("graph_txn_idx", None)
        emb = self.encoder(graph_data, node_type=self.target_node_type, node_indices=graph_txn_idx)
        h = self.mlp(emb)
        return self.out(h)
