from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import HeteroConv, GATConv, SAGEConv
    from torch_geometric.data import HeteroData
except ImportError:
    raise ImportError("PyTorch Geometric is required for FAHGT-T. Install with: pip install torch-geometric")


class TemporalEncoding(nn.Module):
    """
    Temporal encoding module that converts timestamps to learnable embeddings.
    Uses sinusoidal positional encoding with learnable scaling.
    """

    def __init__(self, embed_dim: int, max_period: float = 86400.0 * 30) -> None:
        """
        Args:
            embed_dim: Embedding dimension (should be even)
            max_period: Maximum time period in seconds (default: 30 days)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        # Learnable scaling factor
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (E,) tensor of timestamps (can be relative or absolute)
        Returns:
            (E, embed_dim) temporal embeddings
        """
        # Normalize timestamps to [0, 1] range
        if timestamps.numel() == 0:
            return torch.zeros(0, self.embed_dim, device=timestamps.device, dtype=torch.float32)

        # Use relative time encoding (normalize by max period)
        t_norm = (timestamps.float() / self.max_period) * self.scale

        # Sinusoidal encoding
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, device=timestamps.device, dtype=torch.float32)
                            * -(torch.log(torch.tensor(10000.0)) / self.embed_dim))
        pe = torch.zeros(timestamps.shape[0], self.embed_dim, device=timestamps.device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(t_norm.unsqueeze(-1) * div_term)
        pe[:, 1::2] = torch.cos(t_norm.unsqueeze(-1) * div_term)
        return pe


class HeteroTemporalAttention(nn.Module):
    """
    Temporal-aware attention layer for heterogeneous graphs.
    Combines graph structure with temporal edge information.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_types: list,
        heads: int = 4,
        temporal_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            edge_types: List of edge type tuples, e.g., [('user', 'initiates', 'transaction'), ...]
            heads: Number of attention heads
            temporal_dim: Dimension for temporal embeddings
            dropout: Dropout rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_types = edge_types
        self.temporal_dim = temporal_dim

        # Temporal encoding for edges
        self.temporal_encoder = TemporalEncoding(temporal_dim)

        # Per-edge-type attention layers
        self.attentions = nn.ModuleDict()
        for edge_type in edge_types:
            # Use GAT for attention mechanism
            self.attentions[str(edge_type)] = GATConv(
                in_channels,
                out_channels // heads,
                heads=heads,
                dropout=dropout,
                edge_dim=temporal_dim,  # Temporal features as edge attributes
                concat=True,
            )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        timestamp_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_dict: Node features by node type
            edge_index_dict: Edge indices by edge type
            edge_attr_dict: Optional edge attributes
            timestamp_dict: Timestamps by edge type
        Returns:
            Updated node features
        """
        out_dict = {}
        edge_attr_dict = edge_attr_dict or {}
        timestamp_dict = timestamp_dict or {}

        for edge_type in self.edge_types:
            src_type, rel, dst_type = edge_type
            edge_key = str(edge_type)

            if edge_key not in edge_index_dict:
                continue

            edge_index = edge_index_dict[edge_key]
            if edge_index.shape[1] == 0:  # No edges
                continue

            # Get temporal embeddings for edges
            if edge_key in timestamp_dict:
                timestamps = timestamp_dict[edge_key]
                temporal_emb = self.temporal_encoder(timestamps)
            else:
                # Fallback: zero temporal embedding
                temporal_emb = torch.zeros(edge_index.shape[1], self.temporal_dim, device=edge_index.device)

            # Combine with existing edge attributes if any
            if edge_key in edge_attr_dict:
                edge_attr = torch.cat([edge_attr_dict[edge_key], temporal_emb], dim=-1)
            else:
                edge_attr = temporal_emb

            # Apply attention
            if src_type not in x_dict or dst_type not in x_dict:
                continue

            x_src = x_dict[src_type]
            x_dst = x_dict[dst_type]

            # GAT expects edge_dim to match, so we need to ensure compatibility
            # For simplicity, use temporal_emb directly
            try:
                updated = self.attentions[edge_key](
                    (x_src, x_dst),
                    edge_index,
                    edge_attr=temporal_emb if temporal_emb.shape[-1] <= self.temporal_dim else temporal_emb[:, :self.temporal_dim]
                )
            except Exception:
                # Fallback: use without edge attributes if dimension mismatch
                updated = self.attentions[edge_key]((x_src, x_dst), edge_index)

            # Aggregate updates (sum over incoming edges)
            if dst_type not in out_dict:
                out_dict[dst_type] = torch.zeros_like(x_dst)
            out_dict[dst_type] = out_dict[dst_type] + updated

        # Apply layer norm and residual connection where possible
        for node_type in out_dict:
            if node_type in x_dict:
                out_dict[node_type] = self.layer_norm(out_dict[node_type] + x_dict[node_type])
            else:
                out_dict[node_type] = self.layer_norm(out_dict[node_type])

        # Preserve node types that weren't updated
        for node_type in x_dict:
            if node_type not in out_dict:
                out_dict[node_type] = x_dict[node_type]

        return out_dict


class FAHGTT(nn.Module):
    """
    FAHGT-T: Temporal Heterogeneous Graph Transformer encoder.

    Implements a minimal but functional temporal heterogeneous graph encoder
    that processes multi-typed nodes and edges with temporal awareness.

    Architecture:
    - Node feature initialization (learnable embeddings for nodes without features)
    - Multi-layer temporal heterogeneous attention
    - Final projection to output dimension

    Args:
        out_dim: Output embedding dimension for target nodes
        hidden_dim: Hidden dimension for graph layers (default: 64)
        num_layers: Number of graph attention layers (default: 2)
        heads: Number of attention heads per layer (default: 4)
        temporal_dim: Dimension for temporal edge embeddings (default: 16)
        dropout: Dropout rate (default: 0.1)
        time_attr: Name of edge attribute containing timestamps (default: "timestamp")
        node_types: List of node types in the graph (default: ['user', 'transaction', 'merchant', 'device'])
        edge_types: List of edge type tuples (default: common fraud detection schema)
    """

    def __init__(
        self,
        out_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        temporal_dim: int = 16,
        dropout: float = 0.1,
        time_attr: str = "timestamp",
        node_types: Optional[list] = None,
        edge_types: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.time_attr = str(time_attr)

        # Default node and edge types for fraud detection
        self.node_types = node_types or ["user", "transaction", "merchant", "device"]
        self.edge_types = edge_types or [
            ("user", "initiates", "transaction"),
            ("transaction", "at", "merchant"),
            ("transaction", "from", "device"),
        ]

        self.embed_dim = 16

        # Initialize node embeddings (for nodes without features)
        self.node_embeddings = nn.ModuleDict()
        for node_type in self.node_types:
            # Use small embedding dimension, will be projected
            self.node_embeddings[node_type] = nn.Embedding(10000, self.embed_dim)  # Max 10k nodes per type

        # Projection from node embeddings to hidden_dim
        self.embed_proj = nn.ModuleDict()
        for node_type in self.node_types:
            self.embed_proj[node_type] = nn.Linear(self.embed_dim, hidden_dim)

        # Projection for nodes that already have features (initialized from data)
        self.feature_proj = nn.ModuleDict()

        # Graph attention layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i > 0 else hidden_dim
            self.layers.append(
                HeteroTemporalAttention(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    edge_types=self.edge_types,
                    heads=heads,
                    temporal_dim=temporal_dim,
                    dropout=dropout,
                )
            )

        # Final projection to output dimension
        self.output_proj = nn.ModuleDict()
        for node_type in self.node_types:
            self.output_proj[node_type] = nn.Linear(hidden_dim, out_dim)

        self.dropout = nn.Dropout(dropout)

    def _get_node_features(self, hetero_data: HeteroData) -> Dict[str, torch.Tensor]:
        """Extract or initialize node features for all node types."""
        x_dict = {}
        # Get device from model parameters
        device = next(self.parameters()).device
        
        for node_type in self.node_types:
            if hasattr(hetero_data[node_type], "x") and hetero_data[node_type].x is not None:
                # Use provided features
                x = hetero_data[node_type].x.to(device)
                proj = self.feature_proj[node_type] if node_type in self.feature_proj else None
                if proj is None:
                    if x.shape[-1] != self.hidden_dim:
                        raise RuntimeError(
                            f"Feature projector not initialized for node type '{node_type}' "
                            f"(in_dim={x.shape[-1]}). Call prepare_for_training() first."
                        )
                    x_dict[node_type] = x
                else:
                    x_dict[node_type] = proj(x)
            else:
                # Initialize with learnable embeddings
                num_nodes = hetero_data[node_type].num_nodes
                node_ids = torch.arange(num_nodes, device=device)
                emb = self.node_embeddings[node_type](node_ids)
                x_dict[node_type] = self.embed_proj[node_type](emb)
        return x_dict

    def ensure_node_embeddings(self, hetero_data: HeteroData) -> None:
        """Ensure embedding tables are large enough for all node types."""
        for node_type in self.node_types:
            if node_type not in hetero_data.node_types:
                continue
            num_nodes = int(hetero_data[node_type].num_nodes)
            if node_type not in self.node_embeddings:
                continue
            emb = self.node_embeddings[node_type]
            if num_nodes <= emb.num_embeddings:
                continue

            # Expand embedding table to fit all nodes.
            device = emb.weight.device
            new_size = num_nodes
            new_emb = nn.Embedding(new_size, emb.embedding_dim).to(device)
            with torch.no_grad():
                new_emb.weight[: emb.num_embeddings].copy_(emb.weight)
            self.node_embeddings[node_type] = new_emb

    def ensure_feature_projectors(self, hetero_data: HeteroData) -> None:
        """Initialize feature projectors for node types with explicit features."""
        device = next(self.parameters()).device
        for node_type in self.node_types:
            if node_type not in hetero_data.node_types:
                continue
            node_data = hetero_data[node_type]
            if not hasattr(node_data, "x") or node_data.x is None:
                continue
            in_dim = int(node_data.x.shape[-1])
            proj = self.feature_proj[node_type] if node_type in self.feature_proj else None
            if proj is None or proj.in_features != in_dim:
                self.feature_proj[node_type] = nn.Linear(in_dim, self.hidden_dim).to(device)

    def prepare_for_training(self, hetero_data: HeteroData) -> None:
        """Prepare embeddings and projectors that depend on data shapes."""
        self.ensure_node_embeddings(hetero_data)
        self.ensure_feature_projectors(hetero_data)

    def _get_edge_data(self, hetero_data: HeteroData) -> Tuple[Dict, Dict, Dict]:
        """Extract edge indices, attributes, and timestamps."""
        edge_index_dict = {}
        edge_attr_dict = {}
        timestamp_dict = {}

        for edge_type in self.edge_types:
            edge_key = str(edge_type)
            # Check if edge type exists in hetero_data
            if edge_type in hetero_data.edge_types:
                edge_data = hetero_data[edge_type]
                edge_index_dict[edge_key] = edge_data.edge_index

                # Extract timestamps
                if hasattr(edge_data, self.time_attr):
                    timestamp_dict[edge_key] = getattr(edge_data, self.time_attr)
                elif hasattr(edge_data, "timestamp"):
                    timestamp_dict[edge_key] = edge_data.timestamp

                # Extract other edge attributes
                if hasattr(edge_data, "edge_attr") and edge_data.edge_attr is not None:
                    edge_attr_dict[edge_key] = edge_data.edge_attr

        return edge_index_dict, edge_attr_dict, timestamp_dict

    def forward(
        self,
        hetero_data: HeteroData,
        node_type: str = "transaction",
        node_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Produce embeddings for nodes of type `node_type`.

        Args:
            hetero_data: torch_geometric.data.HeteroData instance
            node_type: Node type to embed (e.g., 'user', 'transaction')
            node_indices: Optional indices of nodes to select; if None, returns
                          embeddings for all nodes of that type
        Returns:
            Tensor of shape (N, out_dim) for selected nodes
        """
        # Get node features
        x_dict = self._get_node_features(hetero_data)

        # Get edge data
        edge_index_dict, edge_attr_dict, timestamp_dict = self._get_edge_data(hetero_data)

        # Apply graph attention layers
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict, timestamp_dict)
            # Apply dropout and activation
            for node_type_key in x_dict:
                x_dict[node_type_key] = self.dropout(F.relu(x_dict[node_type_key]))

        # Project to output dimension
        if node_type in x_dict:
            output = self.output_proj[node_type](x_dict[node_type])
        else:
            # Fallback: return zeros if node type not found
            num_nodes = hetero_data[node_type].num_nodes if hasattr(hetero_data, node_type) else 0
            device = next(self.parameters()).device
            output = torch.zeros(num_nodes, self.out_dim, device=device)

        # Select specific indices if requested
        if node_indices is not None:
            output = output.index_select(0, node_indices.to(output.device))

        return output
