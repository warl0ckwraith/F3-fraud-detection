"""
Models package.

Includes:
- FAHGTT: Temporal heterogeneous graph encoder
- GRUSequenceEncoder: Sequence encoder for user/transaction histories
- FusionHead: Fusion/classification head combining graph + sequence embeddings
- FraudModel: End-to-end model wrapper composing encoders and fusion head
- Baseline models: GCN, GAT, GraphSAGE for comparison
"""

from .sequence_encoder import GRUSequenceEncoder
from .fusion_head import FusionHead
from .model import FraudModel

try:
    from .fahgt_t import FAHGTT
    from .baselines import (
        GCNBaseline,
        GATBaseline,
        GraphSAGEBaseline,
        HeteroGNNBaseline,
        BaselineGraphModel,
    )
    _has_pyg = True
except Exception:
    _has_pyg = False

__all__ = [
    "GRUSequenceEncoder",
    "FusionHead",
    "FraudModel",
]

if _has_pyg:
    __all__.extend(
        [
            "FAHGTT",
            "GCNBaseline",
            "GATBaseline",
            "GraphSAGEBaseline",
            "HeteroGNNBaseline",
            "BaselineGraphModel",
        ]
    )
