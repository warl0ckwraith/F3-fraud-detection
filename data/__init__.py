"""
Data package.

This module exposes utilities for:
- Synthetic transaction generation
- Centralized dataset/dataloader construction
- Heterogeneous graph construction (PyG), optional
"""

from .synthetic import generate_synthetic_transactions, load_transactions_jsonl, save_transactions_jsonl
from .loader import TransactionDataset, get_dataloaders
from .graph import build_hetero_graph

__all__ = [
    "generate_synthetic_transactions",
    "load_transactions_jsonl",
    "save_transactions_jsonl",
    "TransactionDataset",
    "get_dataloaders",
    "build_hetero_graph",
]

