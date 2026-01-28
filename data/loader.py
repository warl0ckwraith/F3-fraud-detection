from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset

from .synthetic import generate_synthetic_transactions, load_transactions_jsonl
from .graph import build_hetero_graph


class TransactionDataset(Dataset):
    """
    Centralized transaction dataset that yields sequence data and labels.

    For each transaction, we construct a fixed-length user-centric sequence of
    previous transactions (amount and time-gap features).

    Optionally builds a heterogeneous graph for graph-based models.
    """

    def __init__(
        self,
        transactions: List[Dict[str, Any]],
        seq_len: int = 10,
        use_graph: bool = False,
    ) -> None:
        self.transactions = transactions
        self.seq_len = seq_len
        self.use_graph = use_graph

        # Precompute per-user chronological transactions
        self._sequences: List[torch.Tensor] = [torch.zeros(seq_len, 2, dtype=torch.float32) for _ in transactions]
        self._labels: List[torch.Tensor] = [torch.tensor(0.0) for _ in transactions]
        self._txn_ids: List[int] = [int(rec["txn_id"]) for rec in transactions]

        # Build index by user, sorted by time
        user_to_idxs: Dict[int, List[int]] = defaultdict(list)
        for idx, rec in enumerate(transactions):
            user_to_idxs[int(rec["user_id"])].append(idx)
        for user, idxs in user_to_idxs.items():
            idxs.sort(key=lambda i: transactions[i]["timestamp"])

        # Populate rolling sequences per user
        for user, idxs in user_to_idxs.items():
            history: List[Tuple[float, int]] = []  # (amount_log1p, timestamp)
            for pos, i in enumerate(idxs):
                rec = transactions[i]
                amount_log = float(torch.log1p(torch.tensor(rec["amount"])).item())
                ts = int(rec["timestamp"])
                # Build features from up to last seq_len events
                feats = torch.zeros(self.seq_len, 2, dtype=torch.float32)
                tail = history[-self.seq_len :]
                # Align from the end (most recent at -1)
                start_row = max(0, self.seq_len - len(tail))
                for row, (amt_log_prev, ts_prev) in enumerate(tail, start=start_row):
                    delta_hours = max(0.0, float(ts - ts_prev) / 3600.0)
                    feats[row, 0] = float(amt_log_prev)
                    feats[row, 1] = float(delta_hours)
                self._sequences[i] = feats
                self._labels[i] = torch.tensor(float(rec["label"]), dtype=torch.float32)
                # Append current to history for next rounds
                history.append((amount_log, ts))

        # Build graph if requested (once per dataset)
        self._graph_data: Optional[Any] = None
        if self.use_graph:
            try:
                self._graph_data = build_hetero_graph(transactions, normalize_time=True)
                # Build mapping from txn_id to graph transaction index
                txn_id_to_graph_idx = {int(rec["txn_id"]): i for i, rec in enumerate(transactions)}
                self._txn_id_to_graph_idx = txn_id_to_graph_idx
            except ImportError:
                # PyG not available, graph will be None
                self._graph_data = None

    def __len__(self) -> int:
        return len(self.transactions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.transactions[idx]
        result = {
            "txn_id": self._txn_ids[idx],
            "user_id": int(rec["user_id"]),
            "sequence": self._sequences[idx],  # (seq_len, 2)
            "label": self._labels[idx],  # scalar float tensor
        }

        # Include graph data only when available
        if self.use_graph and self._graph_data is not None:
            result["graph_data"] = self._graph_data
            graph_idx = idx
            if hasattr(self, "_txn_id_to_graph_idx"):
                graph_idx = self._txn_id_to_graph_idx.get(self._txn_ids[idx], idx)
            result["graph_txn_idx"] = graph_idx  # Index in the graph's transaction nodes

        return result


def _split_indices(n: int, val_frac: float, test_frac: float, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(val_frac * n)
    n_test = int(test_frac * n)
    val_idx = perm[:n_val]
    test_idx = perm[n_val : n_val + n_test]
    train_idx = perm[n_val + n_test :]
    return train_idx, val_idx, test_idx


def _split_indices_time(
    transactions: List[Dict[str, Any]], val_frac: float, test_frac: float
) -> Tuple[List[int], List[int], List[int]]:
    idxs = list(range(len(transactions)))
    idxs.sort(key=lambda i: int(transactions[i].get("timestamp", 0)))
    n = len(idxs)
    n_val = int(val_frac * n)
    n_test = int(test_frac * n)
    n_train = max(0, n - n_val - n_test)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train : n_train + n_val]
    test_idx = idxs[n_train + n_val :]
    return train_idx, val_idx, test_idx


def get_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build centralized train/val/test dataloaders from synthetic data.

    Expected config schema:
    data:
      num_users: int
      num_merchants: int
      num_devices: int
      num_transactions: int
      fraud_ratio: float
      seq_len: int
      batch_size: int
      val_split: float
      test_split: float
      split_strategy: str ("random" or "time", optional)
      dataset_path: str (optional, if provided load from JSONL)
      base_timestamp: int (optional, fixed anchor timestamp for determinism)
      synthetic_profile: dict (optional, controls synthetic generator behavior)
      seed: int (optional)
    model:
      use_graph: bool (optional, default False)
    """
    dcfg = cfg.get("data", {})
    mcfg = cfg.get("model", {})
    seed = int(cfg.get("seed", 42))
    arch = str(mcfg.get("architecture", "fraud_model")).lower()
    use_graph = bool(mcfg.get("use_graph", False)) or arch in ("baseline_gnn", "baseline", "gnn")

    dataset_path = dcfg.get("dataset_path")
    if dataset_path:
        transactions = load_transactions_jsonl(str(dataset_path))
    else:
        transactions = generate_synthetic_transactions(
            num_users=int(dcfg.get("num_users", 1000)),
            num_merchants=int(dcfg.get("num_merchants", 500)),
            num_devices=int(dcfg.get("num_devices", 800)),
            num_transactions=int(dcfg.get("num_transactions", 50_000)),
            fraud_ratio=float(dcfg.get("fraud_ratio", 0.02)),
            seed=seed,
            base_timestamp=dcfg.get("base_timestamp"),
            profile=dcfg.get("synthetic_profile"),
        )
    seq_len = int(dcfg.get("seq_len", 10))

    split_strategy = str(dcfg.get("split_strategy", "random")).lower()
    if split_strategy == "time":
        train_idx, val_idx, test_idx = _split_indices_time(
            transactions=transactions,
            val_frac=float(dcfg.get("val_split", 0.15)),
            test_frac=float(dcfg.get("test_split", 0.15)),
        )
    else:
        train_idx, val_idx, test_idx = _split_indices(
            n=len(transactions),
            val_frac=float(dcfg.get("val_split", 0.15)),
            test_frac=float(dcfg.get("test_split", 0.15)),
            seed=seed,
        )
    train_transactions = [transactions[i] for i in train_idx]
    val_transactions = [transactions[i] for i in val_idx]
    test_transactions = [transactions[i] for i in test_idx]

    train_dataset = TransactionDataset(train_transactions, seq_len=seq_len, use_graph=use_graph)
    val_dataset = TransactionDataset(val_transactions, seq_len=seq_len, use_graph=use_graph)
    test_dataset = TransactionDataset(test_transactions, seq_len=seq_len, use_graph=use_graph)

    batch_size = int(dcfg.get("batch_size", 256))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader
