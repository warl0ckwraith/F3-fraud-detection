from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import torch


def build_hetero_graph(transactions: List[Dict[str, Any]], normalize_time: bool = True) -> Any:
    """
    Construct a Heterogeneous Graph from transaction records with node and edge features.

    Node types:
      - 'user': User nodes (no features initially, will use learnable embeddings)
      - 'merchant': Merchant nodes (no features initially)
      - 'device': Device nodes (no features initially)
      - 'transaction': Transaction nodes with features (amount, time-of-day, etc.)

    Edge types:
      - ('user', 'initiates', 'transaction'): User initiates transaction
      - ('transaction', 'at', 'merchant'): Transaction at merchant
      - ('transaction', 'from', 'device'): Transaction from device

    All edges include timestamps for temporal encoding.

    Args:
        transactions: List of transaction dictionaries with keys:
            - txn_id, user_id, merchant_id, device_id, timestamp, amount, label
        normalize_time: If True, normalize timestamps to relative time (0-based)

    Returns:
        torch_geometric.data.HeteroData instance
    """
    try:
        from torch_geometric.data import HeteroData  # type: ignore
    except Exception as e:
        raise ImportError("PyTorch Geometric is required to build the heterogeneous graph.") from e

    if not transactions:
        raise ValueError("Cannot build graph from empty transaction list")

    # Build ID remapping for compact indices
    users = sorted({int(r["user_id"]) for r in transactions})
    merchants = sorted({int(r["merchant_id"]) for r in transactions})
    devices = sorted({int(r["device_id"]) for r in transactions})
    txns = [int(r["txn_id"]) for r in transactions]

    uid_map = {u: i for i, u in enumerate(users)}
    mid_map = {m: i for i, m in enumerate(merchants)}
    did_map = {d: i for i, d in enumerate(devices)}
    tid_map = {t: i for i, t in enumerate(txns)}

    # Extract timestamps for normalization
    all_timestamps = [int(r["timestamp"]) for r in transactions]
    min_ts = min(all_timestamps) if all_timestamps else 0
    max_ts = max(all_timestamps) if all_timestamps else 1

    data = HeteroData()
    data["user"].num_nodes = len(users)
    data["merchant"].num_nodes = len(merchants)
    data["device"].num_nodes = len(devices)
    data["transaction"].num_nodes = len(txns)

    # Build transaction node features
    txn_features = []
    user_src, user_dst, user_ts = [], [], []
    tm_src, tm_dst, tm_ts = [], [], []
    td_src, td_dst, td_ts = [], [], []

    # Compute user statistics for potential node features
    user_amounts = defaultdict(list)
    user_txn_counts = defaultdict(int)

    for r in transactions:
        u = uid_map[int(r["user_id"])]
        m = mid_map[int(r["merchant_id"])]
        d = did_map[int(r["device_id"])]
        t = tid_map[int(r["txn_id"])]
        ts = int(r["timestamp"])
        amount = float(r["amount"])

        # Normalize timestamp if requested
        if normalize_time:
            ts_normalized = (ts - min_ts) / max(1, max_ts - min_ts) * 86400.0 * 30  # Scale to ~30 days
        else:
            ts_normalized = ts

        # Transaction node features: [log(amount+1), time_of_day (hour), day_of_week]
        import time
        dt = time.gmtime(ts)
        hour = dt.tm_hour / 24.0  # Normalize to [0, 1]
        day_of_week = dt.tm_wday / 7.0  # Normalize to [0, 1]
        amount_log = torch.log1p(torch.tensor(amount)).item()

        txn_features.append([amount_log, hour, day_of_week])

        # Build edges
        user_src.append(u)
        user_dst.append(t)
        user_ts.append(ts_normalized)

        tm_src.append(t)
        tm_dst.append(m)
        tm_ts.append(ts_normalized)

        td_src.append(t)
        td_dst.append(d)
        td_ts.append(ts_normalized)

        # Collect user statistics
        user_amounts[u].append(amount)
        user_txn_counts[u] += 1

    # Set transaction node features
    if txn_features:
        data["transaction"].x = torch.tensor(txn_features, dtype=torch.float32)
    else:
        # Fallback: empty features (will use learnable embeddings)
        data["transaction"].x = None

    # Build edge tensors
    data["user", "initiates", "transaction"].edge_index = torch.tensor([user_src, user_dst], dtype=torch.long)
    data["user", "initiates", "transaction"].timestamp = torch.tensor(user_ts, dtype=torch.float32)

    data["transaction", "at", "merchant"].edge_index = torch.tensor([tm_src, tm_dst], dtype=torch.long)
    data["transaction", "at", "merchant"].timestamp = torch.tensor(tm_ts, dtype=torch.float32)

    data["transaction", "from", "device"].edge_index = torch.tensor([td_src, td_dst], dtype=torch.long)
    data["transaction", "from", "device"].timestamp = torch.tensor(td_ts, dtype=torch.float32)

    return data


