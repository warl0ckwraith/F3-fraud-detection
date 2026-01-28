"""
Federated learning components (stubs).

Includes:
- Non-IID partitioning utilities
- Client training loop
- Server FedAvg aggregation
"""

from .partition import non_iid_split
from .client import FLClient
from .server import FedAvgServer

__all__ = ["non_iid_split", "FLClient", "FedAvgServer"]


