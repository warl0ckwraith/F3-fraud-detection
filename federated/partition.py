from typing import Dict, List, Sequence, Tuple
from collections import defaultdict
import random


def non_iid_split(
    user_ids: Sequence[int],
    num_clients: int,
    min_users_per_client: int = 50,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Simple non-IID partitioning by assigning each client a biased subset of users.
    """
    rng = random.Random(seed)
    users = list(set(int(u) for u in user_ids))
    rng.shuffle(users)
    # Start with roughly even split
    splits = [users[i::num_clients] for i in range(num_clients)]
    # Introduce skew by moving chunks between clients
    for i in range(num_clients):
        if i % 2 == 0 and i + 1 < num_clients and len(splits[i]) > min_users_per_client:
            move = max(0, len(splits[i]) // 5)
            chunk = splits[i][:move]
            splits[i] = splits[i][move:]
            splits[i + 1].extend(chunk)
    return {cid: sorted(s) for cid, s in enumerate(splits)}


