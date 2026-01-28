from typing import Dict, Optional

import torch


def recall_at_k(y_true: torch.Tensor, y_scores: torch.Tensor, k: float = 0.03) -> float:
    """
    Recall among top-k fraction (if k in (0,1]) or top-k samples (if k > 1).
    """
    y_true = y_true.detach().view(-1).cpu()
    y_scores = y_scores.detach().view(-1).cpu()
    n = y_true.numel()
    k_count = int(k * n) if 0 < k <= 1 else int(k)
    k_count = max(1, min(n, k_count))
    _, idx = torch.topk(y_scores, k_count)
    selected = torch.zeros_like(y_true, dtype=torch.bool)
    selected[idx] = True
    tp = (selected & (y_true == 1)).sum().item()
    pos = (y_true == 1).sum().item()
    if pos == 0:
        return 0.0
    return float(tp / pos)


def pr_auc(y_true: torch.Tensor, y_scores: torch.Tensor) -> Optional[float]:
    """
    Average precision (area under PR curve). Returns None if sklearn unavailable.
    """
    try:
        from sklearn.metrics import average_precision_score  # type: ignore

        return float(average_precision_score(y_true.detach().cpu().numpy(), y_scores.detach().cpu().numpy()))
    except Exception:
        return None


def cost_metrics(
    y_true: torch.Tensor,
    y_scores: torch.Tensor,
    threshold: float = 0.5,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
) -> Dict[str, float]:
    """
    Simple cost at a fixed threshold.
    """
    y_true = y_true.detach().view(-1).cpu()
    y_scores = y_scores.detach().view(-1).cpu()
    preds = (y_scores >= threshold).int()
    fp = ((preds == 1) & (y_true == 0)).sum().item()
    fn = ((preds == 0) & (y_true == 1)).sum().item()
    total = y_true.numel()
    return {
        "fp": float(fp),
        "fn": float(fn),
        "cost": float(cost_fp * fp + cost_fn * fn),
        "threshold": float(threshold),
        "rate_fp": float(fp / total),
        "rate_fn": float(fn / total),
    }


