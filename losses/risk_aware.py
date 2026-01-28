from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn


class RiskAwareBCELoss(nn.Module):
    """
    BCE-with-logits loss with optional VaR/ES-style tail emphasis on predicted risk.

    This is a research-friendly starting point and not a finalized financial risk model.

    weight modes:
      - 'none': standard BCE
      - 'var':  upweights samples whose predicted p_hat is in the top alpha-tail
      - 'es':   weights proportionally to the excess over VaR within the tail
    """

    def __init__(
        self,
        alpha: float = 0.95,
        lambda_risk: float = 1.0,
        weighting: Literal["none", "var", "es"] = "var",
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.lambda_risk = float(lambda_risk)
        self.weighting = weighting

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1) or (B,) raw logits
            targets: (B,) binary labels {0,1}
        Returns:
            Scalar loss.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.weighting == "none":
            return bce.mean()

        with torch.no_grad():
            p_hat = torch.sigmoid(logits)
            q_alpha = torch.quantile(p_hat, self.alpha)

        if self.weighting == "var":
            w = 1.0 + self.lambda_risk * (p_hat >= q_alpha).float()
        elif self.weighting == "es":
            tail_excess = (p_hat - q_alpha).clamp_min(0.0)
            denom = tail_excess.mean().clamp_min(1e-8)
            w = 1.0 + self.lambda_risk * (tail_excess / denom)
        else:
            w = torch.ones_like(bce)

        return (w * bce).mean()


