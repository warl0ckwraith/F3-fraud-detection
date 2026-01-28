from typing import Any, Dict, List, Optional
import json
import math
import os
import time

import torch


def generate_synthetic_transactions(
    num_users: int,
    num_merchants: int,
    num_devices: int,
    num_transactions: int,
    fraud_ratio: float = 0.02,
    seed: int = 42,
    base_timestamp: Optional[int] = None,
    profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate a synthetic transactions dataset with richer, more realistic patterns.

    Each record includes:
      - txn_id: int
      - user_id: int in [0, num_users)
      - merchant_id: int in [0, num_merchants)
      - device_id: int in [0, num_devices)
      - timestamp: int (unix seconds)
      - amount: float
      - label: int (0=legit, 1=fraud)

    Notes:
      - This generator induces risk via user/merchant/device factors, time-of-day,
        cross-border behavior, velocity, and new-device usage.
      - Amounts follow a log-normal distribution with user and channel effects.
      - Timestamps span ~30 days with per-user Poisson arrival processes.
      - Use base_timestamp for deterministic timestamps across runs.
    """
    if num_transactions <= 0:
        return []

    profile = profile or {}
    gen = torch.Generator().manual_seed(seed)
    now = int(time.time()) if base_timestamp is None else int(base_timestamp)
    horizon_days = int(profile.get("horizon_days", 30))
    horizon_secs = horizon_days * 24 * 3600

    amount_log_mean = float(profile.get("amount_log_mean", 3.5))
    amount_log_sigma = float(profile.get("amount_log_sigma", 0.8))
    user_amount_mu_sigma = float(profile.get("user_amount_mu_sigma", 0.35))
    user_activity_sigma = float(profile.get("user_activity_sigma", 1.0))
    merchant_pop_sigma = float(profile.get("merchant_pop_sigma", 1.0))
    new_device_prob = float(profile.get("new_device_prob", 0.05))
    cross_border_prob = float(profile.get("cross_border_prob", 0.08))
    velocity_hours = float(profile.get("velocity_hours", 1.0))

    # Discrete distributions
    country_probs = torch.tensor(profile.get("country_probs", [0.35, 0.25, 0.2, 0.12, 0.08]), dtype=torch.float32)
    country_probs = country_probs / country_probs.sum()
    mcc_probs = torch.tensor(profile.get("mcc_probs", [0.25, 0.2, 0.18, 0.15, 0.12, 0.1]), dtype=torch.float32)
    mcc_probs = mcc_probs / mcc_probs.sum()
    channel_probs = torch.tensor(profile.get("channel_probs", [0.5, 0.3, 0.2]), dtype=torch.float32)
    channel_probs = channel_probs / channel_probs.sum()
    channel_amount_bias = torch.tensor(profile.get("channel_amount_bias", [0.0, 0.2, 0.35]), dtype=torch.float32)
    if channel_amount_bias.numel() != channel_probs.numel():
        channel_amount_bias = torch.zeros(channel_probs.numel(), dtype=torch.float32)

    # User attributes
    user_activity = torch.exp(torch.randn(num_users, generator=gen) * user_activity_sigma)
    user_activity = user_activity / user_activity.sum()
    user_counts = torch.bincount(
        torch.multinomial(user_activity, num_transactions, replacement=True, generator=gen),
        minlength=num_users,
    )
    user_risk = torch.sigmoid(torch.randn(num_users, generator=gen))
    user_amount_mu = torch.randn(num_users, generator=gen) * user_amount_mu_sigma + amount_log_mean
    user_country = torch.multinomial(country_probs, num_users, replacement=True, generator=gen)
    user_primary_device = torch.randint(0, max(1, num_devices), (num_users,), generator=gen)

    # Merchant attributes
    merchant_pop = torch.exp(torch.randn(num_merchants, generator=gen) * merchant_pop_sigma)
    merchant_pop = merchant_pop / merchant_pop.sum()
    merchant_risk = torch.rand(num_merchants, generator=gen)
    risky_merchant_frac = float(profile.get("risky_merchant_frac", 0.1))
    if num_merchants > 0:
        k = max(1, int(risky_merchant_frac * num_merchants))
        top_idx = torch.topk(merchant_risk, k).indices
        merchant_risk[top_idx] = 0.7 + 0.3 * merchant_risk[top_idx]
    merchant_country = torch.multinomial(country_probs, num_merchants, replacement=True, generator=gen)
    merchant_mcc = torch.multinomial(mcc_probs, num_merchants, replacement=True, generator=gen)

    # Device attributes
    device_risk = torch.rand(max(1, num_devices), generator=gen)
    risky_device_frac = float(profile.get("risky_device_frac", 0.05))
    if num_devices > 0:
        k = max(1, int(risky_device_frac * num_devices))
        top_idx = torch.topk(device_risk, k).indices
        device_risk[top_idx] = 0.7 + 0.3 * device_risk[top_idx]

    risk_weights = {
        "bias": -2.5,
        "user": 1.2,
        "merchant": 1.5,
        "device": 1.0,
        "night": 0.8,
        "cross_border": 0.9,
        "new_device": 1.0,
        "high_amount": 0.9,
        "velocity": 0.7,
    }
    if isinstance(profile.get("risk_weights"), dict):
        risk_weights.update(profile["risk_weights"])

    records: List[Dict[str, Any]] = []
    risk_scores: List[float] = []

    for u in range(num_users):
        n = int(user_counts[u].item())
        if n == 0:
            continue

        merch_ids = torch.multinomial(merchant_pop, n, replacement=True, generator=gen)
        channels = torch.multinomial(channel_probs, n, replacement=True, generator=gen)
        device_ids = torch.full((n,), int(user_primary_device[u].item()), dtype=torch.long)
        if num_devices > 0:
            use_new = torch.rand(n, generator=gen) < new_device_prob
            if use_new.any():
                device_ids[use_new] = torch.randint(0, num_devices, (int(use_new.sum().item()),), generator=gen)

        # Poisson arrival process per user
        rate = max(1, n) / float(horizon_secs)
        u_rand = torch.rand(n, generator=gen).clamp_min(1e-6)
        inter = -torch.log(u_rand) / rate
        times = torch.cumsum(inter, 0)
        if times[-1] > 0:
            times = times * (horizon_secs / times[-1])
        timestamps = now - (horizon_secs - times)

        prev_ts = None
        mu = float(user_amount_mu[u].item())
        user_cty = int(user_country[u].item())
        primary_device = int(user_primary_device[u].item())

        for i in range(n):
            ts_val = float(timestamps[i].item())
            ts_int = int(ts_val)
            delta_hours = 0.0
            if prev_ts is not None:
                delta_hours = max(0.0, (ts_val - prev_ts) / 3600.0)
            prev_ts = ts_val

            m_id = int(merch_ids[i].item())
            d_id = int(device_ids[i].item())
            ch = int(channels[i].item())

            amount_log = mu + float(channel_amount_bias[ch].item()) + float(torch.randn(1, generator=gen).item()) * amount_log_sigma
            amount = math.exp(amount_log)

            dt = time.gmtime(ts_int)
            hour = dt.tm_hour
            is_night = 1.0 if (hour <= 5 or hour >= 22) else 0.0

            merch_cty = int(merchant_country[m_id].item())
            cross_border = 1.0 if user_cty != merch_cty else 0.0
            if cross_border_prob > 0.0 and torch.rand(1, generator=gen).item() < cross_border_prob:
                cross_border = 1.0

            new_device = 1.0 if d_id != primary_device else 0.0
            high_amount = 1.0 if amount_log > (mu + 1.5 * amount_log_sigma) else 0.0
            velocity_short = 1.0 if delta_hours < velocity_hours else 0.0

            risk_score = (
                risk_weights["bias"]
                + risk_weights["user"] * float(user_risk[u].item())
                + risk_weights["merchant"] * float(merchant_risk[m_id].item())
                + risk_weights["device"] * float(device_risk[d_id].item())
                + risk_weights["night"] * is_night
                + risk_weights["cross_border"] * cross_border
                + risk_weights["new_device"] * new_device
                + risk_weights["high_amount"] * high_amount
                + risk_weights["velocity"] * velocity_short
            )

            records.append(
                {
                    "txn_id": 0,  # filled after shuffling
                    "user_id": int(u),
                    "merchant_id": m_id,
                    "device_id": d_id,
                    "timestamp": ts_int,
                    "amount": float(amount),
                    "channel": ch,
                    "country": user_cty,
                    "merchant_country": merch_cty,
                    "mcc": int(merchant_mcc[m_id].item()),
                    "is_cross_border": float(cross_border),
                    "is_new_device": float(new_device),
                    "velocity_hours": float(delta_hours),
                    "risk_score": float(risk_score),
                }
            )
            risk_scores.append(float(risk_score))

    if not records:
        return []

    risk_tensor = torch.tensor(risk_scores, dtype=torch.float32)
    p_raw = torch.sigmoid(risk_tensor)
    mean_p = float(p_raw.mean().item()) if p_raw.numel() > 0 else 1.0
    if mean_p <= 0.0:
        mean_p = 1.0
    p = (p_raw / mean_p) * float(fraud_ratio)
    p = p.clamp(0.0, 0.95)
    labels = torch.bernoulli(p, generator=gen).to(torch.int64).tolist()

    perm = torch.randperm(len(records), generator=gen).tolist()
    shuffled: List[Dict[str, Any]] = []
    for new_id, idx in enumerate(perm):
        rec = records[idx]
        rec["txn_id"] = int(new_id)
        rec["label"] = int(labels[idx])
        shuffled.append(rec)
    return shuffled


def save_transactions_jsonl(transactions: List[Dict[str, Any]], path: str) -> None:
    """Save transaction records to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in transactions:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def load_transactions_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load transaction records from a JSONL file."""
    transactions: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            transactions.append(json.loads(line))
    return transactions
