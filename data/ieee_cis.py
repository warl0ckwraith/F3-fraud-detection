"""
IEEE-CIS Fraud Detection Dataset Loader.

Based on patterns from waittim/graph-fraud-detection and DGFraud.
Supports loading and preprocessing IEEE-CIS fraud detection dataset into
heterogeneous graph format.
"""
from typing import Any, Dict, List, Optional, Tuple
import os
import pandas as pd
import numpy as np
import torch

from .graph import build_hetero_graph


def load_ieee_cis_data(
    data_dir: str,
    train_file: str = "train_transaction.csv",
    train_identity_file: Optional[str] = "train_identity.csv",
    test_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load IEEE-CIS fraud detection dataset.

    Args:
        data_dir: Directory containing dataset files
        train_file: Name of training transaction file
        train_identity_file: Name of training identity file (optional)
        test_file: Name of test file (optional)

    Returns:
        Tuple of (train_df, test_df)
    """
    train_path = os.path.join(data_dir, train_file)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")

    train_df = pd.read_csv(train_path)

    # Merge identity data if available
    if train_identity_file:
        identity_path = os.path.join(data_dir, train_identity_file)
        if os.path.exists(identity_path):
            identity_df = pd.read_csv(identity_path)
            train_df = train_df.merge(identity_df, on="TransactionID", how="left")

    test_df = None
    if test_file:
        test_path = os.path.join(data_dir, test_file)
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            if train_identity_file:
                identity_path = os.path.join(data_dir, train_identity_file.replace("train", "test"))
                if os.path.exists(identity_path):
                    identity_df = pd.read_csv(identity_path)
                    test_df = test_df.merge(identity_df, on="TransactionID", how="left")

    return train_df, test_df


def preprocess_ieee_cis(
    df: pd.DataFrame,
    fill_na: bool = True,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Preprocess IEEE-CIS dataset.

    Args:
        df: Raw dataframe
        fill_na: Whether to fill NaN values
        normalize: Whether to normalize numerical features

    Returns:
        Preprocessed dataframe
    """
    df = df.copy()

    # Handle missing values
    if fill_na:
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ["TransactionID", "isFraud"]:
                df[col].fillna(df[col].median(), inplace=True)

        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown", inplace=True)

    # Normalize numerical features
    if normalize:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ["TransactionID", "isFraud"]:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std

    return df


def ieee_cis_to_transactions(
    df: pd.DataFrame,
    use_identity: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convert IEEE-CIS dataframe to transaction records format.

    Maps IEEE-CIS columns to our transaction schema:
    - TransactionID -> txn_id
    - card1, card2, etc. -> user_id (derived)
    - ProductCD, etc. -> merchant_id (derived)
    - DeviceType, DeviceInfo -> device_id (derived)
    - TransactionDT -> timestamp
    - TransactionAmt -> amount
    - isFraud -> label

    Args:
        df: Preprocessed IEEE-CIS dataframe
        use_identity: Whether to use identity features

    Returns:
        List of transaction dictionaries
    """
    transactions = []

    # Create mappings for entities
    # User: based on card features
    if "card1" in df.columns:
        df["user_id"] = df["card1"].fillna(0).astype(int) % 10000  # Modulo to limit range
    else:
        df["user_id"] = df.index % 10000

    # Merchant: based on ProductCD or other merchant-like features
    if "ProductCD" in df.columns:
        product_map = {prod: i for i, prod in enumerate(df["ProductCD"].unique())}
        df["merchant_id"] = df["ProductCD"].map(product_map)
    else:
        df["merchant_id"] = (df.index // 100) % 500  # Simple hash

    # Device: based on DeviceType/DeviceInfo
    if "DeviceType" in df.columns:
        device_map = {dev: i for i, dev in enumerate(df["DeviceType"].fillna("unknown").unique())}
        df["device_id"] = df["DeviceType"].fillna("unknown").map(device_map)
    else:
        df["device_id"] = (df.index // 50) % 800

    # Convert to transaction records
    for idx, row in df.iterrows():
        txn = {
            "txn_id": int(row.get("TransactionID", idx)),
            "user_id": int(row["user_id"]),
            "merchant_id": int(row["merchant_id"]),
            "device_id": int(row["device_id"]),
            "timestamp": int(row.get("TransactionDT", idx * 3600)),  # Use TransactionDT if available
            "amount": float(row.get("TransactionAmt", 0.0)),
            "label": int(row.get("isFraud", 0)),
        }
        transactions.append(txn)

    return transactions


def load_ieee_cis_graph_dataset(
    data_dir: str,
    use_graph: bool = True,
    fill_na: bool = True,
    normalize: bool = True,
    use_identity: bool = True,
) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
    """
    Load IEEE-CIS dataset and convert to transaction format + graph.

    Args:
        data_dir: Directory containing dataset files
        use_graph: Whether to build graph
        fill_na: Whether to fill NaN values
        normalize: Whether to normalize features
        use_identity: Whether to use identity features

    Returns:
        Tuple of (transactions, graph_data)
    """
    train_df, _ = load_ieee_cis_data(data_dir)
    train_df = preprocess_ieee_cis(train_df, fill_na=fill_na, normalize=normalize)
    transactions = ieee_cis_to_transactions(train_df, use_identity=use_identity)

    graph_data = None
    if use_graph:
        try:
            graph_data = build_hetero_graph(transactions, normalize_time=True)
        except Exception as e:
            print(f"Warning: Could not build graph: {e}")
            graph_data = None

    return transactions, graph_data

