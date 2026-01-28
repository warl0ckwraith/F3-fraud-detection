"""
Experiment management and logging utilities.

Based on patterns from DGFraud and federated_fraud_detection.
Provides experiment tracking, metrics logging, and result saving.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class ExperimentLogger:
    """
    Simple experiment logger for tracking training runs and metrics.
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "runs",
        save_config: bool = True,
        save_model: bool = False,
    ) -> None:
        """
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save experiment outputs
            save_config: Whether to save config
            save_model: Whether to save model checkpoints
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.save_config = save_config
        self.save_model = save_model

        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.output_dir / f"{experiment_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.metrics: Dict[str, List[float]] = {}
        self.config: Optional[Dict[str, Any]] = None

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        self.config = config
        if self.save_config:
            config_path = self.exp_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def save_metrics(self, filename: str = "metrics.json") -> None:
        """Save all logged metrics to file."""
        metrics_path = self.exp_dir / filename
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def save_model_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        filename: Optional[str] = None,
    ) -> None:
        """Save model checkpoint."""
        if not self.save_model:
            return

        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        checkpoint_path = self.exp_dir / filename
        torch.save(checkpoint, checkpoint_path)

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        summary = {
            "experiment_name": self.experiment_name,
            "output_dir": str(self.exp_dir),
            "metrics": {k: {"mean": sum(v) / len(v), "last": v[-1]} for k, v in self.metrics.items()},
        }
        return summary

    def print_summary(self) -> None:
        """Print experiment summary."""
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print(f"Experiment: {summary['experiment_name']}")
        print(f"Output: {summary['output_dir']}")
        print(f"\nMetrics:")
        for name, stats in summary["metrics"].items():
            print(f"  {name}: mean={stats['mean']:.4f}, last={stats['last']:.4f}")
        print(f"{'='*60}\n")


def compare_experiments(experiment_dirs: List[str], output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare multiple experiments.

    Args:
        experiment_dirs: List of experiment directory paths
        output_file: Optional file to save comparison

    Returns:
        Comparison dictionary
    """
    comparisons = {}
    for exp_dir in experiment_dirs:
        metrics_path = Path(exp_dir) / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            comparisons[exp_dir] = metrics

    if output_file:
        with open(output_file, "w") as f:
            json.dump(comparisons, f, indent=2)

    return comparisons

