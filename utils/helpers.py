# utils/helpers.py
import random
import numpy as np
import torch
import os
import json


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_metrics(log_file: str) -> list:
    """Load metrics from JSONL file."""
    metrics = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
    return metrics
