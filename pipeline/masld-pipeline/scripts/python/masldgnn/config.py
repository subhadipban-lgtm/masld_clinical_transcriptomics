# -*- coding: utf-8 -*-
"""config.py — Central configuration loader.

Reads config.yaml at the pipeline root and exposes every threshold
as a Python dict.  All scripts import from here so nothing is
hard-coded in multiple places.
"""

import os
import yaml
from pathlib import Path


def _find_config() -> Path:
    """Walk up from this file to find config.yaml."""
    d = Path(__file__).resolve().parent  # masldgnn/
    for _ in range(5):
        candidate = d / "config.yaml"
        if candidate.is_file():
            return candidate
        d = d.parent
    raise FileNotFoundError(
        "config.yaml not found. Place it at the pipeline root."
    )


_cfg_cache = None


def load_config() -> dict:
    """Load (and cache) the YAML config."""
    global _cfg_cache
    if _cfg_cache is not None:
        return _cfg_cache
    path = _find_config()
    with open(path) as f:
        _cfg_cache = yaml.safe_load(f)
    return _cfg_cache


def get_device():
    """Return torch device (MPS on Apple Silicon, else CPU)."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_stats(stats: dict, stage: str, results_dir: str = "results") -> str:
    """Write a stats dict to results/stats_{stage}.json.

    Every analysis script MUST call this after computing any metric.
    Rule: computed values win over manuscript values; surface mismatches,
    never reconcile them silently (AGENT.md Rule 3).

    Parameters
    ----------
    stats       : dict of metric name → value (JSON-serialisable)
    stage       : short label, e.g. "graph_audit", "dge_01", "gnn_eval"
    results_dir : directory to write into (created if absent)

    Returns
    -------
    str : absolute path of the written JSON file
    """
    import json
    from pathlib import Path

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stats_{stage}.json"

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[write_stats] Saved → {out_path}")
    return str(out_path)
