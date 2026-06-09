"""General utilities."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_checkpoint(path: str | os.PathLike, model: torch.nn.Module, optimizer, epoch: int, best_metric: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )


def load_checkpoint(path: str | os.PathLike, model: torch.nn.Module, optimizer=None, map_location="cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def write_json(path: str | os.PathLike, data: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
