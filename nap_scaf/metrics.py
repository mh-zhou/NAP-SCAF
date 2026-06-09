"""Segmentation metrics for BraTS-style labels."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

Tensor = torch.Tensor

REGIONS: Mapping[str, Iterable[int]] = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
}


def _region_mask(array: np.ndarray, labels: Iterable[int]) -> np.ndarray:
    mask = np.zeros_like(array, dtype=bool)
    for label in labels:
        mask |= array == label
    return mask


def dice_binary(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def hd95_binary(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    pred_dist = distance_transform_edt(~pred)
    target_dist = distance_transform_edt(~target)
    distances = np.concatenate([pred_dist[target], target_dist[pred]])
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, 95))


def brats_region_metrics(pred: Tensor, target: Tensor) -> Dict[str, float]:
    """Compute WT, TC, and ET Dice/HD95 for a batch."""

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    metrics: Dict[str, float] = {}
    for region, labels in REGIONS.items():
        dice_values = []
        hd95_values = []
        for p, t in zip(pred_np, target_np):
            p_bin = _region_mask(p, labels)
            t_bin = _region_mask(t, labels)
            dice_values.append(dice_binary(p_bin, t_bin))
            hd95_values.append(hd95_binary(p_bin, t_bin))
        metrics[f"{region}_dice"] = float(np.nanmean(dice_values))
        metrics[f"{region}_hd95"] = float(np.nanmean(hd95_values))
    metrics["Avg_dice"] = float(np.mean([metrics[f"{r}_dice"] for r in REGIONS]))
    metrics["Avg_hd95"] = float(np.nanmean([metrics[f"{r}_hd95"] for r in REGIONS]))
    return metrics


class AverageMeter:
    """Accumulate scalar metrics."""

    def __init__(self) -> None:
        self.sum: Dict[str, float] = {}
        self.count = 0

    def update(self, values: Dict[str, float], n: int = 1) -> None:
        for key, value in values.items():
            if not np.isfinite(value):
                continue
            self.sum[key] = self.sum.get(key, 0.0) + float(value) * n
        self.count += n

    def average(self) -> Dict[str, float]:
        if self.count == 0:
            return {}
        return {key: value / self.count for key, value in self.sum.items()}
