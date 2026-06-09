"""Training losses for NAP-SCAF."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class SoftDiceLoss(nn.Module):
    """Multi-class soft Dice loss."""

    def __init__(self, num_classes: int = 4, smooth: float = 1e-6, include_background: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target.long(), self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dims)
        denominator = torch.sum(probs + target_one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if not self.include_background:
            dice = dice[1:]
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """Cross-entropy plus Dice loss."""

    def __init__(self, num_classes: int = 4, dice_weight: float = 0.5, ce_weight: float = 0.5) -> None:
        super().__init__()
        self.dice = SoftDiceLoss(num_classes=num_classes, include_background=True)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        ce = F.cross_entropy(logits, target.long())
        dice = self.dice(logits, target)
        return self.ce_weight * ce + self.dice_weight * dice


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL divergence between q(z|x) and standard normal."""

    axes = tuple(range(1, mu.ndim))
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=axes)
    return kl.mean()


def reconstruction_loss(reconstruction: Tensor, target: Tensor) -> Tensor:
    """Mean squared reconstruction loss."""

    if reconstruction.shape[-2:] != target.shape[-2:]:
        reconstruction = F.interpolate(reconstruction, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return F.mse_loss(reconstruction, target)


def total_nap_scaf_loss(
    output,
    target: Tensor,
    image: Optional[Tensor] = None,
    num_classes: int = 4,
    rec_weight: float = 0.05,
    kl_weight: float = 1e-4,
) -> Tensor:
    """Compute the segmentation loss and optional NAP variational losses.

    ``output`` may be either logits or the dictionary returned by
    ``model(image, return_aux=True)``.
    """

    seg_loss = SegmentationLoss(num_classes=num_classes)
    if isinstance(output, dict):
        loss = seg_loss(output["logits"], target)
        if image is not None and "reconstruction" in output:
            loss = loss + rec_weight * reconstruction_loss(output["reconstruction"], image)
        if "mu" in output and "logvar" in output:
            loss = loss + kl_weight * kl_divergence(output["mu"], output["logvar"])
        return loss
    return seg_loss(output, target)
