"""Loss functions for semantic segmentation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocseg.constants import IGNORE_INDEX, NUM_CLASSES


class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int = IGNORE_INDEX, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.contiguous()
        target = target.contiguous()
        num_classes = logits.shape[1]
        valid_mask = target != self.ignore_index
        target_clamped = target.clone()
        target_clamped[~valid_mask] = 0

        probabilities = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target_clamped, num_classes=num_classes).permute(0, 3, 1, 2).contiguous().float()
        valid_mask = valid_mask.unsqueeze(1).contiguous()

        probabilities = probabilities * valid_mask
        target_one_hot = target_one_hot * valid_mask

        intersection = (probabilities * target_one_hot).sum(dim=(0, 2, 3))
        denominator = probabilities.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        valid_classes = target_one_hot.sum(dim=(0, 2, 3)) > 0
        dice = dice[valid_classes]
        if dice.numel() == 0:
            return logits.new_tensor(0.0)
        return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.contiguous()
        target = target.contiguous()
        return self.ce_weight * self.cross_entropy(logits, target) + self.dice_weight * self.dice(logits, target)


def build_loss(loss_cfg: dict[str, Any]) -> nn.Module:
    name = loss_cfg.get("name", "ce_dice")
    ignore_index = int(loss_cfg.get("ignore_index", IGNORE_INDEX))

    if name == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    if name == "dice":
        return DiceLoss(ignore_index=ignore_index)
    if name == "ce_dice":
        return CombinedSegmentationLoss(
            ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
            dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
            ignore_index=ignore_index,
        )
    raise ValueError(f"Unknown loss configuration: {name}")
