"""Lightweight LoRA wrappers for optional backbone adaptation."""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base_layer
        self.scale = alpha / max(rank, 1)
        self.lora_a = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


class LoRAConv2d(nn.Module):
    def __init__(self, base_layer: nn.Conv2d, rank: int, alpha: float):
        super().__init__()
        self.base = base_layer
        self.scale = alpha / max(rank, 1)
        self.lora_a = nn.Conv2d(base_layer.in_channels, rank, kernel_size=1, bias=False)
        self.lora_b = nn.Conv2d(rank, base_layer.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


def apply_lora_to_matching_modules(module: nn.Module, target_substrings: list[str], rank: int, alpha: float) -> list[str]:
    patched: list[str] = []
    for name, child in list(module.named_children()):
        full_match = any(token in name for token in target_substrings)
        if isinstance(child, nn.Linear) and full_match:
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            patched.append(name)
        elif isinstance(child, nn.Conv2d) and full_match:
            setattr(module, name, LoRAConv2d(child, rank=rank, alpha=alpha))
            patched.append(name)
        else:
            nested = apply_lora_to_matching_modules(child, target_substrings, rank=rank, alpha=alpha)
            patched.extend([f"{name}.{item}" for item in nested])
    return patched
