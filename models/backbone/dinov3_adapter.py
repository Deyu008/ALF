"""DINOv3 backbone loader for MuSc."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch import nn

# Ensure the local DINOv3 repository is importable *before* importing it.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DINOV3_PACKAGE_ROOT = _REPO_ROOT / "dinov3"
if str(_DINOV3_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DINOV3_PACKAGE_ROOT))

from dinov3.hub.backbones import (  # type: ignore[attr-defined]
    Weights as Dinov3Weights,
    dinov3_vitb16,
    dinov3_vitl16,
    dinov3_vitl16plus,
    dinov3_vits16,
    dinov3_vits16plus,
)


@dataclass(frozen=True)
class _Dinov3Entry:
    builder: Callable[..., nn.Module]
    default_weights: Dinov3Weights
    local_weight_filename: Optional[str] = None


_DINOV3_REGISTRY: Dict[str, _Dinov3Entry] = {
    "dinov3_vits16": _Dinov3Entry(
        builder=dinov3_vits16,
        default_weights=Dinov3Weights.LVD1689M,
    ),
    "dinov3_vits16plus": _Dinov3Entry(
        builder=dinov3_vits16plus,
        default_weights=Dinov3Weights.LVD1689M,
    ),
    "dinov3_vitb16": _Dinov3Entry(
        builder=dinov3_vitb16,
        default_weights=Dinov3Weights.LVD1689M,
    ),
    "dinov3_vitl16": _Dinov3Entry(
        builder=dinov3_vitl16,
        default_weights=Dinov3Weights.LVD1689M,
        local_weight_filename="dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    ),
    "dinov3_vitl16plus": _Dinov3Entry(
        builder=dinov3_vitl16plus,
        default_weights=Dinov3Weights.LVD1689M,
    ),
}


class DinoV3Backbone(nn.Module):
    """Thin wrapper adding MuSc-friendly helper methods around a DINOv3 ViT."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.embed_dim = getattr(model, "embed_dim", None)
        self.patch_size = getattr(model, "patch_size", None)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    # type: ignore[override]
    def get_intermediate_layers(self, x: torch.Tensor, *, n: int, **kwargs):
        # We always request class tokens and storage tokens to restore the original MuSc layout.
        outputs = self.model.get_intermediate_layers(
            x=x,
            n=n,
            return_class_token=True,
            return_extra_tokens=True,
            **kwargs,
        )
        merged_outputs = []
        for patches, class_token, _ in outputs:
            tokens = [class_token.unsqueeze(1)]
            tokens.append(patches)
            merged_outputs.append(torch.cat(tokens, dim=1))
        return tuple(merged_outputs)


def load_dinov3_backbone(name: str, device: Optional[torch.device] = None) -> DinoV3Backbone:
    name = name.lower()
    if name not in _DINOV3_REGISTRY:
        raise ValueError(
            f"Unsupported DINOv3 backbone '{name}'. Supported keys: {sorted(_DINOV3_REGISTRY.keys())}.")

    entry = _DINOV3_REGISTRY[name]
    weight_arg: object = entry.default_weights
    if entry.local_weight_filename is not None:
        local_path = (_REPO_ROOT / "weights" /
                      entry.local_weight_filename).resolve()
        if local_path.exists():
            weight_arg = str(local_path)
    model = entry.builder(
        pretrained=True, weights=weight_arg, check_hash=False)
    backbone = DinoV3Backbone(model=model)
    if device is not None:
        backbone.to(device)
    return backbone
