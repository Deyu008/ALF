"""Hierarchical CLS-to-Patch fusion utilities for MuSc."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_EPS = 1e-8


@dataclass
class FusionStats:
    """Holds per-batch diagnostics for downstream logging."""

    entropy: torch.Tensor
    weights: torch.Tensor
    neff: torch.Tensor
    lambda_scale: Optional[torch.Tensor] = None
    top_p: Optional[torch.Tensor] = None
    temperature: Optional[torch.Tensor] = None


@dataclass
class AdaptiveConfig:
    enabled: bool = False
    neff_floor: float = 1.5
    neff_ceil: float = 4.0
    lambda_min_scale: float = 0.3
    lambda_max_scale: float = 1.0
    log_stats: bool = True
    top_p_enabled: bool = False
    top_p_min: float = 0.8
    top_p_max: float = 0.98
    temperature_enabled: bool = False
    temperature_min: float = 0.6
    temperature_max: float = 1.2

    def __post_init__(self) -> None:
        if self.neff_ceil < self.neff_floor:
            self.neff_floor, self.neff_ceil = self.neff_ceil, self.neff_floor
        if self.lambda_max_scale < self.lambda_min_scale:
            self.lambda_min_scale, self.lambda_max_scale = (
                self.lambda_max_scale,
                self.lambda_min_scale,
            )
        if self.top_p_max < self.top_p_min:
            self.top_p_min, self.top_p_max = self.top_p_max, self.top_p_min
        self.top_p_min = float(max(_EPS, min(self.top_p_min, 1.0)))
        self.top_p_max = float(max(self.top_p_min, min(self.top_p_max, 1.0)))
        if self.temperature_max < self.temperature_min:
            self.temperature_min, self.temperature_max = (
                self.temperature_max,
                self.temperature_min,
            )
        self.temperature_min = float(max(_EPS, self.temperature_min))
        self.temperature_max = float(max(self.temperature_min, self.temperature_max))

    @classmethod
    def from_dict(cls, cfg: Dict) -> "AdaptiveConfig":
        if not cfg:
            return cls()
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            neff_floor=float(cfg.get("neff_floor", 1.5)),
            neff_ceil=float(cfg.get("neff_ceil", 4.0)),
            lambda_min_scale=float(cfg.get("lambda_min_scale", 0.3)),
            lambda_max_scale=float(cfg.get("lambda_max_scale", 1.0)),
            log_stats=bool(cfg.get("log_stats", True)),
            top_p_enabled=bool(cfg.get("top_p_enabled", False)),
            top_p_min=float(cfg.get("top_p_min", 0.8)),
            top_p_max=float(cfg.get("top_p_max", 0.98)),
            temperature_enabled=bool(cfg.get("temperature_enabled", False)),
            temperature_min=float(cfg.get("temperature_min", 0.6)),
            temperature_max=float(cfg.get("temperature_max", 1.2)),
        )


class HierarchicalFusion:
    """Implements CLS→Patch attention and layerwise fusion without training."""

    def __init__(self, cfg: Dict, device: torch.device) -> None:
        self.device = device
        self.temperature: float = float(cfg.get("temperature", 0.7))
        self.top_p: float = float(cfg.get("top_p", 1.0))
        self.lambda_patch: float = float(cfg.get("lambda_patch", 0.0))
        self.patch_mode: str = str(cfg.get("patch_mode", "mult")).lower()
        self.gamma_img: float = float(cfg.get("gamma_img", 0.0))
        self.cls_topk_ratio: float = float(cfg.get("cls_topk_ratio", 0.3))
        self.stats: List[FusionStats] = []
        self.adaptive = AdaptiveConfig.from_dict(cfg.get("adaptive", {}))

    def reset_stats(self) -> None:
        self.stats.clear()

    def compute_batch(self, patch_tokens: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Computes per-layer saliency maps and returns fused attention."""
        if not patch_tokens:
            raise ValueError("`patch_tokens` must be a non-empty list.")

        attn_maps: List[torch.Tensor] = []
        contexts: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        raw_logits_store: List[torch.Tensor] = []
        patch_vecs_store: List[torch.Tensor] = []

        temperature = max(self.temperature, _EPS)

        for layer_tokens in patch_tokens:
            layer_tokens = layer_tokens.to(self.device)
            cls_token = layer_tokens[:, 0, :].float()
            patch_vecs = layer_tokens[:, 1:, :].float()

            cls_norm = F.normalize(cls_token, dim=-1)
            patch_norm = F.normalize(patch_vecs, dim=-1)

            attn_logits_raw = torch.matmul(
                cls_norm.unsqueeze(1), patch_norm.transpose(1, 2)
            ).squeeze(1)
            raw_logits_store.append(attn_logits_raw)
            patch_vecs_store.append(patch_vecs)

            attn_logits = attn_logits_raw / temperature
            attn = torch.softmax(attn_logits, dim=-1)
            attn = self._apply_top_p(attn)
            attn = attn.clamp_min(_EPS)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(_EPS)

            context = torch.matmul(attn.unsqueeze(1), patch_vecs).squeeze(1)
            entropy = -(attn * torch.log(attn)).sum(dim=-1)

            attn_maps.append(attn)
            contexts.append(context)
            entropies.append(entropy)

        attn_stack = torch.stack(attn_maps, dim=1)
        entropy_stack = torch.stack(entropies, dim=1)
        raw_weights = torch.exp(-entropy_stack).clamp_min(_EPS)
        weights = raw_weights / raw_weights.sum(dim=1, keepdim=True).clamp_min(_EPS)
        fused_attn = torch.sum(weights.unsqueeze(-1) * attn_stack, dim=1)
        neff = 1.0 / torch.sum(weights ** 2, dim=1).clamp_min(_EPS)

        lambda_scale, top_p_override, temperature_override = self._compute_adaptive_controls(neff)

        base_top_p = attn_stack.new_full((neff.size(0),), float(self.top_p))
        top_p_values = (
            top_p_override if top_p_override is not None else base_top_p
        )

        base_temperature = attn_stack.new_full((neff.size(0),), float(temperature))
        temperature_values = (
            temperature_override if temperature_override is not None else base_temperature
        )

        need_top_p = top_p_override is not None and not torch.allclose(
            top_p_values, base_top_p, atol=1e-4
        )
        need_temperature = temperature_override is not None and not torch.allclose(
            temperature_values, base_temperature, atol=1e-4
        )

        if need_top_p or need_temperature:
            attn_maps = []
            contexts = []
            entropies = []

            top_p_tensor = top_p_values.to(self.device, dtype=attn_stack.dtype)
            temperature_tensor = temperature_values.to(self.device, dtype=attn_stack.dtype).clamp_min(_EPS)

            for idx, attn_logits_raw in enumerate(raw_logits_store):
                attn_logits_raw = attn_logits_raw.to(self.device)
                logits = attn_logits_raw / temperature_tensor.unsqueeze(-1)
                attn = torch.softmax(logits, dim=-1)
                attn = self._apply_top_p(attn, top_p_tensor)
                attn = attn.clamp_min(_EPS)
                attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(_EPS)

                patch_vecs = patch_vecs_store[idx].to(self.device)
                context = torch.matmul(attn.unsqueeze(1), patch_vecs).squeeze(1)
                entropy = -(attn * torch.log(attn)).sum(dim=-1)

                attn_maps.append(attn)
                contexts.append(context)
                entropies.append(entropy)

            attn_stack = torch.stack(attn_maps, dim=1)
            entropy_stack = torch.stack(entropies, dim=1)
            raw_weights = torch.exp(-entropy_stack).clamp_min(_EPS)
            weights = raw_weights / raw_weights.sum(dim=1, keepdim=True).clamp_min(_EPS)
            fused_attn = torch.sum(weights.unsqueeze(-1) * attn_stack, dim=1)
            neff = 1.0 / torch.sum(weights ** 2, dim=1).clamp_min(_EPS)

            if self.adaptive.enabled and self.lambda_patch > 0:
                lambda_scale = self._compute_lambda_scale(neff)

        if lambda_scale is not None:
            lambda_scale = lambda_scale.to(self.device)

        stats_lambda = (
            lambda_scale.detach().cpu()
            if lambda_scale is not None and self.adaptive.log_stats
            else None
        )
        stats_top_p = (
            top_p_values.detach().cpu()
            if (self.adaptive.top_p_enabled and self.adaptive.log_stats)
            else None
        )
        stats_temperature = (
            temperature_values.detach().cpu()
            if (self.adaptive.temperature_enabled and self.adaptive.log_stats)
            else None
        )

        self.stats.append(
            FusionStats(
                entropy_stack.detach().cpu(),
                weights.detach().cpu(),
                neff.detach().cpu(),
                stats_lambda,
                stats_top_p,
                stats_temperature,
            )
        )

        return {
            "layer_attn": attn_maps,
            "fused_attn": fused_attn,
            "weights": weights,
            "contexts": contexts,
            "entropy": entropy_stack,
            "neff": neff,
            "lambda_scale": lambda_scale,
            "top_p_values": top_p_values,
            "temperature_values": temperature_values,
        }

    def apply_patch_gate(
        self,
        patch_tokens: List[torch.Tensor],
        fused_attn: torch.Tensor,
        lambda_scale: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if self.lambda_patch <= 0:
            return patch_tokens

        fused_norm = self._normalize_map(fused_attn)
        if lambda_scale is not None:
            lambda_eff: torch.Tensor | float = self.lambda_patch * lambda_scale.to(fused_norm.device, fused_norm.dtype)
        else:
            lambda_eff = self.lambda_patch

        if isinstance(lambda_eff, torch.Tensor):
            gate = 1.0 + lambda_eff.view(-1, 1, 1) * fused_norm.unsqueeze(-1)
        else:
            gate = (1.0 + lambda_eff * fused_norm).unsqueeze(-1)

        gate = gate.to(patch_tokens[0].device)
        gated_tokens: List[torch.Tensor] = []
        for layer_tokens in patch_tokens:
            cls_token = layer_tokens[:, :1, :]
            patch_vecs = layer_tokens[:, 1:, :]
            gate_val = gate.to(layer_tokens.dtype)
            gated_patches = patch_vecs * gate_val
            gated_tokens.append(torch.cat([cls_token, gated_patches], dim=1))
        return gated_tokens

    def rescore_patch_scores(
        self,
        scores: torch.Tensor,
        fused_attn: torch.Tensor,
        lambda_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.lambda_patch <= 0:
            return scores
        fused_norm = self._normalize_map(fused_attn).to(scores.dtype)
        if lambda_scale is not None:
            lambda_eff: torch.Tensor | float = self.lambda_patch * lambda_scale.to(scores.device, scores.dtype)
        else:
            lambda_eff = self.lambda_patch
        if self.patch_mode == "add":
            if isinstance(lambda_eff, torch.Tensor):
                return scores + lambda_eff.view(-1, 1) * fused_norm
            return scores + lambda_eff * fused_norm
        if isinstance(lambda_eff, torch.Tensor):
            return scores * (1.0 + lambda_eff.view(-1, 1) * fused_norm)
        return scores * (1.0 + lambda_eff * fused_norm)

    def rescore_image_scores(
        self,
        image_scores: List[float],
        cls_tokens: torch.Tensor,
    ) -> List[float]:
        if self.gamma_img <= 0 or cls_tokens.ndim != 2 or cls_tokens.shape[0] <= 1:
            return image_scores
        base = torch.tensor(image_scores, device=self.device, dtype=torch.float32)
        cls_norm = F.normalize(cls_tokens.to(self.device).float(), dim=-1)
        sim = torch.matmul(cls_norm, cls_norm.t())
        sim.fill_diagonal_(-1.0)

        topk = max(1, int(self.cls_topk_ratio * (cls_norm.shape[0] - 1)))
        topk_values, _ = torch.topk(sim, k=topk, dim=-1)
        inconsistency = 1.0 - topk_values.mean(dim=-1)
        # Normalize across the whole batch (not per-sample). For (N, 1) vectors,
        # per-row min-max would degenerate to zeros and make rescoring a no-op.
        inc_min = inconsistency.min()
        inc_max = inconsistency.max()
        denom = (inc_max - inc_min).clamp_min(_EPS)
        inconsistency = (inconsistency - inc_min) / denom
        fused = self.gamma_img * base + (1.0 - self.gamma_img) * inconsistency
        return fused.cpu().tolist()

    def gather_stats(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self.stats:
            return None
        entropy = torch.cat([s.entropy for s in self.stats], dim=0)
        weights = torch.cat([s.weights for s in self.stats], dim=0)
        neff = torch.cat([s.neff for s in self.stats], dim=0)
        lambda_scale = None
        if any(s.lambda_scale is not None for s in self.stats):
            lambda_list = [s.lambda_scale for s in self.stats if s.lambda_scale is not None]
            if lambda_list:
                lambda_scale = torch.cat(lambda_list, dim=0)
        top_p = None
        if any(s.top_p is not None for s in self.stats):
            top_p_list = [s.top_p for s in self.stats if s.top_p is not None]
            if top_p_list:
                top_p = torch.cat(top_p_list, dim=0)
        temperature = None
        if any(s.temperature is not None for s in self.stats):
            temp_list = [s.temperature for s in self.stats if s.temperature is not None]
            if temp_list:
                temperature = torch.cat(temp_list, dim=0)
        return {
            "entropy": entropy,
            "weights": weights,
            "neff": neff,
            "lambda_scale": lambda_scale,
            "top_p": top_p,
            "temperature": temperature,
        }

    def _apply_top_p(
        self, attn: torch.Tensor, top_p_override: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if top_p_override is None and self.top_p >= 1.0:
            return attn
        if top_p_override is None:
            top_p_values = attn.new_full((attn.shape[0],), float(self.top_p))
        else:
            top_p_values = top_p_override.to(attn.device, dtype=attn.dtype)
        top_p_values = top_p_values.clamp(min=_EPS, max=1.0)
        if torch.all(top_p_values >= 1.0 - 1e-6):
            return attn
        sorted_attn, sorted_idx = torch.sort(attn, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_attn, dim=-1)
        keep_mask = cumulative <= top_p_values.unsqueeze(-1)
        keep_mask[..., 0] = True
        filtered = sorted_attn * keep_mask
        filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        restored = torch.zeros_like(attn).scatter_(-1, sorted_idx, filtered)
        return restored

    def _normalize_map(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 1:
            values = values.unsqueeze(-1)
        if values.ndim == 2:
            min_val = values.min(dim=-1, keepdim=True).values
            max_val = values.max(dim=-1, keepdim=True).values
        else:
            min_val = values.min(dim=-1, keepdim=True).values
            max_val = values.max(dim=-1, keepdim=True).values
        denom = (max_val - min_val).clamp_min(_EPS)
        return (values - min_val) / denom

    def _compute_lambda_scale(self, neff: torch.Tensor) -> torch.Tensor:
        cfg = self.adaptive
        alpha = self._normalize_neff(neff)
        # 反向调度：不确定性高(α大)时，减弱patch gate力度
        scale = cfg.lambda_max_scale - alpha * (
            cfg.lambda_max_scale - cfg.lambda_min_scale
        )
        return scale.clamp(cfg.lambda_min_scale, cfg.lambda_max_scale)

    def _compute_adaptive_controls(
        self, neff: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        cfg = self.adaptive
        alpha = self._normalize_neff(neff)

        lambda_scale: Optional[torch.Tensor] = None
        if cfg.enabled and self.lambda_patch > 0:
            lambda_scale = self._compute_lambda_scale(neff)

        top_p: Optional[torch.Tensor] = None
        if cfg.top_p_enabled:
            # 不确定性高(α大)时，扩大覆盖（更大top_p）
            top_p = cfg.top_p_min + alpha * (cfg.top_p_max - cfg.top_p_min)
            top_p = top_p.clamp(min=_EPS, max=1.0)

        temperature: Optional[torch.Tensor] = None
        if cfg.temperature_enabled:
            # 不确定性高(α大)时，提高温度（更平滑）
            temperature = cfg.temperature_min + alpha * (
                cfg.temperature_max - cfg.temperature_min
            )
            temperature = temperature.clamp_min(_EPS)

        return lambda_scale, top_p, temperature

    def _normalize_neff(self, neff: torch.Tensor) -> torch.Tensor:
        cfg = self.adaptive
        neff = neff.to(torch.float32)
        if cfg.neff_ceil <= cfg.neff_floor:
            return torch.ones_like(neff)
        alpha = (neff - cfg.neff_floor) / (cfg.neff_ceil - cfg.neff_floor)
        return alpha.clamp(0.0, 1.0)
