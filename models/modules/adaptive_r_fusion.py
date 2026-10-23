"""Adaptive Multi-Scale Fusion (AMSF) for MuSc anomaly detection.

This module implements saliency-based adaptive weighting for multi-scale
anomaly map fusion. The core insight is that different scales have different
sensitivities to different anomaly types:
- Small r (r=1): Sensitive to fine texture anomalies, but also noisy
- Large r (r=5): Sensitive to structural anomalies, lower resolution
- Medium r (r=3): Balanced

The adaptive weighting is based on response saliency - scales with more
pronounced "peaks" (high contrast responses) receive higher weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_EPS = 1e-8


@dataclass
class AMSFConfig:
    """Configuration for Adaptive Multi-Scale Fusion.
    
    Attributes:
        enabled: Whether AMSF is enabled.
        weighting_method: Method to compute adaptive weights.
            - 'sharpness': Peak sharpness (std/mean of top-k), favors localized responses
            - 'contrast': Peak-to-background ratio, robust measure
            - 'gradient': Edge sharpness, favors sharp boundaries
            - 'entropy': Response distribution entropy
            - 'kurtosis': Response distribution kurtosis
            - 'consistency': Legacy cross-scale correlation method
            - 'sharpness_consistency': Sharpness scaled by cross-scale consistency
            - 'saliency': DEPRECATED - favors large scales
        temperature: Softmax temperature (lower = sharper weights).
        weight_floor: Minimum weight per scale.
        weight_ceil: Maximum weight per scale.
        saliency_percentile: Percentile for defining "peaks" (default 95).
        consistency_score_mix: Mixing factor for 'sharpness_consistency'.
            0 -> sharpness only, 1 -> sharpness * consistency.
        log_weights: Whether to log weights for analysis.

        calibration_enabled: Whether to calibrate the fused map distribution to a
            reference map (by default the equal-weight baseline mean map).
            Motivation: this repo's AUPRO is computed via global threshold sweep
            over the whole dataset (see utils/metrics.py), so a distribution shift
            in anomaly maps can significantly affect AUPRO.
        calibration_method: Calibration method.
            - 'mean_std': match per-image mean/std
            - 'bg_mean_std': match mean/std on background pixels (exclude top percentile)
            - 'quantile': match per-image low/high quantiles
            - 'tail_quantile': align only the upper tail (piecewise scaling)
        calibration_quantiles: (low, high) quantiles used by 'quantile' method.
        calibration_bg_percentile: Percentile used to define background pixels for
            'bg_mean_std' method. Pixels with reference values above this percentile
            are treated as potential anomalies and excluded from stats.
        calibration_consistency_threshold: If > 0 and consistency gating is enabled,
            only apply calibration when per-sample consistency is below this threshold
            (i.e., low-confidence samples).
        calibration_gate_threshold: If > 0 and consistency gating is enabled, only
            apply calibration when per-sample gate is below this threshold.

        calibration_tail_anchor_quantile: Anchor quantile (q0) for 'tail_quantile'.
            Only values above q0 are rescaled.
        calibration_tail_quantile: Tail quantile (q1) for 'tail_quantile'. Values at
            q1 are aligned to reference q1.
        calibration_tail_scale_min/max: Clip range for the tail scaling factor.
        calibration_tail_clamp_to_ref: If true, clamp the calibrated tail to the
            reference tail quantile (q1) for extra robustness.
        calibration_drift_threshold: If > 0, apply calibration only when the
            fused map distribution drifts from the reference by this threshold.

        shrink_to_uniform: Optional convex mixing toward uniform weights to
            reduce per-image weight variance. 0 disables.
        shrink_to_uniform_adaptive: Enable confidence-based shrink to uniform.
        shrink_to_uniform_min/max: Min/max shrink when adaptive is enabled.
        shrink_to_uniform_metric: Confidence metric ('margin' | 'entropy').
        shrink_to_uniform_margin_low/high: Margin thresholds for confidence.
        shrink_to_uniform_entropy_low/high: Entropy thresholds for confidence.

        post_smooth_kernel: Optional odd kernel size for spatial smoothing on the
            fused patch map (avg pooling). <= 1 disables.
        post_smooth_strength: Blending factor in [0,1]. 0 disables.
        post_smooth_keep_top_quantile: If in (0,1), keep the top-q patches from
            the original fused map (tail protection), and only smooth the rest.

        patch_gate_enabled: Whether to apply a patch-wise reliability gate on
            the fused residual `(fused - reference)`.
        patch_gate_method: Patch gate method.
            - 'adjacent_agreement': reliability = max over adjacent scales of
              `(1 - |m_i - m_{i+1}|) * mean(m_i, m_{i+1})`, where `m` are per-scale
              normalized maps in [0, 1].
        patch_gate_tau/beta: Sigmoid parameters for the patch gate.
        patch_gate_floor: Minimum patch gate value (lower bound).
        patch_gate_max: Maximum patch gate value (upper bound). Values > 1 enable
            bidirectional gating (boost on high-reliability patches).
        patch_gate_smooth_kernel: Optional odd kernel size for spatial smoothing
            of the patch gate (avg pooling); <= 1 disables.
        patch_gate_protect_quantile: If in (0,1), force gate>=1 on patches whose
            reference value is above this per-image quantile (protect strong
            anomalies from being suppressed; keep boost if enabled).
        patch_gate_protect_source: Which map to use to build the protection mask
            for `patch_gate_protect_quantile`.
            - 'reference' (default): use the reference map (baseline mean).
            - 'fused': use the fused map *before* applying PatchGate (after the
              sample-level gate).
            - 'max_scale': use per-patch max over scales (protect if ANY scale is
              strongly activated).
        patch_gate_power: Optional power on patch_gate in [0,1] (>=1 makes it
            sharper). Applied before floor/clamp/smoothing.
        patch_gate_mode: Where to apply patch_gate.
            - 'all' (default): apply on all patches.
            - 'background': only apply on background patches (based on reference
              map percentile), and force gate=1 on non-background patches.
        patch_gate_bg_percentile: Background percentile on reference map used by
            patch_gate_mode='background'.
        patch_gate_consistency_threshold / patch_gate_gate_threshold: If > 0 and
            sample-level consistency gating is enabled, only apply patch gating
            when per-sample consistency / gate is below the threshold.
        patch_gate_apply_logic: How to combine the above conditions when both
            thresholds are set.
            - 'or' (default): apply PatchGate if ANY condition is met.
            - 'and': apply PatchGate only if ALL active conditions are met.
    """
    enabled: bool = False
    weighting_method: str = 'sharpness'  # 'sharpness' | 'contrast' | 'gradient' | 'entropy' | ...
    temperature: float = 0.1
    weight_floor: float = 0.1
    weight_ceil: float = 0.7
    saliency_percentile: float = 95.0
    consistency_score_mix: float = 0.5
    log_weights: bool = True

    calibration_enabled: bool = False
    calibration_method: str = 'mean_std'
    calibration_quantiles: Tuple[float, float] = (0.1, 0.9)
    calibration_bg_percentile: float = 0.9
    calibration_consistency_threshold: float = -1.0
    calibration_gate_threshold: float = -1.0

    calibration_tail_anchor_quantile: float = 0.9
    calibration_tail_quantile: float = 0.99
    calibration_tail_scale_min: float = 0.5
    calibration_tail_scale_max: float = 2.0
    calibration_tail_clamp_to_ref: bool = False
    calibration_drift_threshold: float = -1.0

    shrink_to_uniform: float = 0.0
    shrink_to_uniform_adaptive: bool = False
    shrink_to_uniform_min: float = 0.0
    shrink_to_uniform_max: float = 0.0
    shrink_to_uniform_metric: str = "margin"
    shrink_to_uniform_margin_low: float = 0.05
    shrink_to_uniform_margin_high: float = 0.25
    shrink_to_uniform_entropy_low: float = 0.3
    shrink_to_uniform_entropy_high: float = 0.7

    post_smooth_kernel: int = 0
    post_smooth_strength: float = 0.0
    post_smooth_keep_top_quantile: float = -1.0

    consistency_gate_enabled: bool = False
    consistency_method: str = "topk_overlap"  # 'topk_overlap' | 'spearman' | 'cosine'
    consistency_topk_ratio: float = 0.01
    consistency_tau: float = 0.5
    consistency_beta: float = 10.0
    consistency_gate_hard_floor: float = 0.0

    patch_gate_enabled: bool = False
    patch_gate_method: str = "adjacent_agreement"
    patch_gate_tau: float = 0.3
    patch_gate_beta: float = 10.0
    patch_gate_floor: float = 0.0
    patch_gate_max: float = 1.0
    patch_gate_smooth_kernel: int = 0
    patch_gate_protect_quantile: float = -1.0
    patch_gate_protect_source: str = "reference"
    patch_gate_power: float = 1.0
    patch_gate_mode: str = "all"
    patch_gate_bg_percentile: float = 0.9
    patch_gate_consistency_threshold: float = -1.0
    patch_gate_gate_threshold: float = -1.0
    patch_gate_apply_logic: str = "or"

    @classmethod
    def from_dict(cls, cfg: Dict) -> "AMSFConfig":
        if not cfg:
            return cls()

        # Parse quantiles robustly.
        q_raw = cfg.get("calibration_quantiles", (0.1, 0.9))
        q0, q1 = 0.1, 0.9
        try:
            if isinstance(q_raw, (list, tuple)) and len(q_raw) >= 2:
                q0, q1 = float(q_raw[0]), float(q_raw[1])
        except Exception:
            q0, q1 = 0.1, 0.9

        # Background percentile for bg_mean_std calibration.
        bg_p = cfg.get("calibration_bg_percentile", 0.9)
        try:
            bg_p = float(bg_p)
        except Exception:
            bg_p = 0.9

        cal_cons_th = cfg.get("calibration_consistency_threshold", -1.0)
        try:
            cal_cons_th = float(cal_cons_th)
        except Exception:
            cal_cons_th = -1.0

        cal_gate_th = cfg.get("calibration_gate_threshold", -1.0)
        try:
            cal_gate_th = float(cal_gate_th)
        except Exception:
            cal_gate_th = -1.0

        tail_anchor_q = cfg.get("calibration_tail_anchor_quantile", 0.9)
        tail_q = cfg.get("calibration_tail_quantile", 0.99)
        tail_scale_min = cfg.get("calibration_tail_scale_min", 0.5)
        tail_scale_max = cfg.get("calibration_tail_scale_max", 2.0)
        try:
            tail_anchor_q = float(tail_anchor_q)
        except Exception:
            tail_anchor_q = 0.9
        try:
            tail_q = float(tail_q)
        except Exception:
            tail_q = 0.99
        try:
            tail_scale_min = float(tail_scale_min)
        except Exception:
            tail_scale_min = 0.5
        try:
            tail_scale_max = float(tail_scale_max)
        except Exception:
            tail_scale_max = 2.0
        tail_clamp = bool(cfg.get("calibration_tail_clamp_to_ref", False))

        drift_th = cfg.get("calibration_drift_threshold", -1.0)
        try:
            drift_th = float(drift_th)
        except Exception:
            drift_th = -1.0

        shrink_min = cfg.get("shrink_to_uniform_min", cfg.get("shrink_to_uniform", 0.0))
        shrink_max = cfg.get("shrink_to_uniform_max", cfg.get("shrink_to_uniform", 0.0))
        try:
            shrink_min = float(shrink_min)
        except Exception:
            shrink_min = 0.0
        try:
            shrink_max = float(shrink_max)
        except Exception:
            shrink_max = 0.0

        shrink_margin_low = cfg.get("shrink_to_uniform_margin_low", 0.05)
        shrink_margin_high = cfg.get("shrink_to_uniform_margin_high", 0.25)
        shrink_entropy_low = cfg.get("shrink_to_uniform_entropy_low", 0.3)
        shrink_entropy_high = cfg.get("shrink_to_uniform_entropy_high", 0.7)
        try:
            shrink_margin_low = float(shrink_margin_low)
        except Exception:
            shrink_margin_low = 0.05
        try:
            shrink_margin_high = float(shrink_margin_high)
        except Exception:
            shrink_margin_high = 0.25
        try:
            shrink_entropy_low = float(shrink_entropy_low)
        except Exception:
            shrink_entropy_low = 0.3
        try:
            shrink_entropy_high = float(shrink_entropy_high)
        except Exception:
            shrink_entropy_high = 0.7

        post_k = cfg.get("post_smooth_kernel", 0)
        try:
            post_k = int(post_k)
        except Exception:
            post_k = 0
        post_strength = cfg.get("post_smooth_strength", 0.0)
        try:
            post_strength = float(post_strength)
        except Exception:
            post_strength = 0.0
        post_keep_q = cfg.get("post_smooth_keep_top_quantile", -1.0)
        try:
            post_keep_q = float(post_keep_q)
        except Exception:
            post_keep_q = -1.0

        # Patch-wise gate params.
        pg_tau = cfg.get("patch_gate_tau", 0.3)
        pg_beta = cfg.get("patch_gate_beta", 10.0)
        pg_floor = cfg.get("patch_gate_floor", 0.0)
        pg_max = cfg.get("patch_gate_max", 1.0)
        pg_kernel = cfg.get("patch_gate_smooth_kernel", 0)
        pg_protect_q = cfg.get("patch_gate_protect_quantile", -1.0)
        pg_protect_src = cfg.get("patch_gate_protect_source", "reference")
        pg_cons_th = cfg.get("patch_gate_consistency_threshold", -1.0)
        pg_gate_th = cfg.get("patch_gate_gate_threshold", -1.0)
        pg_apply_logic = cfg.get("patch_gate_apply_logic", "or")
        cons_mix = cfg.get("consistency_score_mix", 0.5)
        try:
            pg_tau = float(pg_tau)
        except Exception:
            pg_tau = 0.3
        try:
            pg_beta = float(pg_beta)
        except Exception:
            pg_beta = 10.0
        try:
            pg_floor = float(pg_floor)
        except Exception:
            pg_floor = 0.0
        try:
            pg_max = float(pg_max)
        except Exception:
            pg_max = 1.0
        if not np.isfinite(pg_max) or pg_max <= 0.0:
            pg_max = 1.0
        pg_max = max(1.0, min(10.0, pg_max))
        try:
            pg_kernel = int(pg_kernel)
        except Exception:
            pg_kernel = 0
        try:
            pg_protect_q = float(pg_protect_q)
        except Exception:
            pg_protect_q = -1.0
        try:
            cons_mix = float(cons_mix)
        except Exception:
            cons_mix = 0.5
        cons_mix = max(0.0, min(1.0, cons_mix))

        try:
            pg_protect_src = str(pg_protect_src).lower().strip()
        except Exception:
            pg_protect_src = "reference"
        if pg_protect_src not in {"reference", "fused", "max_scale"}:
            pg_protect_src = "reference"

        pg_power = cfg.get("patch_gate_power", 1.0)
        try:
            pg_power = float(pg_power)
        except Exception:
            pg_power = 1.0
        pg_mode = str(cfg.get("patch_gate_mode", "all")).lower().strip()
        if pg_mode in {"bg", "background", "bg_only", "background_only"}:
            pg_mode = "background"
        elif pg_mode not in {"all", "background"}:
            pg_mode = "all"
        pg_bg_p = cfg.get("patch_gate_bg_percentile", cfg.get("calibration_bg_percentile", 0.9))
        try:
            pg_bg_p = float(pg_bg_p)
        except Exception:
            pg_bg_p = 0.9
        if not (0.5 < pg_bg_p < 1.0):
            pg_bg_p = 0.9
        pg_bg_p = max(0.5, min(0.999, pg_bg_p))

        try:
            pg_cons_th = float(pg_cons_th)
        except Exception:
            pg_cons_th = -1.0
        try:
            pg_gate_th = float(pg_gate_th)
        except Exception:
            pg_gate_th = -1.0
        try:
            pg_apply_logic = str(pg_apply_logic).lower()
        except Exception:
            pg_apply_logic = "or"
        if pg_apply_logic not in ("or", "and"):
            pg_apply_logic = "or"

        return cls(
            enabled=bool(cfg.get("enabled", False)),
            weighting_method=str(cfg.get("weighting_method", "sharpness")).lower(),
            temperature=float(cfg.get("temperature", 0.1)),
            weight_floor=float(cfg.get("weight_floor", 0.1)),
            weight_ceil=float(cfg.get("weight_ceil", 0.7)),
            saliency_percentile=float(cfg.get("saliency_percentile", 95.0)),
            log_weights=bool(cfg.get("log_weights", True)),

            calibration_enabled=bool(cfg.get("calibration_enabled", False)),
            calibration_method=str(cfg.get("calibration_method", "mean_std")).lower(),
            calibration_quantiles=(q0, q1),
            calibration_bg_percentile=bg_p,
            calibration_consistency_threshold=cal_cons_th,
            calibration_gate_threshold=cal_gate_th,

            calibration_tail_anchor_quantile=tail_anchor_q,
            calibration_tail_quantile=tail_q,
            calibration_tail_scale_min=tail_scale_min,
            calibration_tail_scale_max=tail_scale_max,
            calibration_tail_clamp_to_ref=tail_clamp,
            calibration_drift_threshold=drift_th,

            shrink_to_uniform=float(cfg.get("shrink_to_uniform", 0.0)),
            shrink_to_uniform_adaptive=bool(cfg.get("shrink_to_uniform_adaptive", False)),
            shrink_to_uniform_min=shrink_min,
            shrink_to_uniform_max=shrink_max,
            shrink_to_uniform_metric=str(cfg.get("shrink_to_uniform_metric", "margin")).lower(),
            shrink_to_uniform_margin_low=shrink_margin_low,
            shrink_to_uniform_margin_high=shrink_margin_high,
            shrink_to_uniform_entropy_low=shrink_entropy_low,
            shrink_to_uniform_entropy_high=shrink_entropy_high,

            post_smooth_kernel=post_k,
            post_smooth_strength=post_strength,
            post_smooth_keep_top_quantile=post_keep_q,

            consistency_gate_enabled=bool(cfg.get("consistency_gate_enabled", False)),
            consistency_method=str(cfg.get("consistency_method", "topk_overlap")).lower(),
            consistency_topk_ratio=float(cfg.get("consistency_topk_ratio", 0.01)),
            consistency_tau=float(cfg.get("consistency_tau", 0.5)),
            consistency_beta=float(cfg.get("consistency_beta", 10.0)),
            consistency_gate_hard_floor=float(cfg.get("consistency_gate_hard_floor", 0.0)),

            patch_gate_enabled=bool(cfg.get("patch_gate_enabled", False)),
            patch_gate_method=str(cfg.get("patch_gate_method", "adjacent_agreement")).lower(),
            patch_gate_tau=pg_tau,
            patch_gate_beta=pg_beta,
            patch_gate_floor=pg_floor,
            patch_gate_max=pg_max,
            patch_gate_smooth_kernel=pg_kernel,
            patch_gate_protect_quantile=pg_protect_q,
            patch_gate_protect_source=pg_protect_src,
            patch_gate_power=pg_power,
            patch_gate_mode=pg_mode,
            patch_gate_bg_percentile=pg_bg_p,
            patch_gate_consistency_threshold=pg_cons_th,
            patch_gate_gate_threshold=pg_gate_th,
            patch_gate_apply_logic=pg_apply_logic,
            consistency_score_mix=cons_mix,
        )


class AdaptiveRFusion:
    """Adaptive multi-scale fusion based on response saliency.
    
    Core Insight:
    - Different scales capture different types of anomalies
    - A scale with more prominent "peaks" (high contrast) is more confident
    - Saliency = how much the top responses stand out from the background
    
    Methods:
    - saliency: Ratio of top-percentile mean to overall mean (recommended)
    - entropy: Negative entropy of normalized distribution
    - kurtosis: Excess kurtosis measuring tail heaviness
    - consistency: Legacy cross-scale correlation method
    """

    def __init__(self, config: AMSFConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.weight_history: List[Dict] = []
        self.gate_history: List[Dict] = []
        self.patch_gate_history: List[Dict] = []
        self.calibration_history: List[Dict] = []

    def reset_stats(self) -> None:
        """Reset accumulated statistics."""
        self.weight_history.clear()
        self.gate_history.clear()
        self.patch_gate_history.clear()
        self.calibration_history.clear()

    def _normalize_maps(self, maps: torch.Tensor) -> torch.Tensor:
        """Normalize each anomaly map to [0, 1] range."""
        if maps.dim() == 2:
            mins = maps.min(dim=1, keepdim=True).values
            maxs = maps.max(dim=1, keepdim=True).values
        else:
            flat = maps.view(maps.size(0), -1)
            mins = flat.min(dim=1, keepdim=True).values.view(-1, *([1] * (maps.dim() - 1)))
            maxs = flat.max(dim=1, keepdim=True).values.view(-1, *([1] * (maps.dim() - 1)))
        return (maps - mins) / (maxs - mins + _EPS)

    def _compute_saliency_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Compute saliency score based on peak prominence.
        
        **DEPRECATED**: This method favors large scales due to spatial smoothing.
        Use _compute_peak_sharpness_score instead.
        
        Args:
            anomaly_map: 1D tensor of anomaly values
            
        Returns:
            Saliency score (scalar tensor)
        """
        k = int(len(anomaly_map) * (100 - self.config.saliency_percentile) / 100)
        k = max(k, 1)
        
        topk_vals, _ = torch.topk(anomaly_map, k)
        top_mean = topk_vals.mean()
        overall_mean = anomaly_map.mean() + _EPS
        
        saliency = (top_mean / overall_mean) - 1.0
        return saliency

    def _compute_peak_sharpness_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Compute peak sharpness score - favors scales with more localized responses.
        
        Peak sharpness = std(top_k%) / mean(top_k%)
        
        Higher sharpness means more variation within the peak region,
        indicating the scale is capturing localized anomalies rather than
        broad diffuse responses.
        
        Args:
            anomaly_map: 1D tensor of anomaly values
            
        Returns:
            Peak sharpness score (scalar tensor)
        """
        k = int(len(anomaly_map) * (100 - self.config.saliency_percentile) / 100)
        k = max(k, 1)
        
        topk_vals, _ = torch.topk(anomaly_map, k)
        top_mean = topk_vals.mean() + _EPS
        # NOTE: torch.std default uses unbiased=True, which returns NaN when k==1.
        # Keep the historical behavior for k>=2, and be safe for k==1.
        if topk_vals.numel() <= 1:
            top_std = torch.zeros_like(top_mean)
        else:
            top_std = topk_vals.std(unbiased=True)
        
        # Coefficient of variation of peak region
        sharpness = top_std / top_mean
        return sharpness

    def _compute_contrast_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Compute contrast score - ratio of peak to background.
        
        Contrast = (max - median) / (median + eps)
        
        Higher contrast means clearer separation between anomaly and normal.
        This is more robust than saliency as it uses median instead of mean.
        
        Args:
            anomaly_map: 1D tensor of anomaly values
            
        Returns:
            Contrast score (scalar tensor)
        """
        max_val = anomaly_map.max()
        median_val = anomaly_map.median()
        
        contrast = (max_val - median_val) / (median_val + _EPS)
        return contrast

    def _compute_gradient_score(self, anomaly_map: torch.Tensor, hw_shape: tuple = None) -> torch.Tensor:
        """Compute gradient magnitude score - measures edge sharpness.
        
        Higher gradient magnitude means sharper transitions, indicating
        well-defined anomaly boundaries.
        
        Args:
            anomaly_map: 1D tensor of anomaly values
            hw_shape: Optional (H, W) shape for 2D gradient computation
            
        Returns:
            Gradient magnitude score (scalar tensor)
        """
        # Use sorted difference as proxy for gradient
        sorted_vals, _ = torch.sort(anomaly_map)
        diffs = sorted_vals[1:] - sorted_vals[:-1]
        
        # Top gradient magnitudes indicate sharp transitions
        k = max(len(diffs) // 20, 1)  # top 5%
        topk_diffs, _ = torch.topk(diffs, k)
        
        return topk_diffs.mean()

    def _compute_entropy_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Compute negative entropy score.
        
        Lower entropy = more peaked distribution = more confident.
        We return negative entropy so higher = better.
        
        Args:
            anomaly_map: 1D tensor of anomaly values
            
        Returns:
            Negative entropy score (scalar tensor)
        """
        # Normalize to probability distribution
        p = F.softmax(anomaly_map, dim=0)
        
        # Entropy
        entropy = -(p * torch.log(p + _EPS)).sum()
        
        # Return negative entropy (higher is better)
        return -entropy

    def _compute_kurtosis_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Compute excess kurtosis score.
        
        Higher kurtosis = heavier tails = more extreme values = peaks.
        
        Args:
            anomaly_map: 1D tensor of anomaly values
            
        Returns:
            Excess kurtosis score (scalar tensor)
        """
        mean = anomaly_map.mean()
        std = anomaly_map.std() + _EPS
        
        # Standardize
        z = (anomaly_map - mean) / std
        
        # Excess kurtosis (Fisher's definition)
        kurtosis = (z ** 4).mean() - 3.0
        
        return kurtosis

    def _compute_consistency_scores(self, maps: torch.Tensor) -> torch.Tensor:
        """Legacy: Compute cross-scale consistency scores (for comparison).

        Historically this repo used exactly 3 scales (r=[1,3,5]), but ALF fuses
        multiple backbone layers and may have 4+ inputs. Make this implementation
        robust to arbitrary `num_scales`.
        """
        num_scales = maps.size(0)
        if num_scales <= 1:
            return torch.ones((num_scales,), device=maps.device, dtype=maps.dtype)

        flat_maps = maps.view(num_scales, -1)

        def to_ranks(t: torch.Tensor) -> torch.Tensor:
            # Stable enough for our legacy usage; uses 0..P-1 ranks.
            sorted_indices = torch.argsort(t)
            ranks = torch.zeros_like(t)
            ranks[sorted_indices] = torch.arange(len(t), dtype=t.dtype, device=t.device)
            return ranks

        ranks = torch.stack([to_ranks(flat_maps[i]) for i in range(num_scales)], dim=0)
        ranks = ranks - ranks.mean(dim=1, keepdim=True)
        ranks = ranks / (ranks.std(dim=1, keepdim=True) + _EPS)

        # Pearson correlation on ranks == Spearman correlation.
        corr = (ranks @ ranks.t()) / float(ranks.shape[1])
        corr = corr.clamp(min=-1.0, max=1.0)

        # Per-scale confidence = mean correlation to other scales.
        conf = (corr.sum(dim=1) - 1.0) / float(num_scales - 1)
        return conf

    def _sanitize_topk_ratio(self) -> float:
        ratio = self.config.consistency_topk_ratio
        try:
            ratio = float(ratio)
        except Exception:
            return 0.01
        if ratio <= 0.0 or ratio >= 1.0:
            return 0.01
        return ratio

    def _compute_topk_overlap(self, maps_norm: torch.Tensor) -> torch.Tensor:
        """Compute average pairwise top-k Jaccard overlap across scales."""
        num_scales = maps_norm.size(0)
        if num_scales < 2:
            return torch.tensor(0.0, device=maps_norm.device)

        ratio = self._sanitize_topk_ratio()
        flat_maps = maps_norm.view(num_scales, -1)
        num_patches = flat_maps.size(1)
        k = max(1, int(num_patches * ratio))

        topk_indices = []
        for i in range(num_scales):
            _, idx = torch.topk(flat_maps[i], k)
            topk_indices.append(idx)

        overlaps = []
        for i in range(num_scales):
            for j in range(i + 1, num_scales):
                idx_i = topk_indices[i]
                idx_j = topk_indices[j]
                inter = (idx_i[:, None] == idx_j[None, :]).any(dim=1).float().sum()
                union = idx_i.numel() + idx_j.numel() - inter
                overlaps.append(inter / (union + _EPS))

        return torch.stack(overlaps).mean()

    def _compute_pairwise_similarity(self, maps_norm: torch.Tensor, method: str) -> torch.Tensor:
        """Compute average pairwise similarity for spearman/cosine methods."""
        num_scales = maps_norm.size(0)
        if num_scales < 2:
            return torch.tensor(0.0, device=maps_norm.device)

        flat_maps = maps_norm.view(num_scales, -1)

        def _spearman(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            def to_ranks(t: torch.Tensor) -> torch.Tensor:
                sorted_indices = torch.argsort(t)
                ranks = torch.zeros_like(t)
                ranks[sorted_indices] = torch.arange(len(t), dtype=t.dtype, device=t.device)
                return ranks
            rank_x, rank_y = to_ranks(x), to_ranks(y)
            mean_x, mean_y = rank_x.mean(), rank_y.mean()
            cov = ((rank_x - mean_x) * (rank_y - mean_y)).mean()
            return cov / ((rank_x.std() + _EPS) * (rank_y.std() + _EPS))

        def _cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x_norm = x / (x.norm() + _EPS)
            y_norm = y / (y.norm() + _EPS)
            return (x_norm * y_norm).sum()

        sims = []
        for i in range(num_scales):
            for j in range(i + 1, num_scales):
                if method == "spearman":
                    sims.append(_spearman(flat_maps[i], flat_maps[j]))
                elif method == "cosine":
                    sims.append(_cosine(flat_maps[i], flat_maps[j]))
                else:
                    raise ValueError(f"Unknown consistency_method: {method}")

        sim = torch.stack(sims).mean()
        return (sim + 1.0) / 2.0

    def _compute_consistency_gate(self, maps_norm: torch.Tensor) -> torch.Tensor:
        _, gate = self._compute_consistency_and_gate(maps_norm)
        return gate

    def _compute_consistency_and_gate(self, maps_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        method = self.config.consistency_method
        if method == "topk_overlap":
            consistency = self._compute_topk_overlap(maps_norm)
        elif method in ("spearman", "cosine"):
            consistency = self._compute_pairwise_similarity(maps_norm, method)
        else:
            raise ValueError(f"Unknown consistency_method: {method}")

        tau = float(self.config.consistency_tau)
        beta = float(self.config.consistency_beta)
        gate = torch.sigmoid(torch.tensor(beta, device=consistency.device) * (consistency - tau)).clamp(0.0, 1.0)
        hard_floor = float(getattr(self.config, "consistency_gate_hard_floor", 0.0))
        if hard_floor > 0.0:
            gate = torch.where(gate < hard_floor, torch.zeros_like(gate), gate)
        return consistency, gate

    def _should_apply_patch_gate(
        self, consistency: Optional[torch.Tensor], gate: Optional[torch.Tensor]
    ) -> bool:
        th_cons = float(getattr(self.config, "patch_gate_consistency_threshold", -1.0))
        th_gate = float(getattr(self.config, "patch_gate_gate_threshold", -1.0))
        if not self.config.consistency_gate_enabled or (th_cons <= 0.0 and th_gate <= 0.0):
            return True

        logic = str(getattr(self.config, "patch_gate_apply_logic", "or")).lower()
        if logic not in ("or", "and"):
            logic = "or"

        conds: List[bool] = []
        if th_cons > 0.0 and consistency is not None and torch.isfinite(consistency):
            conds.append(float(consistency.detach().cpu().item()) < th_cons)
        if th_gate > 0.0 and gate is not None and torch.isfinite(gate):
            conds.append(float(gate.detach().cpu().item()) < th_gate)
        if not conds:
            return False
        if logic == "and" and len(conds) >= 2:
            return all(conds)
        return any(conds)

    def _sanitize_patch_gate_kernel(self) -> int:
        k_raw = getattr(self.config, "patch_gate_smooth_kernel", 0)
        try:
            k = int(k_raw)
        except Exception:
            return 0
        if k <= 1:
            return 0
        # Use an odd kernel for symmetric padding.
        if k % 2 == 0:
            k += 1
        return max(0, min(31, k))

    def _sanitize_patch_gate_power(self) -> float:
        p_raw = getattr(self.config, "patch_gate_power", 1.0)
        try:
            p = float(p_raw)
        except Exception:
            return 1.0
        if not np.isfinite(p) or p <= 0.0:
            return 1.0
        return max(0.1, min(10.0, p))

    def _sanitize_patch_gate_bg_percentile(self) -> float:
        p_raw = getattr(self.config, "patch_gate_bg_percentile", getattr(self.config, "calibration_bg_percentile", 0.9))
        try:
            p = float(p_raw)
        except Exception:
            return 0.9
        if not (0.5 < p < 1.0):
            return 0.9
        return max(0.5, min(0.999, p))

    def _sanitize_patch_gate_protect_source(self) -> str:
        src = str(getattr(self.config, "patch_gate_protect_source", "reference")).lower().strip()
        if src not in {"reference", "fused", "max_scale"}:
            return "reference"
        return src

    def _select_patch_gate_protect_map(
        self,
        *,
        reference: torch.Tensor,
        fused_pre_patch_gate: torch.Tensor,
        raw_maps: torch.Tensor,
    ) -> torch.Tensor:
        src = self._sanitize_patch_gate_protect_source()
        if src == "fused":
            return fused_pre_patch_gate
        if src == "max_scale":
            return raw_maps.max(dim=0).values
        return reference

    def _smooth_patch_gate(self, gate: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if kernel_size <= 1:
            return gate
        if gate.dim() == 1:
            gate_b = gate.unsqueeze(0)
            squeeze = True
        else:
            gate_b = gate
            squeeze = False

        N, P = gate_b.shape
        H = int(np.sqrt(P))
        if H * H != P:
            return gate

        gate_2d = gate_b.view(N, 1, H, H)
        gate_2d = F.avg_pool2d(gate_2d, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        gate_sm = gate_2d.view(N, P)
        return gate_sm[0] if squeeze else gate_sm

    def _apply_patch_gate_protection(
        self, gate: torch.Tensor, protect_map: torch.Tensor, quantile: float
    ) -> torch.Tensor:
        if not (0.0 < quantile < 1.0):
            return gate
        if gate.dim() == 1:
            gate_b = gate.unsqueeze(0)
            ref_b = protect_map.unsqueeze(0) if protect_map.dim() == 1 else protect_map
            squeeze = True
        else:
            gate_b = gate
            ref_b = protect_map
            squeeze = False
        if ref_b.shape != gate_b.shape:
            return gate

        try:
            th = torch.quantile(ref_b, quantile, dim=1, keepdim=True)
        except Exception:
            return gate
        mask = ref_b >= th
        ones = torch.ones_like(gate_b)
        # Protect strong anomalies from being suppressed; keep boost if enabled.
        gate_b = torch.where(mask, torch.maximum(gate_b, ones), gate_b)
        return gate_b[0] if squeeze else gate_b

    def _compute_patch_consistency_score(self, maps_norm: torch.Tensor) -> torch.Tensor:
        """Compute patch-wise reliability score from normalized maps.

        This is designed to be more backbone-agnostic than CLS→Patch fusion:
        it operates purely on anomaly maps across r.
        """
        if maps_norm.dim() == 2:
            maps = maps_norm.unsqueeze(1)  # (R, 1, P)
            squeeze = True
        elif maps_norm.dim() == 3:
            maps = maps_norm  # (R, N, P)
            squeeze = False
        else:
            raise ValueError(f"Expected maps_norm dim 2 or 3, got {maps_norm.dim()}")

        num_scales, N, P = maps.shape
        if num_scales < 2:
            score = torch.ones((N, P), device=maps.device, dtype=maps.dtype)
            return score[0] if squeeze else score

        m0 = maps[:-1]
        m1 = maps[1:]
        diff = (m0 - m1).abs()
        agreement = (1.0 - diff).clamp(0.0, 1.0)
        magnitude = 0.5 * (m0 + m1)
        pair_score = agreement * magnitude
        score = pair_score.max(dim=0).values
        return score[0] if squeeze else score

    def _compute_patch_gate(
        self,
        maps_norm: torch.Tensor,
        reference: Optional[torch.Tensor],
        protect_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        method = str(getattr(self.config, "patch_gate_method", "adjacent_agreement")).lower()
        if method != "adjacent_agreement":
            raise ValueError(f"Unknown patch_gate_method: {method}")

        score = self._compute_patch_consistency_score(maps_norm)
        tau = float(getattr(self.config, "patch_gate_tau", 0.3))
        beta = float(getattr(self.config, "patch_gate_beta", 10.0))
        gate = torch.sigmoid(
            torch.tensor(beta, device=score.device, dtype=score.dtype) * (score - tau)
        ).clamp(0.0, 1.0)

        power = self._sanitize_patch_gate_power()
        if abs(power - 1.0) > 1e-6:
            gate = gate.pow(power)

        floor = float(getattr(self.config, "patch_gate_floor", 0.0))
        max_v = float(getattr(self.config, "patch_gate_max", 1.0))
        if not np.isfinite(max_v) or max_v <= 0.0:
            max_v = 1.0
        max_v = max(1.0, min(10.0, max_v))
        if floor > 0.0 or abs(max_v - 1.0) > 1e-6:
            floor = max(0.0, min(1.0, float(floor)))
            if max_v < floor:
                max_v = floor
            gate = floor + (max_v - floor) * gate

        kernel_size = self._sanitize_patch_gate_kernel()
        if kernel_size > 1:
            gate = self._smooth_patch_gate(gate, kernel_size)

        protect_q = float(getattr(self.config, "patch_gate_protect_quantile", -1.0))
        if 0.0 < protect_q < 1.0:
            if protect_map is None:
                protect_map = reference
            if protect_map is not None:
                gate = self._apply_patch_gate_protection(gate, protect_map, protect_q)

        pg_mode = str(getattr(self.config, "patch_gate_mode", "all")).lower().strip()
        # Background mask should be computed on the reference map (baseline mean).
        # `protect_map` is reserved for protection only; mixing them here can make the
        # background definition unstable across datasets/classes.
        mask_map = reference if reference is not None else protect_map
        if mask_map is not None and pg_mode in {"bg", "background", "bg_only", "background_only"}:
            p_bg = self._sanitize_patch_gate_bg_percentile()
            if gate.dim() == 1:
                gate_b = gate.unsqueeze(0)
                ref_b = mask_map.unsqueeze(0) if mask_map.dim() == 1 else mask_map
                squeeze = True
            else:
                gate_b = gate
                ref_b = mask_map
                squeeze = False
            if ref_b.shape == gate_b.shape:
                try:
                    th = torch.quantile(ref_b, p_bg, dim=1, keepdim=True)
                    bg_mask = ref_b <= th
                    gate_b = torch.where(bg_mask, gate_b, torch.ones_like(gate_b))
                    gate = gate_b[0] if squeeze else gate_b
                except Exception:
                    pass
        return gate

    def _sanitize_quantiles(self) -> Tuple[float, float]:
        q0, q1 = self.config.calibration_quantiles
        try:
            q0 = float(q0)
            q1 = float(q1)
        except Exception:
            return (0.1, 0.9)
        if not (0.0 < q0 < q1 < 1.0):
            return (0.1, 0.9)
        return (q0, q1)

    def _sanitize_bg_percentile(self) -> float:
        p = float(getattr(self.config, "calibration_bg_percentile", 0.9))
        # Background should cover most pixels; keep it in a conservative range.
        if not (0.5 < p < 1.0):
            return 0.9
        return max(0.5, min(0.999, p))

    def _sanitize_tail_quantiles(self) -> Tuple[float, float]:
        q0 = float(getattr(self.config, "calibration_tail_anchor_quantile", 0.9))
        q1 = float(getattr(self.config, "calibration_tail_quantile", 0.99))
        # Keep a conservative, stable range.
        q0 = max(0.5, min(0.995, q0))
        q1 = max(0.8, min(0.999, q1))
        if not (0.0 < q0 < q1 < 1.0):
            return (0.9, 0.99)
        return (q0, q1)

    def _sanitize_tail_scale_clip(self) -> Tuple[float, float]:
        s_min = float(getattr(self.config, "calibration_tail_scale_min", 0.5))
        s_max = float(getattr(self.config, "calibration_tail_scale_max", 2.0))
        if not (s_min > 0.0 and s_max > 0.0):
            return (0.5, 2.0)
        if s_max < s_min:
            s_min, s_max = s_max, s_min
        s_min = max(0.05, min(10.0, s_min))
        s_max = max(0.05, min(10.0, s_max))
        if s_max < s_min:
            return (0.5, 2.0)
        return (s_min, s_max)

    def _align_tail_quantile_with_params(
        self, fused: torch.Tensor, reference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Piecewise tail alignment: keep lower part, rescale upper tail.

        For each sample, compute q0 (anchor) and q1 (tail) on fused, and q1 on reference.
        Values <= fused_q0 are unchanged. Values > fused_q0 are scaled so that fused_q1
        aligns to ref_q1.
        """
        if fused.shape != reference.shape:
            scale = torch.ones((fused.shape[0], 1), device=fused.device, dtype=fused.dtype) if fused.dim() > 1 else torch.ones((1, 1), device=fused.device, dtype=fused.dtype)
            return fused, scale

        q0, q1 = self._sanitize_tail_quantiles()
        s_min, s_max = self._sanitize_tail_scale_clip()
        clamp_to_ref = bool(getattr(self.config, "calibration_tail_clamp_to_ref", False))

        orig_shape = fused.shape
        if fused.dim() == 1:
            fused_flat = fused.view(1, -1)
            ref_flat = reference.view(1, -1)
        else:
            fused_flat = fused.reshape(fused.shape[0], -1)
            ref_flat = reference.reshape(reference.shape[0], -1)

        fused_q = fused_flat.float()
        ref_q = ref_flat.float()

        def _quantile(x: torch.Tensor, q: float) -> torch.Tensor:
            try:
                return torch.quantile(x, q, dim=1, keepdim=True)
            except TypeError:
                return torch.quantile(x, q, dim=1).unsqueeze(1)

        try:
            fused_q0 = _quantile(fused_q, q0)
            fused_q1 = _quantile(fused_q, q1)
            ref_q1 = _quantile(ref_q, q1)
        except Exception:
            scale = torch.ones((fused_flat.shape[0], 1), device=fused.device, dtype=fused.dtype)
            return fused, scale

        denom = (fused_q1 - fused_q0).clamp(min=_EPS)
        scale = (ref_q1 - fused_q0) / denom
        scale = scale.clamp(min=s_min, max=s_max)

        fused_q0_t = fused_q0.to(dtype=fused_flat.dtype)
        scale_t = scale.to(dtype=fused_flat.dtype)
        aligned_flat = torch.where(
            fused_flat <= fused_q0_t,
            fused_flat,
            fused_q0_t + (fused_flat - fused_q0_t) * scale_t,
        )
        if clamp_to_ref:
            ref_q1_t = ref_q1.to(dtype=fused_flat.dtype)
            aligned_flat = torch.minimum(aligned_flat, ref_q1_t)

        return aligned_flat.view(orig_shape), scale

    def _align_mean_std(self, fused: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Affine calibration: match per-sample mean/std to reference."""
        if fused.shape != reference.shape:
            return fused

        # Support (P,), (N,P) and (N,*,*) by flattening the non-batch dims.
        orig_shape = fused.shape
        if fused.dim() == 1:
            fused_flat = fused.view(1, -1)
            ref_flat = reference.view(1, -1)
        else:
            fused_flat = fused.reshape(fused.shape[0], -1)
            ref_flat = reference.reshape(reference.shape[0], -1)

        fused_mean = fused_flat.mean(dim=1, keepdim=True)
        fused_std = fused_flat.std(dim=1, keepdim=True, unbiased=False) + _EPS
        ref_mean = ref_flat.mean(dim=1, keepdim=True)
        ref_std = ref_flat.std(dim=1, keepdim=True, unbiased=False) + _EPS

        aligned_flat = (fused_flat - fused_mean) * (ref_std / fused_std) + ref_mean
        return aligned_flat.view(orig_shape)

    def _align_mean_std_with_params(
        self, fused: torch.Tensor, reference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if fused.shape != reference.shape:
            scale = torch.ones((fused.shape[0], 1), device=fused.device, dtype=fused.dtype) if fused.dim() > 1 else torch.ones((1, 1), device=fused.device, dtype=fused.dtype)
            shift = torch.zeros_like(scale)
            return fused, scale, shift

        orig_shape = fused.shape
        if fused.dim() == 1:
            fused_flat = fused.view(1, -1)
            ref_flat = reference.view(1, -1)
        else:
            fused_flat = fused.reshape(fused.shape[0], -1)
            ref_flat = reference.reshape(reference.shape[0], -1)

        fused_mean = fused_flat.mean(dim=1, keepdim=True)
        fused_std = fused_flat.std(dim=1, keepdim=True, unbiased=False) + _EPS
        ref_mean = ref_flat.mean(dim=1, keepdim=True)
        ref_std = ref_flat.std(dim=1, keepdim=True, unbiased=False) + _EPS

        scale = ref_std / fused_std
        shift = ref_mean - fused_mean * scale
        aligned_flat = fused_flat * scale + shift
        return aligned_flat.view(orig_shape), scale, shift

    def _align_bg_mean_std_with_params(
        self, fused: torch.Tensor, reference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Affine calibration on background pixels: match mean/std to reference.

        Background pixels are defined by reference <= quantile(reference, p_bg).
        Returns (aligned, scale, shift, bg_count).
        """
        if fused.shape != reference.shape:
            scale = torch.ones((fused.shape[0], 1), device=fused.device, dtype=fused.dtype) if fused.dim() > 1 else torch.ones((1, 1), device=fused.device, dtype=fused.dtype)
            shift = torch.zeros_like(scale)
            bg_count = torch.zeros_like(scale)
            return fused, scale, shift, bg_count

        p_bg = self._sanitize_bg_percentile()

        orig_shape = fused.shape
        if fused.dim() == 1:
            fused_flat = fused.view(1, -1)
            ref_flat = reference.view(1, -1)
        else:
            fused_flat = fused.reshape(fused.shape[0], -1)
            ref_flat = reference.reshape(reference.shape[0], -1)

        num = fused_flat.shape[1]
        # kthvalue uses 1..num (k-th smallest)
        k = int(p_bg * float(num))
        k = max(1, min(num, k))

        try:
            thresh = ref_flat.kthvalue(k, dim=1, keepdim=True).values
        except Exception:
            # Conservative fallback.
            aligned, scale, shift = self._align_mean_std_with_params(fused, reference)
            bg_count = torch.full_like(scale, float(num))
            return aligned, scale, shift, bg_count

        mask = (ref_flat <= thresh).to(dtype=fused_flat.dtype)
        bg_count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        fused_mean = (fused_flat * mask).sum(dim=1, keepdim=True) / bg_count
        ref_mean = (ref_flat * mask).sum(dim=1, keepdim=True) / bg_count

        fused_var = ((fused_flat - fused_mean) ** 2 * mask).sum(dim=1, keepdim=True) / bg_count
        ref_var = ((ref_flat - ref_mean) ** 2 * mask).sum(dim=1, keepdim=True) / bg_count

        fused_std = fused_var.sqrt() + _EPS
        ref_std = ref_var.sqrt() + _EPS

        scale = ref_std / fused_std
        shift = ref_mean - fused_mean * scale
        aligned_flat = fused_flat * scale + shift
        return aligned_flat.view(orig_shape), scale, shift, bg_count

    def _align_quantile(self, fused: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Affine calibration: match low/high quantiles to reference."""
        if fused.shape != reference.shape:
            return fused

        q0, q1 = self._sanitize_quantiles()

        orig_shape = fused.shape
        if fused.dim() == 1:
            fused_flat = fused.view(1, -1)
            ref_flat = reference.view(1, -1)
        else:
            fused_flat = fused.reshape(fused.shape[0], -1)
            ref_flat = reference.reshape(reference.shape[0], -1)

        # quantile can be picky across torch versions / dtypes; compute in float32.
        fused_q = fused_flat.float()
        ref_q = ref_flat.float()

        def _quantile(x: torch.Tensor, q: float) -> torch.Tensor:
            try:
                return torch.quantile(x, q, dim=1, keepdim=True)
            except TypeError:
                # Older torch may not support keepdim
                return torch.quantile(x, q, dim=1).unsqueeze(1)

        try:
            fused_low = _quantile(fused_q, q0)
            fused_high = _quantile(fused_q, q1)
            ref_low = _quantile(ref_q, q0)
            ref_high = _quantile(ref_q, q1)
        except Exception:
            # Conservative fallback.
            return self._align_mean_std(fused, reference)

        scale = (ref_high - ref_low) / (fused_high - fused_low + _EPS)
        aligned_q = (fused_q - fused_low) * scale + ref_low
        aligned = aligned_q.to(dtype=fused_flat.dtype)
        return aligned.view(orig_shape)

    def _compute_calibration_drift(self, fused: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute distribution drift between fused and reference maps."""
        if fused.shape != reference.shape:
            return torch.tensor(0.0, device=fused.device)

        if fused.dim() == 1:
            fused_b = fused.view(1, -1)
            ref_b = reference.view(1, -1)
        else:
            fused_b = fused.reshape(fused.shape[0], -1)
            ref_b = reference.reshape(reference.shape[0], -1)

        mean_f = fused_b.mean(dim=1)
        std_f = fused_b.std(dim=1)
        mean_r = ref_b.mean(dim=1)
        std_r = ref_b.std(dim=1)
        denom = std_r + _EPS
        drift = (mean_f - mean_r).abs() / denom + (std_f - std_r).abs() / denom
        return drift

    def _calibrate_to_reference(
        self,
        fused: torch.Tensor,
        reference: Optional[torch.Tensor],
        *,
        record_stats: bool = False,
        skip_drift: bool = False,
    ) -> torch.Tensor:
        if not self.config.calibration_enabled:
            return fused
        if reference is None:
            return fused
        if fused.shape != reference.shape:
            return fused

        if not skip_drift:
            drift_th = float(getattr(self.config, "calibration_drift_threshold", -1.0))
            if drift_th > 0.0:
                drift = self._compute_calibration_drift(fused, reference)
                if fused.dim() == 1:
                    if float(drift.item()) < drift_th:
                        return fused
                else:
                    mask = drift.view(-1) >= drift_th
                    if not mask.any():
                        return fused
                    if not mask.all():
                        fused_out = fused.clone()
                        fused_out[mask] = self._calibrate_to_reference(
                            fused[mask], reference[mask], record_stats=record_stats, skip_drift=True
                        )
                        return fused_out

        # Be defensive about device/dtype mismatch when fuse() is called externally.
        if reference.device != fused.device or reference.dtype != fused.dtype:
            reference = reference.to(device=fused.device, dtype=fused.dtype)

        method = str(self.config.calibration_method).lower().strip()
        if method in ("none", "off", "false"):
            return fused
        if method in ("mean_std", "mean-std", "meanstd"):
            aligned, scale, shift = self._align_mean_std_with_params(fused, reference)
            if record_stats and self.config.log_weights:
                scales = scale.detach().view(-1).cpu().numpy().tolist()
                shifts = shift.detach().view(-1).cpu().numpy().tolist()
                for s, b in zip(scales, shifts):
                    self.calibration_history.append(
                        {"method": "mean_std", "scale": float(s), "shift": float(b)}
                    )
            return aligned
        if method in ("bg_mean_std", "bg-mean-std", "bgmeanstd"):
            aligned, scale, shift, bg_count = self._align_bg_mean_std_with_params(fused, reference)
            if record_stats and self.config.log_weights:
                scales = scale.detach().view(-1).cpu().numpy().tolist()
                shifts = shift.detach().view(-1).cpu().numpy().tolist()
                counts = bg_count.detach().view(-1).cpu().numpy().tolist()
                p_bg = self._sanitize_bg_percentile()
                for s, b, c in zip(scales, shifts, counts):
                    self.calibration_history.append(
                        {"method": "bg_mean_std", "scale": float(s), "shift": float(b), "bg_p": float(p_bg), "bg_count": float(c)}
                    )
            return aligned
        if method in ("tail_quantile", "tail-quantile", "tailquantile", "tail_q", "tailq"):
            aligned, scale = self._align_tail_quantile_with_params(fused, reference)
            if record_stats and self.config.log_weights:
                scales = scale.detach().view(-1).cpu().numpy().tolist()
                q0, q1 = self._sanitize_tail_quantiles()
                for s in scales:
                    self.calibration_history.append(
                        {"method": "tail_quantile", "scale": float(s), "shift": 0.0, "tail_q0": float(q0), "tail_q1": float(q1)}
                    )
            return aligned
        if method in ("quantile", "quantiles"):
            return self._align_quantile(fused, reference)
        raise ValueError(f"Unknown calibration_method: {self.config.calibration_method}")

    def _compute_shrink_confidence(self, weights: torch.Tensor) -> float:
        """Compute confidence for adaptive shrink-to-uniform."""
        num_scales = weights.numel()
        if num_scales <= 1:
            return 1.0

        metric = str(getattr(self.config, "shrink_to_uniform_metric", "margin")).lower()
        if metric == "entropy":
            p = weights.clamp(min=_EPS)
            h = -(p * torch.log(p)).sum() / float(np.log(num_scales))
            conf_raw = 1.0 - h
            low = float(getattr(self.config, "shrink_to_uniform_entropy_low", 0.3))
            high = float(getattr(self.config, "shrink_to_uniform_entropy_high", 0.7))
        else:
            w_sorted, _ = torch.sort(weights, descending=True)
            if num_scales >= 2:
                margin = w_sorted[0] - w_sorted[1]
            else:
                margin = torch.tensor(1.0, device=weights.device)
            conf_raw = margin
            low = float(getattr(self.config, "shrink_to_uniform_margin_low", 0.05))
            high = float(getattr(self.config, "shrink_to_uniform_margin_high", 0.25))

        if high <= low:
            return 0.0
        conf = (conf_raw - low) / (high - low)
        conf = conf.clamp(0.0, 1.0)
        return float(conf.detach().cpu().item())

    def _post_smooth_fused(self, fused: torch.Tensor) -> torch.Tensor:
        """Apply optional spatial smoothing on the fused patch map.

        This is designed to be train-free and baseline-preserving (strength=0 is
        an exact no-op). Tail protection keeps top-q patches unchanged.
        """
        k = int(getattr(self.config, "post_smooth_kernel", 0))
        strength = float(getattr(self.config, "post_smooth_strength", 0.0))
        if k <= 1 or strength <= 0.0:
            return fused
        strength = max(0.0, min(1.0, strength))

        # Only support square patch grids.
        if fused.dim() == 1:
            num_patches = int(fused.numel())
        else:
            num_patches = int(fused.shape[-1])
        h = int(np.sqrt(num_patches))
        if h * h != num_patches:
            return fused

        # Use an odd kernel for symmetric padding.
        if k % 2 == 0:
            k = k + 1
        pad = k // 2

        orig_shape = fused.shape
        if fused.dim() == 1:
            fused_2d = fused.view(1, 1, h, h)
        else:
            fused_2d = fused.view(fused.shape[0], 1, h, h)

        # Average pooling smoothing.
        smoothed = F.avg_pool2d(fused_2d, kernel_size=k, stride=1, padding=pad)
        out_2d = fused_2d * (1.0 - strength) + smoothed * strength

        keep_q = float(getattr(self.config, "post_smooth_keep_top_quantile", -1.0))
        if 0.0 < keep_q < 1.0:
            fused_flat = fused_2d.view(fused_2d.shape[0], -1)
            out_flat = out_2d.view(out_2d.shape[0], -1)

            fused_q = fused_flat.float()
            try:
                t = torch.quantile(fused_q, keep_q, dim=1, keepdim=True)
            except TypeError:
                t = torch.quantile(fused_q, keep_q, dim=1).unsqueeze(1)
            mask = fused_flat >= t.to(dtype=fused_flat.dtype)
            out_flat = torch.where(mask, fused_flat, out_flat)
            out_2d = out_flat.view_as(out_2d)

        return out_2d.view(orig_shape)

    def compute_weights(self, anomaly_maps_r: torch.Tensor) -> torch.Tensor:
        """Compute adaptive weights for multi-scale fusion.
        
        Args:
            anomaly_maps_r: (num_scales, num_patches) anomaly maps
            
        Returns:
            weights: (num_scales,) tensor summing to 1
        """
        num_scales = anomaly_maps_r.size(0)
        flat_maps = anomaly_maps_r.view(num_scales, -1)
        
        # Normalize maps for fair comparison
        maps_norm = self._normalize_maps(flat_maps)
        
        # Compute scores based on selected method
        method = self.config.weighting_method
        
        if method == 'saliency':
            # Original saliency (deprecated - favors large scales)
            scores = torch.stack([
                self._compute_saliency_score(maps_norm[i])
                for i in range(num_scales)
            ])
        elif method == 'sharpness':
            # Peak sharpness - favors scales with localized responses
            scores = torch.stack([
                self._compute_peak_sharpness_score(maps_norm[i])
                for i in range(num_scales)
            ])
        elif method == 'contrast':
            # Contrast ratio - robust measure of peak vs background
            scores = torch.stack([
                self._compute_contrast_score(maps_norm[i])
                for i in range(num_scales)
            ])
        elif method == 'gradient':
            # Gradient magnitude - edge sharpness
            scores = torch.stack([
                self._compute_gradient_score(maps_norm[i])
                for i in range(num_scales)
            ])
        elif method == 'entropy':
            scores = torch.stack([
                self._compute_entropy_score(maps_norm[i])
                for i in range(num_scales)
            ])
        elif method == 'kurtosis':
            scores = torch.stack([
                self._compute_kurtosis_score(maps_norm[i])
                for i in range(num_scales)
            ])
        elif method == 'consistency':
            scores = self._compute_consistency_scores(maps_norm)
        elif method == 'sharpness_consistency':
            sharp_scores = torch.stack([
                self._compute_peak_sharpness_score(maps_norm[i])
                for i in range(num_scales)
            ])
            cons_scores = self._compute_consistency_scores(maps_norm)
            cons_scores = (cons_scores + 1.0) * 0.5
            cons_scores = cons_scores.clamp(min=0.0, max=1.0)
            mix = float(getattr(self.config, "consistency_score_mix", 0.5))
            mix = max(0.0, min(1.0, mix))
            scores = sharp_scores * ((1.0 - mix) + mix * cons_scores)
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        # Apply softmax with temperature
        weights = F.softmax(scores / self.config.temperature, dim=0)
        
        # Apply floor and ceiling constraints
        weights = weights.clamp(min=self.config.weight_floor, max=self.config.weight_ceil)
        
        # Re-normalize to sum to 1
        weights = weights / weights.sum()

        # Optional: shrink weights toward uniform to reduce per-image variance.
        shrink_conf: Optional[float] = None
        shrink_min = None
        shrink_max = None
        if bool(getattr(self.config, "shrink_to_uniform_adaptive", False)):
            shrink_min = float(getattr(self.config, "shrink_to_uniform_min", 0.0))
            shrink_max = float(getattr(self.config, "shrink_to_uniform_max", 0.0))
            shrink_min = max(0.0, min(1.0, shrink_min))
            shrink_max = max(0.0, min(1.0, shrink_max))
            if shrink_max < shrink_min:
                shrink_min, shrink_max = shrink_max, shrink_min
            shrink_conf = self._compute_shrink_confidence(weights)
            alpha = shrink_max - shrink_conf * (shrink_max - shrink_min)
        else:
            alpha = float(self.config.shrink_to_uniform)

        alpha = max(0.0, min(1.0, float(alpha)))
        if alpha > 0:
            uniform = torch.full_like(weights, 1.0 / num_scales)
            weights = weights * (1.0 - alpha) + uniform * alpha
            # Keep floor/ceil as hard constraints.
            weights = weights.clamp(min=self.config.weight_floor, max=self.config.weight_ceil)
            weights = weights / weights.sum()
        
        # Log weights
        if self.config.log_weights:
            record = {
                'weights': weights.detach().cpu().numpy().tolist(),
                'scores': scores.detach().cpu().numpy().tolist(),
                'method': method,
                'shrink_alpha': float(alpha),
            }
            if shrink_conf is not None:
                record.update({
                    'shrink_conf': float(shrink_conf),
                    'shrink_metric': str(getattr(self.config, "shrink_to_uniform_metric", "margin")),
                    'shrink_min': float(shrink_min if shrink_min is not None else 0.0),
                    'shrink_max': float(shrink_max if shrink_max is not None else 0.0),
                })
            self.weight_history.append(record)
        
        return weights

    def fuse(self, anomaly_maps_r: torch.Tensor, baseline_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform adaptive fusion of multi-scale anomaly maps.
        
        Args:
            anomaly_maps_r: (num_scales, num_patches) or (num_scales, N, num_patches)
            
        Returns:
            Fused anomaly map with shape (num_patches,) or (N, num_patches).
        """
        if not self.config.enabled:
            return anomaly_maps_r.mean(dim=0)
        
        # Handle batched input (num_scales, N, num_patches)
        if anomaly_maps_r.dim() == 3:
            num_scales, N, num_patches = anomaly_maps_r.shape
            results = []
            consistencies: List[float] = []
            gates: List[float] = []
            for i in range(N):
                sample_maps = anomaly_maps_r[:, i, :]
                weights = self.compute_weights(sample_maps)
                fused = (sample_maps * weights.view(-1, 1)).sum(dim=0)
                reference = baseline_map[i] if baseline_map is not None else sample_maps.mean(dim=0)
                maps_norm: Optional[torch.Tensor] = None
                consistency_t: Optional[torch.Tensor] = None
                gate_t: Optional[torch.Tensor] = None
                if self.config.consistency_gate_enabled:
                    maps_norm = self._normalize_maps(sample_maps)
                    consistency_t, gate_t = self._compute_consistency_and_gate(maps_norm)
                    fused = reference + gate_t * (fused - reference)
                    consistencies.append(float(consistency_t.detach().cpu().item()))
                    gates.append(float(gate_t.detach().cpu().item()))
                    if self.config.log_weights:
                        self.gate_history.append({
                            "gate": float(gate_t.detach().cpu().item()),
                            "consistency": float(consistency_t.detach().cpu().item()),
                            "consistency_method": str(self.config.consistency_method),
                            "consistency_tau": float(self.config.consistency_tau),
                            "consistency_beta": float(self.config.consistency_beta),
                            "consistency_topk_ratio": float(self._sanitize_topk_ratio()),
                            "consistency_gate_hard_floor": float(getattr(self.config, "consistency_gate_hard_floor", 0.0)),
                        })
                else:
                    consistencies.append(float("nan"))
                    gates.append(float("nan"))

                if bool(getattr(self.config, "patch_gate_enabled", False)):
                    if maps_norm is None:
                        maps_norm = self._normalize_maps(sample_maps)
                    if self._should_apply_patch_gate(consistency_t, gate_t):
                        protect_map = self._select_patch_gate_protect_map(
                            reference=reference,
                            fused_pre_patch_gate=fused,
                            raw_maps=sample_maps,
                        )
                        patch_gate = self._compute_patch_gate(
                            maps_norm, reference=reference, protect_map=protect_map
                        )
                        fused = reference + patch_gate * (fused - reference)
                        if self.config.log_weights:
                            self.patch_gate_history.append({
                                "patch_gate_mean": float(patch_gate.mean().detach().cpu().item()),
                                "patch_gate_std": float(patch_gate.std().detach().cpu().item()),
                                "patch_gate_min": float(patch_gate.min().detach().cpu().item()),
                                "patch_gate_max": float(patch_gate.max().detach().cpu().item()),
                            })
                results.append(fused)

            fused_batch = torch.stack(results, dim=0)
            if self.config.calibration_enabled:
                # Align fused distribution to a reference (default: equal-weight baseline).
                reference = baseline_map
                if reference is None:
                    reference = anomaly_maps_r.mean(dim=0)
                th_cons = float(getattr(self.config, "calibration_consistency_threshold", -1.0))
                th_gate = float(getattr(self.config, "calibration_gate_threshold", -1.0))
                if self.config.consistency_gate_enabled and (th_cons > 0.0 or th_gate > 0.0):
                    cons_t = torch.tensor(consistencies, device=fused_batch.device, dtype=torch.float32)
                    gate_t = torch.tensor(gates, device=fused_batch.device, dtype=torch.float32)
                    mask = torch.zeros((N,), device=fused_batch.device, dtype=torch.bool)
                    if th_cons > 0.0:
                        mask = mask | (torch.isfinite(cons_t) & (cons_t < th_cons))
                    if th_gate > 0.0:
                        mask = mask | (torch.isfinite(gate_t) & (gate_t < th_gate))
                    if mask.any():
                        fused_batch[mask] = self._calibrate_to_reference(
                            fused_batch[mask], reference[mask], record_stats=True
                        )
                else:
                    fused_batch = self._calibrate_to_reference(
                        fused_batch, reference, record_stats=True
                    )
            fused_batch = self._post_smooth_fused(fused_batch)
            return fused_batch
        
        # Single sample: (num_scales, num_patches)
        weights = self.compute_weights(anomaly_maps_r)
        fused = (anomaly_maps_r * weights.view(-1, 1)).sum(dim=0)
        consistency_val: Optional[float] = None
        gate_val: Optional[float] = None
        maps_norm_single: Optional[torch.Tensor] = None
        consistency_t: Optional[torch.Tensor] = None
        gate_t: Optional[torch.Tensor] = None
        if self.config.consistency_gate_enabled:
            reference = baseline_map if baseline_map is not None else anomaly_maps_r.mean(dim=0)
            maps_norm_single = self._normalize_maps(anomaly_maps_r)
            consistency_t, gate_t = self._compute_consistency_and_gate(maps_norm_single)
            fused = reference + gate_t * (fused - reference)
            consistency_val = float(consistency_t.detach().cpu().item())
            gate_val = float(gate_t.detach().cpu().item())
            if self.config.log_weights:
                self.gate_history.append({
                    "gate": float(gate_t.detach().cpu().item()),
                    "consistency": float(consistency_t.detach().cpu().item()),
                    "consistency_method": str(self.config.consistency_method),
                    "consistency_tau": float(self.config.consistency_tau),
                    "consistency_beta": float(self.config.consistency_beta),
                    "consistency_topk_ratio": float(self._sanitize_topk_ratio()),
                    "consistency_gate_hard_floor": float(getattr(self.config, "consistency_gate_hard_floor", 0.0)),
                })
        else:
            reference = baseline_map if baseline_map is not None else anomaly_maps_r.mean(dim=0)

        if bool(getattr(self.config, "patch_gate_enabled", False)):
            if maps_norm_single is None:
                maps_norm_single = self._normalize_maps(anomaly_maps_r)
            if self._should_apply_patch_gate(consistency_t, gate_t):
                protect_map = self._select_patch_gate_protect_map(
                    reference=reference,
                    fused_pre_patch_gate=fused,
                    raw_maps=anomaly_maps_r,
                )
                patch_gate = self._compute_patch_gate(
                    maps_norm_single, reference=reference, protect_map=protect_map
                )
                fused = reference + patch_gate * (fused - reference)
                if self.config.log_weights:
                    self.patch_gate_history.append({
                        "patch_gate_mean": float(patch_gate.mean().detach().cpu().item()),
                        "patch_gate_std": float(patch_gate.std().detach().cpu().item()),
                        "patch_gate_min": float(patch_gate.min().detach().cpu().item()),
                        "patch_gate_max": float(patch_gate.max().detach().cpu().item()),
                    })
        if self.config.calibration_enabled:
            reference = baseline_map
            if reference is None:
                reference = anomaly_maps_r.mean(dim=0)
            th_cons = float(getattr(self.config, "calibration_consistency_threshold", -1.0))
            th_gate = float(getattr(self.config, "calibration_gate_threshold", -1.0))
            if self.config.consistency_gate_enabled and (th_cons > 0.0 or th_gate > 0.0):
                apply = False
                if th_cons > 0.0 and consistency_val is not None and consistency_val < th_cons:
                    apply = True
                if th_gate > 0.0 and gate_val is not None and gate_val < th_gate:
                    apply = True
                if apply:
                    fused = self._calibrate_to_reference(fused, reference, record_stats=True)
            else:
                fused = self._calibrate_to_reference(fused, reference, record_stats=True)
        fused = self._post_smooth_fused(fused)
        return fused

    def get_weight_stats(self) -> Dict:
        """Get statistics about weight distribution across samples."""
        if (
            not self.weight_history
            and not self.gate_history
            and not self.patch_gate_history
            and not self.calibration_history
        ):
            return {}

        stats: Dict = {}
        if self.weight_history:
            all_weights = np.array([h["weights"] for h in self.weight_history])
            all_scores = np.array([h["scores"] for h in self.weight_history])
            stats.update({
                "mean_weights": all_weights.mean(axis=0).tolist(),
                "std_weights": all_weights.std(axis=0).tolist(),
                "min_weights": all_weights.min(axis=0).tolist(),
                "max_weights": all_weights.max(axis=0).tolist(),
                "mean_scores": all_scores.mean(axis=0).tolist(),
                "std_scores": all_scores.std(axis=0).tolist(),
                "num_samples": len(self.weight_history),
                "method": self.config.weighting_method,
            })

        if self.gate_history:
            all_gates = np.array([h["gate"] for h in self.gate_history], dtype=np.float64)
            all_cons = np.array([h["consistency"] for h in self.gate_history], dtype=np.float64)
            stats.update({
                "mean_gate": float(all_gates.mean()),
                "std_gate": float(all_gates.std()),
                "min_gate": float(all_gates.min()),
                "max_gate": float(all_gates.max()),
                "mean_consistency": float(all_cons.mean()),
                "std_consistency": float(all_cons.std()),
                "min_consistency": float(all_cons.min()),
                "max_consistency": float(all_cons.max()),
                "consistency_method": str(self.config.consistency_method),
                "consistency_tau": float(self.config.consistency_tau),
                "consistency_beta": float(self.config.consistency_beta),
                "consistency_topk_ratio": float(self._sanitize_topk_ratio()),
                "consistency_gate_hard_floor": float(getattr(self.config, "consistency_gate_hard_floor", 0.0)),
                "num_gate_samples": len(self.gate_history),
            })

        if self.calibration_history:
            scales = np.array([h.get("scale", 1.0) for h in self.calibration_history], dtype=np.float64)
            shifts = np.array([h.get("shift", 0.0) for h in self.calibration_history], dtype=np.float64)
            methods = [str(h.get("method", "")) for h in self.calibration_history]
            method = methods[0] if methods else ""
            stats.update({
                "mean_calib_scale": float(scales.mean()),
                "std_calib_scale": float(scales.std()),
                "min_calib_scale": float(scales.min()),
                "max_calib_scale": float(scales.max()),
                "mean_calib_shift": float(shifts.mean()),
                "std_calib_shift": float(shifts.std()),
                "min_calib_shift": float(shifts.min()),
                "max_calib_shift": float(shifts.max()),
                "calibration_method": method,
                "num_calib_samples": len(self.calibration_history),
            })
            if method == "bg_mean_std":
                p_bg = self._sanitize_bg_percentile()
                counts = np.array([h.get("bg_count", 0.0) for h in self.calibration_history], dtype=np.float64)
                stats.update({
                    "calibration_bg_percentile": float(p_bg),
                    "mean_bg_count": float(counts.mean()) if len(counts) else 0.0,
                })
            if method == "tail_quantile":
                q0 = self.calibration_history[0].get("tail_q0") if self.calibration_history else None
                q1 = self.calibration_history[0].get("tail_q1") if self.calibration_history else None
                if q0 is not None and q1 is not None:
                    stats.update({
                        "calibration_tail_q0": float(q0),
                        "calibration_tail_q1": float(q1),
                        "calibration_tail_clamp_to_ref": bool(getattr(self.config, "calibration_tail_clamp_to_ref", False)),
                        "calibration_tail_scale_min": float(getattr(self.config, "calibration_tail_scale_min", 0.5)),
                        "calibration_tail_scale_max": float(getattr(self.config, "calibration_tail_scale_max", 2.0)),
                    })

        if self.patch_gate_history:
            means = np.array([h.get("patch_gate_mean", np.nan) for h in self.patch_gate_history], dtype=np.float64)
            stds = np.array([h.get("patch_gate_std", np.nan) for h in self.patch_gate_history], dtype=np.float64)
            mins = np.array([h.get("patch_gate_min", np.nan) for h in self.patch_gate_history], dtype=np.float64)
            maxs = np.array([h.get("patch_gate_max", np.nan) for h in self.patch_gate_history], dtype=np.float64)
            stats.update({
                "mean_patch_gate": float(np.nanmean(means)),
                "std_patch_gate": float(np.nanmean(stds)),
                "min_patch_gate": float(np.nanmin(mins)),
                "max_patch_gate": float(np.nanmax(maxs)),
                "patch_gate_method": str(getattr(self.config, "patch_gate_method", "adjacent_agreement")),
                "patch_gate_tau": float(getattr(self.config, "patch_gate_tau", 0.3)),
                "patch_gate_beta": float(getattr(self.config, "patch_gate_beta", 10.0)),
                "patch_gate_floor": float(getattr(self.config, "patch_gate_floor", 0.0)),
                "patch_gate_max": float(getattr(self.config, "patch_gate_max", 1.0)),
                "patch_gate_smooth_kernel": int(getattr(self.config, "patch_gate_smooth_kernel", 0)),
                "patch_gate_protect_quantile": float(getattr(self.config, "patch_gate_protect_quantile", -1.0)),
                "patch_gate_protect_source": str(getattr(self.config, "patch_gate_protect_source", "reference")),
                "patch_gate_power": float(getattr(self.config, "patch_gate_power", 1.0)),
                "patch_gate_mode": str(getattr(self.config, "patch_gate_mode", "all")),
                "patch_gate_bg_percentile": float(self._sanitize_patch_gate_bg_percentile()),
                "patch_gate_consistency_threshold": float(getattr(self.config, "patch_gate_consistency_threshold", -1.0)),
                "patch_gate_gate_threshold": float(getattr(self.config, "patch_gate_gate_threshold", -1.0)),
                "patch_gate_apply_logic": str(getattr(self.config, "patch_gate_apply_logic", "or")),
                "num_patch_gate_samples": len(self.patch_gate_history),
            })

        return stats
