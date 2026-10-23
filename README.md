# ALF: Sharpness-Aware Adaptive Layer Fusion for Training-Free Anomaly Detection

[![ICIP 2026](https://img.shields.io/badge/ICIP-2026-00569C.svg)](https://2026.ieeeicip.org/)
[![Setting](https://img.shields.io/badge/Setting-Training--Free%20%7C%20Zero--Shot-teal.svg)](#)
[![Backbones](https://img.shields.io/badge/Backbones-CLIP%20%7C%20DINOv3-7048B8.svg)](#)

This is the official PyTorch implementation of **"ALF: Sharpness-Aware Adaptive Layer Fusion for Training-Free Anomaly Detection"** (ICIP 2026).

## 📢 News

- **[2026-05]** ALF is accepted by **IEEE ICIP 2026** 🎉
- **[2026-10]** Code released

## 📖 Overview

Training-free anomaly detection is attractive for industrial inspection with rapidly changing product lines, since it needs neither category-specific training data nor per-category model maintenance.
**MuSc** attains strong zero-shot performance via transductive mutual scoring on unlabeled test sets, but it fuses multi-layer anomaly maps with **fixed uniform weights**, discarding layer-wise reliability information.

**ALF** closes this gap with a sharpness-aware fusion module:

- Estimates **per-image, per-layer reliability weights** directly from the anomaly response distribution — layers with concentrated, peaky responses are up-weighted, diffuse ones are down-weighted.
- **No learnable parameters** and negligible computational overhead; fully preserves MuSc's training-free, transductive nature.
- Consistent gains on **MVTec AD, VisA, and BTAD** with both **CLIP** and **DINOv3** backbones.

## 🏆 Main Results

Mean AUPRO (%) of **MuSc vs. MuSc-ALF** under the token-aligned protocol (Protocol B: CLIP @ 518×518, DINOv3 @ 592×592). Best in **bold**.

| Backbone | Method | BTAD | MVTec AD | VisA |
|:---|:---|---:|---:|---:|
| CLIP ViT-L/14-336 | MuSc | 83.41 | 93.80 | 92.70 |
| CLIP ViT-L/14-336 | **MuSc-ALF (ours)** | **84.01** | **93.92** | **92.92** |
| DINOv3 ViT-L/16 | MuSc | 80.99 | 95.55 | 89.85 |
| DINOv3 ViT-L/16 | **MuSc-ALF (ours)** | **81.69** | **95.72** | **90.54** |

Zero-shot comparison with representative methods (mean AUPRO, %, Protocol B):

| Method | Setting | MVTec AD | VisA | BTAD |
|:---|:---:|---:|---:|---:|
| WinCLIP | 0-shot | 64.61 | 56.84 | – |
| APRIL-GAN | 0-shot | 44.01 | 86.80 | – |
| ACR | 0-shot | 72.70 | – | – |
| RD4AD | full | – | – | 79.56 |
| PatchCore | full | – | – | 76.91 |
| SimpleNet | full | – | – | 68.02 |
| MuSc (CLIP) | 0-shot | 93.80 | 92.70 | 83.41 |
| **MuSc-ALF (CLIP, ours)** | 0-shot | 93.92 | **92.92** | **84.01** |
| MuSc (DINOv3) | 0-shot | 95.55 | 89.85 | 80.99 |
| **MuSc-ALF (DINOv3, ours)** | 0-shot | **95.72** | 90.54 | 81.69 |

> Resolution-matched results (Protocol A, unified 448×448), the full weighting-method ablation (uniform / contrast / sharpness), and per-category analyses — the largest per-category gain reaches **+4.71 AUPRO** on VisA (DINOv3) — are available in the paper.

## 🛠️ Getting Started

### 1. Environment

```bash
# Option A: conda (recommended)
conda env create -f environment.yml
conda activate muscdinov3

# Option B: pip
pip install -r requirements.txt
```

### 2. Prepare Datasets

Download and organize the benchmarks under `./data/`:

```
data/
├── mvtec_anomaly_detection/   # MVTec AD  (https://www.mvtec.com/company/research/datasets/mvtec-ad)
├── visa/                      # VisA      (https://registry.opendata.aws/visa/)
└── btad/                      # BTAD      (https://avires.dimi.uniud.it/papers/btad/btad.zip)
```

### 3. Run

```bash
# MuSc-ALF on MVTec AD with CLIP @ 518 (Protocol B), matching the paper's main table
python examples/musc_main.py --config configs/ablation_alf_sharpcons_clip518_base.yaml
```

Edit the `datasets:` block of the config to switch dataset (`mvtec_ad` / `visa` / `btad`), backbone, and resolution.
To reproduce the uniform-fusion baseline, set `adaptive_layer_fusion.enabled: false`.

## 📁 Repository Structure

```
├── examples/musc_main.py        # Main entry point
├── configs/                     # Experiment configs (ALF variants, ablations, sweeps)
├── models/                      # Backbones, feature extraction, ALF & fusion modules
├── dinov3/                      # DINOv3 backbone code
├── datasets/                    # Dataset loaders
└── utils/                       # Config loading and evaluation metrics
```

## 📚 Citation

If you find ALF useful in your research, please consider citing:

```bibtex
@inproceedings{yang2026alf,
  title     = {ALF: Sharpness-Aware Adaptive Layer Fusion for Training-Free Anomaly Detection},
  author    = {Yang, Deyu and Ren, Ziliang and Zou, Yu and Gao, Hongchao and Liu, Ying},
  booktitle = {IEEE International Conference on Image Processing (ICIP)},
  year      = {2026}
}
```

## 🙏 Acknowledgements

This repository is built upon [MuSc](https://github.com/xiaozhen228/MuSc) (ICLR 2024). We thank the authors for releasing their code.
