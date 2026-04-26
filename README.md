# EUS Pretraining with DINOv3

Self-supervised pretraining of Vision Transformers on **Endoscopic Ultrasound (EUS)** data,
built on the DINOv3 framework. Supports three SSL objectives (DINOv3, MAE, Barlow Twins) with
domain-specific augmentations designed for ultrasound image characteristics.

---

## Overview

Ultrasound images have properties that make standard natural-image SSL recipes suboptimal:

| Property | Implication |
|---|---|
| Fan-shaped acoustic cone with black dead zones | Standard random crops land in empty regions |
| Fixed anatomical orientation (no arbitrary rotation) | Horizontal flips produce anatomically impossible views |
| Grayscale (no hue/saturation) | Saturation/hue jitter and grayscale conversion are no-ops or harmful |
| Exponential signal attenuation with depth | Real depth-dependent intensity variation should be augmented |
| Acoustic shadow artefacts from ribs and calcifications | Shadow dropout augmentation improves robustness |

This repository adds:
1. **Domain-specific augmentations** for EUS, each independently ablatable
2. **MAE pretraining** (pixel-space masked autoencoder)
3. **Barlow Twins pretraining** (redundancy-reduction contrastive)
4. **DINOv3 DAPT** (domain-adaptive continued pretraining from the official ViT-L checkpoint)

All three methods share the same ViT-L backbone, FSDP infrastructure, and data pipeline,
enabling a clean SSL objective comparison.

---

## Repository Structure

```
dinov3/
├── data/
│   ├── augmentations.py          # DataAugmentationDINO (EUS params added)
│   ├── augmentations_mae.py      # DataAugmentationMAE (single crop, no jitter)
│   ├── transforms.py             # ContentAwareRandomResizedCrop, DepthAttenuation,
│   │                             #   GaussianShadow
│   └── datasets/
│       └── eus.py                # EUSDataset (frame-level, file-list based)
├── train/
│   ├── ssl_meta_arch.py          # DINOv3 objective (unchanged)
│   ├── mae_meta_arch.py          # MAE objective
│   └── barlow_twins_meta_arch.py # Barlow Twins objective
└── configs/
    ├── ssl_default_config.yaml   # All config keys declared here (OmegaConf struct)
    └── train/eus/
        ├── dinov3_default.yaml             # Control: vanilla DINOv3 augmentations
        ├── eus_vitl_base.yaml              # EUS baseline (no physics augs)
        ├── eus_vitl_content_aware_crop.yaml
        ├── eus_vitl_depth_attenuation.yaml
        ├── eus_vitl_gaussian_shadow.yaml
        ├── eus_vitl_physics_augs.yaml      # depth_attenuation + gaussian_shadow
        ├── eus_vitl_full.yaml              # all EUS augs combined
        ├── eus_vitl_mae_scratch.yaml
        ├── eus_vitl_mae_pretrained.yaml
        ├── eus_vitl_barlow_twins_scratch.yaml
        └── eus_vitl_barlow_twins_pretrained.yaml

prepare_dinov3_checkpoint.py      # Convert hub .pth to resume_from_teacher_chkpt format
```

---

## Quick Start

### 1. Dataset Preparation

Organise EUS frames in the following layout:

```
/data/eus/
├── train/
│   ├── <procedure_id>/
│   │   ├── frame_000000.jpg
│   │   ├── frame_000001.jpg
│   │   └── ...
│   └── ...
├── val/
│   └── <procedure_id>/
│       └── ...
└── file_lists/
    ├── train.txt   # one relative path per line, e.g. train/proc001/frame_000000.jpg
    └── val.txt
```

The config string `EUS:split=TRAIN:root=/data/eus` is parsed by the data loader automatically.

**`OFFICIAL_EPOCH_LENGTH`** in all EUS configs is set to `3907` =
`ceil(1,000,000 frames / 256 global batch size)` for 4×A100 with per-GPU batch 64.
Adjust this if your dataset size or batch configuration differs.

### 2. Checkpoint Preparation (for DAPT runs)

Download the official DINOv3 ViT-L/16 checkpoint and convert it to the format expected by
`resume_from_teacher_chkpt`:

```bash
# Download
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
     -O /data/checkpoints/dinov3_vitl16_pretrain.pth

# Convert
python prepare_dinov3_checkpoint.py \
    --input  /data/checkpoints/dinov3_vitl16_pretrain.pth \
    --output /data/checkpoints/dinov3_vitl16_teacher.pth
```

Then set `student.resume_from_teacher_chkpt` in your config (or pass it as an override):

```bash
# inline override example
python -m torch.distributed.run ... train.py \
    --config-file dinov3/configs/train/eus/eus_vitl_full.yaml \
    student.resume_from_teacher_chkpt=/data/checkpoints/dinov3_vitl16_teacher.pth
```

The conversion script handles both hub format (bare backbone state dict) and
training-run format (already has a `"teacher"` key).

### 3. Launch Training

```bash
torchrun --nproc_per_node=4 \
    -m dinov3.train.train \
    --config-file dinov3/configs/train/eus/eus_vitl_full.yaml \
    --output-dir /output/eus_full
```

---

## SSL Objectives

### DINOv3 (default)

**Objective:** DINO (CLS-token teacher-student) + iBOT (masked patch feature prediction) +
KoLeo (diversity regularisation).

**Config:** any `eus_vitl_*.yaml` without a `MODEL.META_ARCHITECTURE` key
(defaults to `SSLMetaArch`).

**Notes:**
- Student and teacher both initialise from the pretrained backbone
- EMA teacher update: `momentum` schedule starting at ~0.992
- Requires `local_crops_number > 0`

---

### MAE — Masked Autoencoder

**Objective:** Pixel-space reconstruction of randomly masked patches via MSE loss on
normalised patch targets.

**Config:** `eus_vitl_mae_scratch.yaml` / `eus_vitl_mae_pretrained.yaml`

**Architecture:**
- **Encoder:** DINOv3 ViT-L. Masked patches are replaced with `self.mask_token` using the
  existing `prepare_tokens_with_masks` mechanism — zero ViT code changes.
- **Decoder:** Lightweight 4-block transformer (256-dim, 8 heads) with 2-D sinusoidal
  positional embeddings. Predicts patch pixels at masked positions.
- **Loss:** `MSE(pred[masks], normalise(target[masks]))` — patch targets are mean/variance
  normalised per patch (controlled by `mae.norm_target`).

**Masking:** Uses the existing `MaskingGenerator` with `ibot.mask_ratio_min_max: [0.5, 0.75]`
and `ibot.mask_sample_probability: 1.0`. `max_num_patches` in `train.py` is derived from
`cfg.ibot.mask_ratio_min_max[1]` so 75% masking (147/196 patches) is supported.

**Augmentation:** Single crop, no colour jitter (`DataAugmentationMAE`). Colour distortions
are inappropriate when the reconstruction target is the raw pixel.

**Key config parameters:**

| Parameter | Default | Description |
|---|---|---|
| `mae.decoder_dim` | 256 | Decoder embedding dimension |
| `mae.decoder_depth` | 4 | Number of decoder transformer blocks |
| `mae.decoder_num_heads` | 8 | Decoder attention heads |
| `mae.norm_target` | true | Normalise patch pixels before MSE |
| `ibot.mask_ratio_min_max` | [0.5, 0.75] | Masking ratio range |

---

### Barlow Twins

**Objective:** Cross-correlation redundancy-reduction between two augmented views of the
same image. Encourages invariant and non-redundant representations.

**Config:** `eus_vitl_barlow_twins_scratch.yaml` / `eus_vitl_barlow_twins_pretrained.yaml`

**Architecture:**
- **Backbone:** DINOv3 ViT-L (CLS token extracted)
- **Projector:** 3-layer MLP (1024 → 2048 → 2048 → 2048) with `LayerNorm` after each linear.
  LayerNorm is used instead of BatchNorm because BatchNorm is incompatible with FSDP.
- **Loss:** `L = Σ_i (C_ii - 1)² + λ Σ_{i≠j} C_ij²` where `C = (1/N) Z₁ᵀ Z₂` is the
  cross-correlation matrix, all-reduced across ranks. The D×D all-reduce (~16 MB for
  `proj_dim=2048`) is cheap relative to forward passes.

**Data:** `DataAugmentationDINO` with `local_crops_number=0` produces exactly 2 global views.
Both views are stacked in `collated_global_crops` as `[2B, C, H, W]`.

**Key config parameters:**

| Parameter | Default | Description |
|---|---|---|
| `barlow_twins.proj_dim` | 2048 | Projector output dimension |
| `barlow_twins.lambda_off_diag` | 0.005 | Off-diagonal redundancy penalty weight |

---

## Domain-Specific Augmentations

All augmentations are independently togglable via config. Default values are `0.0` / disabled,
so the vanilla DINOv3 behaviour is preserved when not set.

### Content-Aware Crop

**Parameter:** `crops.min_content_mean` (float, default `0.0`)

Retries random crops whose mean pixel intensity is below the threshold (in [0, 1] scale).
Avoids landing entirely in the black acoustic dead-zone at the edges of the fan.
Up to `crops.max_crop_retries` (default 10) attempts before accepting the last crop.

**When to use:** Any EUS run. Setting `min_content_mean: 0.05` is a reliable default.

---

### Depth Attenuation

**Parameters:** `crops.depth_attenuation_p`, `crops.depth_attenuation_alpha_min/max`

Simulates exponential acoustic signal attenuation with tissue depth:

```
I_out(r, c) = I_in(r, c) × exp(−α × r / H)
```

`α` is sampled uniformly from `[alpha_min, alpha_max]` each application.
Applied with probability `depth_attenuation_p`.

**Physical motivation:** Ultrasound signal is attenuated as it travels through tissue.
The attenuation coefficient varies with tissue type and patient anatomy (e.g. adipose tissue
in obese patients). Training with this augmentation makes features robust to this variation.

---

### Gaussian Shadow

**Parameters:** `crops.gaussian_shadow_p`, `crops.gaussian_shadow_intensity_min/max`,
`crops.gaussian_shadow_sigma_ratio`

Simulates acoustic shadow dropout from hyperechoic structures (ribs, calcified lesions):

```
I_out(r, c) = I_in(r, c) × (1 − s × G(r, c))
```

`G` is a 2-D Gaussian centred at a random location, `σ = sigma_ratio × min(H, W)`.
`s` (shadow strength) is sampled from `[intensity_min, intensity_max]`.

**Physical motivation:** Dense structures reflect most of the acoustic energy, casting a
shadow of signal dropout on structures behind them. This is one of the most common and
diagnostically significant artefacts in EUS.

---

## Ablation Configurations

| Config | SSL | Content-Aware Crop | Depth Atten. | Gaussian Shadow |
|---|---|---|---|---|
| `dinov3_default` | DINOv3 | — | — | — |
| `eus_vitl_base` | DINOv3 | — | — | — |
| `eus_vitl_content_aware_crop` | DINOv3 | ✓ | — | — |
| `eus_vitl_depth_attenuation` | DINOv3 | — | ✓ | — |
| `eus_vitl_gaussian_shadow` | DINOv3 | — | — | ✓ |
| `eus_vitl_physics_augs` | DINOv3 | — | ✓ | ✓ |
| `eus_vitl_full` | DINOv3 | ✓ | ✓ | ✓ |
| `eus_vitl_mae_scratch` | MAE | — | — | — |
| `eus_vitl_mae_pretrained` | MAE | — | — | — |
| `eus_vitl_barlow_twins_scratch` | Barlow Twins | — | — | — |
| `eus_vitl_barlow_twins_pretrained` | Barlow Twins | — | — | — |

The DINOv3 ablations share the same EUS-adapted base augmentations (no flips, reduced jitter,
no solarize/grayscale) and differ only in which physics augmentations are enabled.

---

## Pretrained Initialisation

The pretrained configs (`*_pretrained.yaml`) initialise the backbone from the official
DINOv3 ViT-L/16 checkpoint. This is domain-adaptive pretraining (DAPT): the model starts
with strong natural-image features and adapts them to the EUS distribution.

The student architecture must exactly match the hub checkpoint. These four overrides
must be set in every EUS config:

```yaml
student:
  norm_layer: layernormbf16          # hub uses LayerNorm in bfloat16 mode
  n_storage_tokens: 4                # 4 register tokens
  mask_k_bias: true                  # bias masking in QKV
  pos_embed_rope_rescale_coords: 2   # RoPE coordinate rescaling factor
  resume_from_teacher_chkpt: ""      # path from prepare_dinov3_checkpoint.py
```

For MAE and Barlow Twins DAPT, only the backbone is loaded from the pretrained checkpoint;
the decoder / projector is randomly initialised. `strict=False` in `load_state_dict` allows
this — backbone keys load, head keys that are absent stay at random init.

---

## Hardware Requirements

All configs are tuned for **4× NVIDIA A100 (80 GB)** with:
- Per-GPU batch size: 64 (global batch: 256)
- Mixed precision: bfloat16
- Sharding: `SHARD_GRAD_OP` (gradient + optimizer state sharding)
- Activation checkpointing: enabled via `ac_compile_parallelize`

To scale to different GPU counts, adjust `train.OFFICIAL_EPOCH_LENGTH`:
```
OFFICIAL_EPOCH_LENGTH = ceil(dataset_size / (num_gpus × per_gpu_batch))
```

---

## Interface Contract

All three MetaArch classes expose an identical interface consumed by `train.py`:

| Method | DINOv3 | MAE | Barlow Twins |
|---|---|---|---|
| `__init__(cfg)` | ✓ | ✓ | ✓ |
| `init_weights()` | ✓ | ✓ backbone+decoder | ✓ backbone+projector |
| `prepare_for_distributed_training()` | ✓ FSDP | ✓ FSDP | ✓ FSDP |
| `get_params_groups()` | ✓ | ✓ | ✓ |
| `forward_backward(data, *, teacher_temp, iteration)` | DINO+iBOT+KoLeo | MSE pixel recon | Cross-corr loss |
| `update_ema(m)` | EMA teacher update | no-op | no-op |
| `update_gram(m)` | gram update | no-op | no-op |
| `build_data_augmentation_dino(cfg)` | DataAugmentationDINO | DataAugmentationMAE | DataAugmentationDINO (local=0) |
| `student` | `nn.ModuleDict{"backbone", "dino_head", "ibot_head"}` | `nn.ModuleDict{"backbone", "decoder"}` | `nn.ModuleDict{"backbone", "projector"}` |

`MODEL.META_ARCHITECTURE` in the config selects the class:
`SSLMetaArch` | `MAEMetaArch` | `BarlowTwinsMetaArch`.

---

## Configuration Reference

### New top-level sections in `ssl_default_config.yaml`

```yaml
mae:
  decoder_dim: 256        # decoder embedding dimension
  decoder_depth: 4        # number of lightweight transformer blocks in decoder
  decoder_num_heads: 8    # decoder attention heads
  norm_target: true       # normalise pixel patches before MSE (MAE paper §3.1)

barlow_twins:
  proj_dim: 2048          # projection MLP output dimension
  lambda_off_diag: 0.005  # weight for off-diagonal redundancy terms
```

### EUS crop parameters (added to `crops:` section)

```yaml
crops:
  min_content_mean: 0.0           # content-aware crop threshold [0,1]; 0=disabled
  max_crop_retries: 10            # max retry attempts for content-aware crop
  depth_attenuation_p: 0.0        # probability of depth attenuation; 0=disabled
  depth_attenuation_alpha_min: 0.01
  depth_attenuation_alpha_max: 0.05
  gaussian_shadow_p: 0.0          # probability of Gaussian shadow; 0=disabled
  gaussian_shadow_intensity_min: 0.3
  gaussian_shadow_intensity_max: 0.8
  gaussian_shadow_sigma_ratio: 0.15
  color_jitter_brightness: 0.4    # configurable jitter (was hardcoded)
  color_jitter_contrast: 0.4
  color_jitter_saturation: 0.2
  color_jitter_hue: 0.1
  color_jitter_prob: 0.8
  random_grayscale_prob: 0.2
  solarize_prob: 0.2
```
