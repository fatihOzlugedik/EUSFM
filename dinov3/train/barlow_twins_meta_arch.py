# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging

import torch
import torch.distributed as dist
import torch.nn as nn

import dinov3.distributed as distributed
from dinov3.checkpointer import init_fsdp_model_from_checkpoint
from dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize
from dinov3.models import build_model_from_cfg
from dinov3.train.param_groups import get_params_groups_with_decay_fsdp

logger = logging.getLogger("dinov3")


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------


class BarlowTwinsProjector(nn.Module):
    """
    3-layer MLP projector using LayerNorm instead of BatchNorm.
    BatchNorm is incompatible with FSDP (NotImplementedError in ac_compile_parallelize).
    LayerNorm is a good approximation when the global batch is large (≥128).
    """

    def __init__(self, in_dim, proj_dim):
        super().__init__()
        layers = []
        for i in range(3):
            in_d = in_dim if i == 0 else proj_dim
            layers += [nn.Linear(in_d, proj_dim, bias=False), nn.LayerNorm(proj_dim)]
            if i < 2:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _barlow_loss(z1, z2, lambda_coeff):
    """
    Distributed Barlow Twins cross-correlation loss.

    z1, z2: [B, D] local embeddings on each rank.
    Cross-correlation matrix c = (1/N) · Z1^T Z2, where N is the global
    batch size. All-reduce of c (D×D ≈ 16 MB for proj_dim=2048) is cheap.
    Gradient flows through z1 and z2 via z1.T @ z2.
    """
    B, D = z1.shape

    # Per-rank normalisation (approximates global batch stats)
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)

    # Local contribution to the cross-correlation matrix
    c = z1.T @ z2  # [D, D]

    # Sum across all ranks → global cross-correlation
    dist.all_reduce(c)
    N_global = B * dist.get_world_size()
    c = c / N_global

    # Invariance: diagonal → 1; Redundancy reduction: off-diagonal → 0
    eye = torch.eye(D, device=c.device, dtype=c.dtype)
    on_diag = (torch.diagonal(c) - 1).pow(2).sum()
    off_diag = (c * (1 - eye)).pow(2).sum()
    return on_diag + lambda_coeff * off_diag


# ---------------------------------------------------------------------------
# BarlowTwinsMetaArch
# ---------------------------------------------------------------------------


class BarlowTwinsMetaArch(nn.Module):
    """
    Barlow Twins meta-architecture for SSL pretraining.

    Two global views are extracted from the same image and passed through the
    shared backbone + projector. The cross-correlation loss encourages the
    embeddings to be invariant to augmentation (on-diagonal) while reducing
    redundancy between dimensions (off-diagonal).

    Interface is identical to SSLMetaArch so train.py needs no extra branches.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        backbone, embed_dim = build_model_from_cfg(cfg, only_teacher=True)
        logger.info(f"BarlowTwinsMetaArch: backbone embed_dim={embed_dim}")

        projector = BarlowTwinsProjector(embed_dim, cfg.barlow_twins.proj_dim)

        # ac_compile_parallelize requires student to be nn.ModuleDict with "backbone" key
        self.student = nn.ModuleDict({"backbone": backbone, "projector": projector})
        self._lambda = cfg.barlow_twins.lambda_off_diag

    def init_weights(self):
        self.student["backbone"].init_weights()
        self.student["projector"].init_weights()
        if self.cfg.student.resume_from_teacher_chkpt:
            init_fsdp_model_from_checkpoint(
                self.student,
                self.cfg.student.resume_from_teacher_chkpt,
                skip_load_keys=["projector"],
                keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
                process_group=distributed.get_process_subgroup(),
            )

    def forward_backward(self, data, *, teacher_temp, iteration=0, **kw):
        # DataAugmentationDINO with local_crops_number=0 produces 2 global crops
        crops = data["collated_global_crops"].cuda(non_blocking=True)  # [2B, C, H, W]
        B = crops.shape[0] // 2
        v1, v2 = crops[:B], crops[B:]

        z1 = self.student["projector"](
            self.student["backbone"].forward_features(v1)["x_norm_clstoken"]
        )  # [B, proj_dim]
        z2 = self.student["projector"](
            self.student["backbone"].forward_features(v2)["x_norm_clstoken"]
        )  # [B, proj_dim]

        loss = _barlow_loss(z1, z2, self._lambda)
        loss.backward()
        return loss, {"loss_bt": loss.item()}

    def update_ema(self, m):
        pass  # no EMA teacher in BarlowTwins

    def update_gram(self, m=0):
        pass  # no gram teacher in BarlowTwins

    def build_data_augmentation_dino(self, cfg):
        from dinov3.data.augmentations import DataAugmentationDINO

        return DataAugmentationDINO(
            global_crops_scale=cfg.crops.global_crops_scale,
            local_crops_scale=cfg.crops.local_crops_scale,
            local_crops_number=0,  # 2 global crops only — no local crops
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            gram_teacher_crops_size=None,
            gram_teacher_no_distortions=False,
            local_crops_subset_of_global_crops=False,
            share_color_jitter=cfg.crops.share_color_jitter,
            horizontal_flips=cfg.crops.horizontal_flips,
            mean=cfg.crops.rgb_mean,
            std=cfg.crops.rgb_std,
            min_content_mean=cfg.crops.min_content_mean,
            max_crop_retries=cfg.crops.max_crop_retries,
            color_jitter_brightness=cfg.crops.color_jitter_brightness,
            color_jitter_contrast=cfg.crops.color_jitter_contrast,
            color_jitter_saturation=cfg.crops.color_jitter_saturation,
            color_jitter_hue=cfg.crops.color_jitter_hue,
            color_jitter_prob=cfg.crops.color_jitter_prob,
            random_grayscale_prob=cfg.crops.random_grayscale_prob,
            solarize_prob=cfg.crops.solarize_prob,
        )

    def get_params_groups(self):
        all_groups = []
        for name, m in self.student.items():
            all_groups += get_params_groups_with_decay_fsdp(
                model=m,
                lr_decay_rate=self.cfg.optim.layerwise_decay if name == "backbone" else 1.0,
                patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult if name == "backbone" else 1.0,
                dino_head_wd_multiplier=1.0,
            )
        return all_groups

    def prepare_for_distributed_training(self):
        ac_compile_parallelize(
            trained_model=self.student,
            inference_only_models=[],
            cfg=self.cfg,
            trained_model_process_group=distributed.get_process_subgroup(),
            inference_only_models_process_groups=[],
        )
