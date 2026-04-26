# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dinov3.distributed as distributed
from dinov3.checkpointer import init_fsdp_model_from_checkpoint
from dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize
from dinov3.models import build_model_from_cfg
from dinov3.train.param_groups import get_params_groups_with_decay_fsdp

logger = logging.getLogger("dinov3")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_2d_sincos_embed(num_patches, dim):
    """
    Standard 2-D sine-cosine positional embedding.
    Returns tensor of shape [1, num_patches, dim] stored as a buffer.
    """
    h = w = int(math.sqrt(num_patches))
    assert h * w == num_patches, "num_patches must be a perfect square"
    assert dim % 4 == 0, "dim must be divisible by 4 for 2-D sincos embed"

    omega = torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)
    omega = 1.0 / (10000.0 ** omega)  # [dim//4]

    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")  # [h, w]

    grid_h = grid_h.flatten()  # [num_patches]
    grid_w = grid_w.flatten()  # [num_patches]

    emb_h = torch.outer(grid_h, omega)  # [num_patches, dim//4]
    emb_w = torch.outer(grid_w, omega)  # [num_patches, dim//4]

    emb = torch.cat(
        [torch.sin(emb_h), torch.cos(emb_h), torch.sin(emb_w), torch.cos(emb_w)],
        dim=-1,
    )  # [num_patches, dim]

    return emb.unsqueeze(0)  # [1, num_patches, dim]


def patchify(images, patch_size):
    """Reshape [B, C, H, W] → [B, num_patches, patch_size²·C]."""
    B, C, H, W = images.shape
    h = H // patch_size
    w = W // patch_size
    x = images.reshape(B, C, h, patch_size, w, patch_size)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, patch_size ** 2 * C)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class _MAEDecoderBlock(nn.Module):
    """Pre-norm transformer block using standard LayerNorm (FSDP-safe)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class MAEDecoder(nn.Module):
    """
    Lightweight MAE decoder: linear projection → sinusoidal pos-embed →
    N transformer blocks → LayerNorm → linear head → pixel predictions.
    """

    def __init__(
        self,
        encoder_dim=1024,
        decoder_dim=256,
        depth=4,
        num_heads=8,
        patch_size=16,
        in_chans=3,
        num_patches=196,
    ):
        super().__init__()
        self.embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.register_buffer("pos_embed", _build_2d_sincos_embed(num_patches, decoder_dim))
        self.blocks = nn.ModuleList(
            [_MAEDecoderBlock(decoder_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_size ** 2 * in_chans, bias=True)

    def forward(self, x):
        # x: [B, N, encoder_dim]
        x = self.embed(x) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))  # [B, N, patch_size^2 * in_chans]

    def init_weights(self):
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.zeros_(self.embed.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)


# ---------------------------------------------------------------------------
# MAEMetaArch
# ---------------------------------------------------------------------------


class MAEMetaArch(nn.Module):
    """
    MAE (Masked Autoencoder) meta-architecture for SSL pretraining.

    The encoder is a DinoVisionTransformer whose existing mask_token substitution
    (prepare_tokens_with_masks) is reused — no ViT changes needed.
    The decoder is a lightweight 4-block transformer that predicts normalised
    pixel patches at masked positions via MSE loss.

    Interface is identical to SSLMetaArch so train.py needs no extra branches.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        backbone, embed_dim = build_model_from_cfg(cfg, only_teacher=True)
        logger.info(f"MAEMetaArch: backbone embed_dim={embed_dim}")

        img_size = cfg.crops.global_crops_size
        patch_size = cfg.student.patch_size
        num_patches = (img_size // patch_size) ** 2

        decoder = MAEDecoder(
            encoder_dim=embed_dim,
            decoder_dim=cfg.mae.decoder_dim,
            depth=cfg.mae.decoder_depth,
            num_heads=cfg.mae.decoder_num_heads,
            patch_size=patch_size,
            in_chans=3,
            num_patches=num_patches,
        )

        # ac_compile_parallelize requires student to be nn.ModuleDict with "backbone" key
        self.student = nn.ModuleDict({"backbone": backbone, "decoder": decoder})
        self._patch_size = patch_size
        self._norm_target = cfg.mae.norm_target

    def init_weights(self):
        self.student["backbone"].init_weights()
        self.student["decoder"].init_weights()
        if self.cfg.student.resume_from_teacher_chkpt:
            init_fsdp_model_from_checkpoint(
                self.student,
                self.cfg.student.resume_from_teacher_chkpt,
                skip_load_keys=["decoder"],
                keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
                process_group=distributed.get_process_subgroup(),
            )

    def forward_backward(self, data, *, teacher_temp, iteration=0, **kw):
        images = data["collated_global_crops"].cuda(non_blocking=True)  # [B, C, H, W]
        masks = data["collated_masks"].cuda(non_blocking=True)          # [B, N] bool

        # Encode — ViT substitutes masked patch embeddings with self.mask_token
        patch_tokens = self.student["backbone"].forward_features(
            images, masks=masks
        )["x_norm_patchtokens"]  # [B, N, D]

        # Decode → pixel predictions
        pred = self.student["decoder"](patch_tokens)  # [B, N, p²C]

        # Pixel targets
        target = patchify(images, self._patch_size)   # [B, N, p²C]
        if self._norm_target:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = F.mse_loss(pred[masks], target[masks])
        loss.backward()
        return loss, {"loss_mae": loss.item()}

    def update_ema(self, m):
        pass  # no EMA teacher in MAE

    def update_gram(self, m=0):
        pass  # no gram teacher in MAE

    def build_data_augmentation_dino(self, cfg):
        from dinov3.data.augmentations_mae import DataAugmentationMAE

        return DataAugmentationMAE(
            global_crops_size=cfg.crops.global_crops_size,
            global_crops_scale=cfg.crops.global_crops_scale,
            horizontal_flips=cfg.crops.horizontal_flips,
            mean=cfg.crops.rgb_mean,
            std=cfg.crops.rgb_std,
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
