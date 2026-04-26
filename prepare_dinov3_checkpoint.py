"""
Convert a DINOv3 hub checkpoint (bare backbone state dict) to the format
expected by student.resume_from_teacher_chkpt:
  {"teacher": {"backbone.<key>": tensor, ...}}

The hub .pth files contain only the backbone weights with no prefix.
The training code expects a "teacher" wrapper and a "backbone." prefix.

Usage:
  python prepare_dinov3_checkpoint.py \\
    --input  /data/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \\
    --output /data/checkpoints/dinov3_vitl16_teacher.pth

Download the hub checkpoint first:
  wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \\
       -O /data/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to downloaded hub .pth file")
    parser.add_argument("--output", required=True, help="Path to write the converted checkpoint")
    args = parser.parse_args()

    raw = torch.load(args.input, map_location="cpu")

    if isinstance(raw, dict) and "teacher" in raw:
        # Already in teacher-checkpoint format (e.g. from a previous training run)
        print("Checkpoint already has 'teacher' key — saving as-is.")
        teacher_sd = raw["teacher"]
    else:
        # Hub format: bare backbone state dict, no "backbone." prefix
        teacher_sd = {f"backbone.{k}": v for k, v in raw.items()}
        print(f"Converted {len(teacher_sd)} backbone keys (added 'backbone.' prefix).")

    torch.save({"teacher": teacher_sd}, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
