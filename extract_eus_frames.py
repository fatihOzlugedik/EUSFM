"""
Extract frames from EUS MP4 files and build the dataset directory structure.

Input layout:
    src/
    ├── train/
    │   ├── procedure_001.mp4
    │   └── procedure_002.mp4
    └── val/
        └── procedure_003.mp4

Output layout:
    dst/
    ├── train/
    │   ├── procedure_001/
    │   │   ├── frame_000000.jpg
    │   │   └── frame_000010.jpg  (every --subsample-th frame)
    │   └── procedure_002/
    ├── val/
    │   └── procedure_003/
    └── file_lists/
        ├── train.txt
        └── val.txt

Usage:
    python extract_eus_frames.py --src /data/raw_eus --dst /data/eus --subsample 10
"""

import argparse
import os
from pathlib import Path

import cv2


def extract_video(mp4_path: Path, out_dir: Path, subsample: int, quality: int) -> list[str]:
    """Extract every `subsample`-th frame from mp4_path into out_dir as JPEG.
    Returns list of relative paths (relative to dataset root) written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {mp4_path}")

    written = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % subsample == 0:
            fname = out_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            written.append(str(fname))
        frame_idx += 1

    cap.release()
    return written


def process_split(src_split: Path, dst_root: Path, split_name: str, subsample: int, quality: int) -> list[str]:
    """Process all MP4s in src_split. Returns relative paths for the file list."""
    mp4_files = sorted(src_split.glob("*.mp4"))
    if not mp4_files:
        print(f"  [warn] No .mp4 files found in {src_split}")
        return []

    all_relative = []
    for mp4 in mp4_files:
        proc_name = mp4.stem
        out_dir = dst_root / split_name / proc_name
        print(f"  {mp4.name} -> {out_dir.relative_to(dst_root)} ...", end=" ", flush=True)
        written = extract_video(mp4, out_dir, subsample, quality)
        # Store paths relative to dst_root
        rel_paths = [str(Path(p).relative_to(dst_root)) for p in written]
        all_relative.extend(rel_paths)
        print(f"{len(rel_paths)} frames")

    return all_relative


def main():
    parser = argparse.ArgumentParser(description="Extract EUS frames from MP4 files")
    parser.add_argument("--src", required=True, help="Source directory with train/ and val/ subdirs of MP4 files")
    parser.add_argument("--dst", required=True, help="Output dataset root directory")
    parser.add_argument("--subsample", type=int, default=10, help="Keep every Nth frame (default: 10)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100 (default: 95)")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    list_dir = dst / "file_lists"
    list_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        src_split = src / split
        if not src_split.exists():
            print(f"Skipping {split}: {src_split} does not exist")
            continue

        print(f"\n[{split}]")
        rel_paths = process_split(src_split, dst, split, args.subsample, args.quality)

        list_file = list_dir / f"{split}.txt"
        with open(list_file, "w") as f:
            f.write("\n".join(rel_paths) + "\n")
        print(f"  -> {len(rel_paths)} total frames written to {list_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()
