# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from enum import Enum
from typing import Any, Callable, Optional

from .extended import ExtendedVisionDataset


class EUSDataset(ExtendedVisionDataset):
    """
    Endoscopic Ultrasound dataset for self-supervised pretraining.

    Expected layout:
        root/
        ├── train/<procedure_id>/frame_XXXXXX.jpg
        ├── val/<procedure_id>/frame_XXXXXX.jpg
        └── file_lists/
            ├── train.txt   (one relative path per line)
            └── val.txt

    Config string: EUS:split=TRAIN:root=/path/to/eus
    """

    class Split(Enum):
        TRAIN = "train"
        VAL = "val"

    def __init__(
        self,
        *,
        split: "EUSDataset.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._split = split
        list_path = os.path.join(root, "file_lists", f"{split.value}.txt")
        with open(list_path) as f:
            self._entries = [line.strip() for line in f if line.strip()]

    def get_image_data(self, index: int) -> bytes:
        path = os.path.join(self.root, self._entries[index])
        with open(path, "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        return None  # SSL — no labels

    def __len__(self) -> int:
        return len(self._entries)
