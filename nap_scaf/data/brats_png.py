"""BraTS PNG slice dataset.

Expected directory layout:

root/
  flair/*.png
  t1/*.png
  t1ce/*.png
  t2/*.png
  mask/*.png
  slice_list.txt        # optional, one slice id per line without extension

Mask labels may be stored either as BraTS labels {0, 1, 2, 4} or as contiguous
labels {0, 1, 2, 3}. They are returned as contiguous labels:
0 background, 1 NCR/NET, 2 ED, 3 ET.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSPNGDataset(Dataset):
    """Load four-modality BraTS 2D slices stored as PNG files."""

    def __init__(
        self,
        root: str | os.PathLike,
        image_size: Tuple[int, int] = (256, 256),
        modalities: Sequence[str] = ("flair", "t1", "t1ce", "t2"),
        mask_dir: str = "mask",
        missing_modalities: Optional[Iterable[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = tuple(image_size)
        self.modalities = tuple(modalities)
        self.mask_dir = mask_dir
        self.missing_modalities = set(missing_modalities or [])

        slice_list = self.root / "slice_list.txt"
        if slice_list.exists():
            self.names = [line.strip() for line in slice_list.read_text().splitlines() if line.strip()]
        else:
            mask_path = self.root / mask_dir
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask directory not found: {mask_path}")
            self.names = sorted(p.stem for p in mask_path.glob("*.png"))
        if not self.names:
            raise RuntimeError(f"No slices found under {self.root}")

    def __len__(self) -> int:
        return len(self.names)

    def _read_grayscale(self, path: Path, interpolation: int) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size[::-1], interpolation=interpolation)
        return image

    @staticmethod
    def _map_brats_labels(mask: np.ndarray) -> np.ndarray:
        if 4 in np.unique(mask):
            mapped = np.zeros_like(mask, dtype=np.uint8)
            mapped[mask == 1] = 1
            mapped[mask == 2] = 2
            mapped[mask == 4] = 3
            return mapped
        return mask.astype(np.uint8)

    def __getitem__(self, index: int):
        name = self.names[index]
        channels: List[np.ndarray] = []
        for modality in self.modalities:
            if modality in self.missing_modalities:
                image = np.zeros(self.image_size, dtype=np.float32)
            else:
                path = self.root / modality / f"{name}.png"
                image = self._read_grayscale(path, cv2.INTER_LINEAR).astype(np.float32) / 255.0
            channels.append(image)
        image_tensor = torch.from_numpy(np.stack(channels, axis=0)).float()

        mask = self._read_grayscale(self.root / self.mask_dir / f"{name}.png", cv2.INTER_NEAREST)
        mask = self._map_brats_labels(mask)
        mask_tensor = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor
