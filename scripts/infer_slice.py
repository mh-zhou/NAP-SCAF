from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from nap_scaf import build_nap_scaf
from nap_scaf.utils import load_checkpoint

MODALITIES = ("flair", "t1", "t1ce", "t2")


def read_modality(path: str, image_size: int) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[:2] != (image_size, image_size):
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return image.astype(np.float32) / 255.0


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on one four-modality slice.")
    for m in MODALITIES:
        parser.add_argument(f"--{m}", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", default="results/prediction.png", type=str)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--mid-channels", default=16, type=int)
    parser.add_argument("--stages", default=4, type=int)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = [read_modality(getattr(args, m), args.image_size) for m in MODALITIES]
    image = torch.from_numpy(np.stack(channels, axis=0)).unsqueeze(0).float().to(device)

    model = build_nap_scaf(
        in_channels=4,
        num_classes=4,
        mid_channels=args.mid_channels,
        stages=args.stages,
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    pred = torch.argmax(model(image), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred).save(args.output)
    print(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()
