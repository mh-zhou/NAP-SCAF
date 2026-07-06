from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nap_scaf import build_nap_scaf
from nap_scaf.data import BraTSPNGDataset
from nap_scaf.metrics import AverageMeter, brats_region_metrics
from nap_scaf.utils import load_checkpoint, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NAP-SCAF on BraTS PNG slices.")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", default="results/eval_metrics.json", type=str)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--mid-channels", default=16, type=int)
    parser.add_argument("--stages", default=4, type=int)
    parser.add_argument("--missing-modalities", default="", type=str, help="Comma-separated modalities to zero-fill, e.g. t1,t2.")
    parser.add_argument("--num-workers", default=4, type=int)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    missing = [m.strip() for m in args.missing_modalities.split(",") if m.strip()]
    dataset = BraTSPNGDataset(args.data_root, image_size=(args.image_size, args.image_size), missing_modalities=missing)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_nap_scaf(
        in_channels=4,
        num_classes=4,
        mid_channels=args.mid_channels,
        stages=args.stages,
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    meter = AverageMeter()
    for image, target in tqdm(loader, desc="eval"):
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(image)
        pred = torch.argmax(logits, dim=1)
        meter.update(brats_region_metrics(pred, target), n=image.size(0))
    metrics = meter.average()
    write_json(args.output, {"missing_modalities": missing, "metrics": metrics})
    print(metrics)


if __name__ == "__main__":
    main()
