from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from nap_scaf import build_nap_scaf
from nap_scaf.data import BraTSPNGDataset
from nap_scaf.losses import total_nap_scaf_loss
from nap_scaf.metrics import AverageMeter, brats_region_metrics
from nap_scaf.utils import save_checkpoint, set_seed, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Train NAP-SCAF on BraTS PNG slices.")
    parser.add_argument("--data-root", required=True, type=str, help="Dataset root containing flair/t1/t1ce/t2/mask folders.")
    parser.add_argument("--output-dir", default="checkpoints/nap_scaf", type=str)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--val-batch-size", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--mid-channels", default=16, type=int)
    parser.add_argument("--stages", default=4, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision.")
    return parser.parse_args()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    meter = AverageMeter()
    for image, target in tqdm(loader, desc="val", leave=False):
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(image)
        pred = torch.argmax(logits, dim=1)
        meter.update(brats_region_metrics(pred, target), n=image.size(0))
    return meter.average()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = BraTSPNGDataset(args.data_root, image_size=(args.image_size, args.image_size))
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_nap_scaf(
        in_channels=4,
        num_classes=4,
        mid_channels=args.mid_channels,
        stages=args.stages,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    best_dice = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        num_samples = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for image, target in pbar:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                output = model(image)
                loss = total_nap_scaf_loss(
                    output,
                    target,
                    num_classes=4,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.detach()) * image.size(0)
            num_samples += image.size(0)
            pbar.set_postfix(loss=train_loss_sum / max(1, num_samples))

        val_metrics = validate(model, val_loader, device)
        train_loss = train_loss_sum / max(1, num_samples)
        record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(record)
        write_json(output_dir / "history.json", {"history": history})
        print(record)

        avg_dice = val_metrics.get("Avg_dice", 0.0)
        save_checkpoint(output_dir / "last.pth", model, optimizer, epoch, avg_dice)
        if avg_dice > best_dice:
            best_dice = avg_dice
            save_checkpoint(output_dir / "best.pth", model, optimizer, epoch, best_dice)
            print(f"Saved best checkpoint with Avg Dice = {best_dice:.4f}")


if __name__ == "__main__":
    main()
