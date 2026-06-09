from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

INPUT_MODALITIES = ("flair", "t1", "t1ce", "t2", "seg")
OUTPUT_DIRS = {"flair": "flair", "t1": "t1", "t1ce": "t1ce", "t2": "t2", "seg": "mask"}


def normalize_to_uint8(array: np.ndarray, is_mask: bool = False) -> np.ndarray:
    if is_mask:
        return array.astype(np.uint8)
    p1, p99 = np.percentile(array, (1, 99))
    array = np.clip(array, p1, p99)
    if p99 <= p1:
        return np.zeros_like(array, dtype=np.uint8)
    return ((array - p1) / (p99 - p1) * 255.0).astype(np.uint8)


def load_case(case_dir: Path) -> Dict[str, np.ndarray]:
    case_name = case_dir.name
    data = {}
    for modality in INPUT_MODALITIES:
        path = case_dir / f"{case_name}_{modality}.nii.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        data[modality] = nib.load(str(path)).get_fdata()
    return data


def valid_slices(data: Dict[str, np.ndarray], min_nonzero_ratio: float) -> List[int]:
    z_dim = data["flair"].shape[2]
    indices = []
    for z in range(z_dim):
        if not np.any(data["seg"][:, :, z] > 0):
            continue
        ok = True
        for modality in ("flair", "t1", "t1ce", "t2"):
            image = data[modality][:, :, z]
            if np.count_nonzero(image) / image.size < min_nonzero_ratio:
                ok = False
                break
        if ok:
            indices.append(z)
    return indices


def parse_args():
    parser = argparse.ArgumentParser(description="Convert BraTS NIfTI volumes to aligned 2D PNG slices.")
    parser.add_argument("--input-dir", required=True, type=str, help="BraTS training root containing case folders.")
    parser.add_argument("--output-dir", required=True, type=str, help="Output PNG dataset root.")
    parser.add_argument("--slices-per-case", default=5, type=int, help="Number of tumor-containing slices sampled per case.")
    parser.add_argument("--min-nonzero-ratio", default=0.05, type=float)
    parser.add_argument("--seed", default=7, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    for dirname in OUTPUT_DIRS.values():
        (output_root / dirname).mkdir(parents=True, exist_ok=True)

    cases = sorted(p for p in input_root.iterdir() if p.is_dir())
    names = []
    failed = []
    for case_dir in tqdm(cases, desc="cases"):
        try:
            data = load_case(case_dir)
            indices = valid_slices(data, args.min_nonzero_ratio)
            if len(indices) > args.slices_per_case:
                indices = sorted(random.sample(indices, args.slices_per_case))
            for z in indices:
                slice_id = f"{case_dir.name}_z{z:03d}"
                for modality in INPUT_MODALITIES:
                    arr = data[modality][:, :, z]
                    arr = normalize_to_uint8(arr, is_mask=(modality == "seg"))
                    Image.fromarray(arr).save(output_root / OUTPUT_DIRS[modality] / f"{slice_id}.png")
                names.append(slice_id)
        except Exception as exc:  # keep preprocessing robust across partially downloaded datasets
            failed.append(f"{case_dir.name}: {exc}")

    (output_root / "slice_list.txt").write_text("\n".join(names), encoding="utf-8")
    if failed:
        (output_root / "failed_cases.txt").write_text("\n".join(failed), encoding="utf-8")
    print(f"Saved {len(names)} slices to {output_root}")
    if failed:
        print(f"Skipped {len(failed)} cases. See failed_cases.txt")


if __name__ == "__main__":
    main()
