# NAP-SCAF

**Similarity-aware feature calibration stabilizes low-contrast and missing-modality brain segmentation**

This repository provides a clean PyTorch implementation of **NAP-SCAF**, a similarity-aware calibration framework for multi-modal brain tumor segmentation. The implementation focuses on the core method used in the paper: a non-local adaptive prior branch, local normalized cross-correlation based skip calibration, and contrast-enhanced region attention.

NAP-SCAF is designed for challenging multi-modal MRI segmentation scenarios where the input modalities may be incomplete, local lesion boundaries are weak, and naïve U-shaped skip fusion can propagate semantically inconsistent encoder features into the decoder.
![Overview of the proposed NAP-SCAF framework](./method_overview.jpg)
## Highlights

- **Non-local Adaptive Prior (NAP)**  
  A variational reconstruction branch generates anatomy-consistent prior features and injects a multi-scale structural reference into the decoder.

- **Similarity-Weighted Gating (SWG)**  
  Skip transmission is controlled by local normalized cross-correlation between the prior-conditioned decoder state and encoder features. High structural agreement allows skip details to pass, while low agreement attenuates unreliable high-resolution textures.

- **Contrast-Enhanced Region Attention (CERA)**  
  Multi-rate depthwise convolutions amplify weak boundary evidence before similarity calibration, improving stability in low-contrast tumor regions.

- **Missing-modality evaluation support**  
  Any subset of input modalities can be zero-filled at test time without changing the model weights.

## Repository structure

```text
NAP_SCAF_clean/
├── nap_scaf/
│   ├── models/
│   │   └── nap_scaf.py          # NAP-SCAF model, NAP, SWG, CERA
│   ├── data/
│   │   └── brats_png.py         # BraTS PNG slice dataset
│   ├── losses.py                # Dice, CE, reconstruction, and KL losses
│   ├── metrics.py               # WT, TC, ET Dice and HD95 metrics
│   └── utils.py                 # checkpoint, seed, JSON utilities
├── scripts/
│   ├── preprocess_brats_png.py  # NIfTI-to-PNG preprocessing
│   ├── train_brats.py           # training entry point
│   ├── evaluate_brats.py        # full and missing-modality evaluation
│   └── infer_slice.py           # inference on one 2D multi-modal slice
├── checkpoints/                 # checkpoint output directory
├── results/                     # evaluation and inference outputs
├── requirements.txt
└── README.md
```

## Installation

Create a new environment and install the dependencies.

```bash
conda create -n nap_scaf python=3.10 -y
conda activate nap_scaf

# Install PyTorch according to your CUDA version from the official PyTorch website.
# Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

The code was written for PyTorch 2.x. CPU execution is supported for debugging, but training the full model requires a CUDA GPU.

## Data preparation

### Expected PNG layout

After preprocessing, the dataset root should follow this structure:

```text
BraTS2021_PNG/
├── flair/
│   ├── BraTS2021_00000_z080.png
│   └── ...
├── t1/
├── t1ce/
├── t2/
├── mask/
└── slice_list.txt
```

The four image folders store grayscale PNG slices. The mask folder stores segmentation labels. The loader accepts either original BraTS labels `{0, 1, 2, 4}` or contiguous labels `{0, 1, 2, 3}`. During loading, labels are mapped to:

| Label | Meaning |
|---:|---|
| 0 | Background |
| 1 | NCR/NET |
| 2 | ED |
| 3 | ET |

The BraTS evaluation regions are computed as:

| Region | Labels |
|---|---|
| WT | 1, 2, 3 |
| TC | 1, 3 |
| ET | 3 |

### Convert BraTS NIfTI volumes to PNG slices

The preprocessing script expects each case folder to contain files named in the standard BraTS format:

```text
BraTS2021_00000/
├── BraTS2021_00000_flair.nii.gz
├── BraTS2021_00000_t1.nii.gz
├── BraTS2021_00000_t1ce.nii.gz
├── BraTS2021_00000_t2.nii.gz
└── BraTS2021_00000_seg.nii.gz
```

Run:

```bash
python scripts/preprocess_brats_png.py \
  --input-dir /path/to/BraTS2021_TrainingData \
  --output-dir data/BraTS2021_PNG \
  --slices-per-case 5 \
  --seed 7
```

The script samples tumor-containing slices, saves aligned four-modality PNGs, writes `slice_list.txt`, and records skipped cases in `failed_cases.txt` when necessary.

## Training

Train NAP-SCAF on the prepared PNG dataset:

```bash
python scripts/train_brats.py \
  --data-root data/BraTS2021_PNG \
  --output-dir checkpoints/nap_scaf_brats2021 \
  --epochs 150 \
  --batch-size 16 \
  --val-batch-size 4 \
  --lr 1e-4 \
  --base-channels 16 \
  --stages 4 \
  --latent-channels 1024 \
  --ncc-window 5 \
  --amp
```

To include the variational reconstruction and KL terms from the NAP branch, add:

```bash
--use-nap-loss --rec-weight 0.05 --kl-weight 1e-4
```

The script saves:

```text
checkpoints/nap_scaf_brats2021/
├── best.pth
├── last.pth
└── history.json
```

The default optimizer is AdamW. The default training split uses an 80/20 random split with seed 7. For official benchmark reporting, replace this split with the official validation split used in your experimental protocol.

## Evaluation

Evaluate the full-modality model:

```bash
python scripts/evaluate_brats.py \
  --data-root data/BraTS2021_PNG \
  --checkpoint checkpoints/nap_scaf_brats2021/best.pth \
  --output results/brats2021_full.json
```

The output JSON contains WT, TC, and ET Dice/HD95 together with averaged metrics.

## Missing-modality evaluation

To test robustness under incomplete inputs, specify one or more modalities to zero-fill:

```bash
python scripts/evaluate_brats.py \
  --data-root data/BraTS2021_PNG \
  --checkpoint checkpoints/nap_scaf_brats2021/best.pth \
  --missing-modalities t1,t2 \
  --output results/brats2021_missing_t1_t2.json
```

Available modality names are:

```text
flair, t1, t1ce, t2
```

The model architecture and checkpoint remain unchanged during missing-modality evaluation.

## Single-slice inference

Run inference on one four-modality 2D slice:

```bash
python scripts/infer_slice.py \
  --flair sample/flair.png \
  --t1 sample/t1.png \
  --t1ce sample/t1ce.png \
  --t2 sample/t2.png \
  --checkpoint checkpoints/nap_scaf_brats2021/best.pth \
  --output results/prediction.png
```

The saved prediction uses contiguous class labels `0, 1, 2, 3`.

## Model usage in Python

```python
import torch
from nap_scaf import build_nap_scaf

model = build_nap_scaf(
    in_channels=4,
    num_classes=4,
    base_channels=16,
    stages=4,
    latent_channels=1024,
    ncc_window=5,
)

x = torch.randn(2, 4, 256, 256)
logits = model(x)
print(logits.shape)  # (2, 4, 256, 256)
```

To access the NAP reconstruction and gate maps during training or visualization:

```python
output = model(x, return_aux=True)
logits = output["logits"]
reconstruction = output["reconstruction"]
gates = output["gates"]
```

## Reproducibility notes

1. Keep modality order fixed as `FLAIR, T1, T1ce, T2`.
2. Use nearest-neighbor interpolation for segmentation masks.
3. Use the same validation split when comparing with baselines.
4. For missing-modality testing, zero-fill absent modalities and keep the same checkpoint.
5. HD95 is sensitive to empty predictions. This implementation returns `nan` for one-sided empty masks and averages with `nanmean`.

## Cleaned codebase policy

This GitHub version intentionally removes temporary files from the original research workspace, including local CSV result dumps, Visio drafts, one-off Grad-CAM experiments, old baseline scripts, duplicate ablation scripts, and machine-specific Windows paths. The retained code is organized around a reproducible pipeline: preprocessing, model construction, training, evaluation, and inference.

## Citation

If this repository is useful for your research, please cite the paper:

```bibtex
@article{zhou2026napscaf,
  title   = {Similarity-aware feature calibration stabilizes low-contrast and missing-modality brain segmentation},
  author  = {Zhou, Meihua and Cheng, Min and Yang, Li and Liu, Yuting},
  journal = {Information Processing and Management},
  volume  = {63},
  pages   = {104960},
  year    = {2026}
}
```

## Acknowledgement

This repository is a cleaned research-code release. It is intended to make the main NAP-SCAF pipeline easier to inspect, run, and extend.

Note: This repository is a cleaned and refactored version of the original research code. Some experimental, redundant, and project-specific files were removed for clarity. AI assistance was used only for code organization and documentation polishing; the core model design and experimental logic follow the NAP-SCAF paper.
