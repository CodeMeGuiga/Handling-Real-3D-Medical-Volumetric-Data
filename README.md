# Handling Real 3D Medical Volumetric Data

An end-to-end project for understanding 3D medical imaging and brain-tumour
segmentation with NIfTI, NiBabel, MONAI, and PyTorch.

## What this project covers

- NIfTI volumes, voxel spacing, affine matrices, and anatomical orientations
- Multi-modal brain MRI (FLAIR, T1, T1ce, and T2)
- 3D preprocessing and augmentation with MONAI
- A modular 3D U-Net and MONAI SegResNet
- Dice + cross-entropy training and sliding-window inference
- Orthogonal slice, montage, MIP, and 3D visualisation

## Repository layout

```text
notebooks/       Guided exploration and segmentation notebooks
src/             Reusable loading, transforms, models, training, and visualisation code
results/         Selected generated visualisations
download_data.py Download script for the public MSD BraTS dataset
Documentation.md Extended project walkthrough
requirements.txt Python dependencies
```

The dataset is intentionally not included in this repository. It is downloaded
locally by the script below and remains ignored by Git.

## Quickstart

```bash
pip install -r requirements.txt
python download_data.py
jupyter lab
```

Open `notebooks/01_nifti_and_affine_exploration.ipynb` first, followed by
`notebooks/02_monai_3d_segmentation.ipynb`.

Training can also be started from the command line:

```bash
python -m src.train --epochs 20 --batch_size 2 --lr 0.0002
```

## Dataset

This project uses the Medical Segmentation Decathlon Task01 BrainTumour dataset.
Please review and comply with the dataset's usage terms before downloading or
redistributing it.
