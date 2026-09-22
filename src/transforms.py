"""
MONAI data transformation pipelines for BraTS 3D multi-modal MRI volumes.
"""

from typing import Tuple, List, Optional
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotated,
    RandGaussianNoised,
    RandGaussianSmoothd,
    CastToTyped,
)
import torch


def get_brats_transforms(
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_samples_per_volume: int = 2,
    mode: str = "train"
) -> Compose:
    """
    Create MONAI transform pipeline for training or validation/inference.
    
    Args:
        roi_size: 3D spatial patch size for sub-volume cropping during training (e.g. 96x96x96 or 64x64x64).
        target_spacing: Target isotropic voxel spacing in mm (e.g. 1.0 x 1.0 x 1.0 mm^3).
        num_samples_per_volume: Number of positive/negative balanced random crops extracted per volume.
        mode: "train", "val", or "test".
        
    Returns:
        Compose: MONAI transformation pipeline.
    """
    keys = ["image", "label"] if mode in ["train", "val"] else ["image"]
    
    # Common preprocessing transforms (Loading, Orientation, Spacing, Normalization)
    base_transforms = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=target_spacing,
            mode=("bilinear", "nearest") if mode in ["train", "val"] else "bilinear",
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=keys, dtype=torch.float32),
    ]
    
    if mode == "train":
        # Label to integer class indices
        train_transforms = base_transforms + [
            CastToTyped(keys="label", dtype=torch.int64),
            # Balanced positive/negative patch cropping centered on tumor foreground vs background
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=2,
                neg=1,
                num_samples=num_samples_per_volume,
                image_key="image",
                image_threshold=0,
            ),
            # 3D spatial data augmentations
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotated(
                keys=["image", "label"],
                range_x=0.2,
                range_y=0.2,
                range_z=0.2,
                mode=("bilinear", "nearest"),
                prob=0.3,
            ),
            RandGaussianNoised(keys="image", prob=0.2, std=0.1),
        ]
        return Compose(train_transforms)
        
    elif mode == "val":
        val_transforms = base_transforms + [
            CastToTyped(keys="label", dtype=torch.int64),
        ]
        return Compose(val_transforms)
        
    else:  # "test" / "inference"
        return Compose(base_transforms)
