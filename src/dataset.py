"""
Dataset loading, JSON parsing, and DataLoader creation for BraTS volumetric data.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from sklearn.model_selection import train_test_split
from monai.data import Dataset, CacheDataset, DataLoader
from .transforms import get_brats_transforms


def load_brats_datalists(
    data_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    max_samples: Optional[int] = None
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict]:
    """
    Parse MSD dataset.json and generate train/val data lists of dicts: {'image': path, 'label': path}.
    
    Args:
        data_dir: Path to Task01_BrainTumour folder containing dataset.json.
        val_ratio: Proportion of training samples to allocate for validation.
        seed: Random seed for split reproducibility.
        max_samples: Optional limit on total samples for fast experiments/debugging.
        
    Returns:
        tuple: (train_files, val_files, metadata_dict)
    """
    json_path = data_dir / "dataset.json"
    if not json_path.exists():
        raise FileNotFoundError(f"dataset.json not found at: {json_path}")
        
    with open(json_path, "r") as f:
        metadata = json.load(f)
        
    raw_training = metadata.get("training", [])
    if max_samples is not None and max_samples > 0:
        raw_training = raw_training[:max_samples]
        
    formatted_data = []
    for item in raw_training:
        img_rel = item["image"].replace("./", "")
        lbl_rel = item["label"].replace("./", "")
        
        img_path = str((data_dir / img_rel).resolve())
        lbl_path = str((data_dir / lbl_rel).resolve())
        
        formatted_data.append({
            "image": img_path,
            "label": lbl_path,
            "id": Path(img_path).stem.replace(".nii", "")
        })
        
    train_files, val_files = train_test_split(
        formatted_data, test_size=val_ratio, random_state=seed, shuffle=True
    )
    
    return train_files, val_files, metadata


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 2,
    val_batch_size: int = 1,
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    val_ratio: float = 0.2,
    num_samples_per_volume: int = 2,
    num_workers: int = 0,
    cache_rate: float = 0.0,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create training and validation DataLoaders for BraTS volumes.
    
    Args:
        data_dir: Path to Task01_BrainTumour directory.
        batch_size: Training batch size.
        val_batch_size: Validation batch size (usually 1 for whole-volume inference).
        roi_size: Patch size extracted per volume during training.
        val_ratio: Validation split fraction.
        num_samples_per_volume: Number of cropped patches per volume in training.
        num_workers: Number of DataLoader workers.
        cache_rate: Fraction of dataset to cache in RAM (0.0 for streaming, 1.0 for all).
        max_samples: Max subjects to use (useful for debugging).
        
    Returns:
        tuple: (train_loader, val_loader, metadata)
    """
    train_files, val_files, metadata = load_brats_datalists(
        data_dir, val_ratio=val_ratio, max_samples=max_samples
    )
    
    train_transforms = get_brats_transforms(
        roi_size=roi_size, num_samples_per_volume=num_samples_per_volume, mode="train"
    )
    val_transforms = get_brats_transforms(roi_size=roi_size, mode="val")
    
    if cache_rate > 0.0:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=cache_rate, num_workers=num_workers)
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)
        
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, metadata
