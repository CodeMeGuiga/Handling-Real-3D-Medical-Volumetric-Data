"""
Download and verify the Medical Segmentation Decathlon (MSD) Task01_BrainTumour (BraTS) dataset.
"""

import os
import sys
import json
import time
from pathlib import Path
import nibabel as nib
import numpy as np
from monai.apps import download_and_extract

# Dataset details from MSD
DATASET_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
DATASET_MD5 = "240a19d752f0d9e9101544901065d872"

def download_brats(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    task_dir = data_dir / "Task01_BrainTumour"
    tar_file = data_dir / "Task01_BrainTumour.tar"
    
    if task_dir.exists() and (task_dir / "dataset.json").exists():
        print(f"[+] Task01_BrainTumour dataset already exists at: {task_dir}")
    else:
        print(f"[>] Downloading MSD Task01_BrainTumour from {DATASET_URL}...")
        print(f"[>] Destination directory: {data_dir}")
        t0 = time.time()
        download_and_extract(
            url=DATASET_URL,
            filepath=str(tar_file),
            output_dir=str(data_dir),
            hash_val=DATASET_MD5,
            hash_type="md5",
        )
        print(f"[+] Download and extraction completed in {time.time() - t0:.1f}s")
    
    # Inspect dataset.json
    dataset_json_path = task_dir / "dataset.json"
    if dataset_json_path.exists():
        with open(dataset_json_path, "r") as f:
            metadata = json.load(f)
        
        print("\n" + "="*50)
        print("MSD Task01_BrainTumour Metadata:")
        print("="*50)
        print(f"Name: {metadata.get('name')}")
        print(f"Description: {metadata.get('description')}")
        print(f"Modalities: {metadata.get('modality')}")
        print(f"Labels: {metadata.get('labels')}")
        print(f"Num Training Samples: {len(metadata.get('training', []))}")
        print(f"Num Test Samples: {len(metadata.get('test', []))}")
        
        # Verify first training sample
        if metadata.get('training'):
            sample = metadata['training'][0]
            img_rel = sample['image'].replace("./", "")
            lbl_rel = sample['label'].replace("./", "")
            img_path = task_dir / img_rel
            lbl_path = task_dir / lbl_rel
            
            print("\n" + "="*50)
            print("First Sample Verification (Nibabel):")
            print("="*50)
            print(f"Image Path: {img_path}")
            img_obj = nib.load(str(img_path))
            print(f"Image Shape: {img_obj.shape} (Modalities, X, Y, Z)")
            print(f"Voxel Spacing (zooms): {img_obj.header.get_zooms()}")
            print(f"Data type: {img_obj.get_data_dtype()}")
            print(f"Affine Matrix:\n{img_obj.affine}")
            
            lbl_obj = nib.load(str(lbl_path))
            lbl_data = lbl_obj.get_fdata()
            print(f"Label Shape: {lbl_obj.shape}")
            print(f"Unique Label Values: {np.unique(lbl_data)}")
            print("="*50)
            print("[+] BraTS dataset is ready for exploration and training!")
    else:
        print(f"[!] Warning: dataset.json not found in {task_dir}")

if __name__ == "__main__":
    current_dir = Path(__file__).parent.resolve()
    data_directory = current_dir / "data"
    download_brats(data_directory)
