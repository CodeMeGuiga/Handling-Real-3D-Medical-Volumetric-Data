"""
Training and validation pipeline for 3D Brain Tumor Volumetric Segmentation.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Activations, Compose
from monai.utils import set_determinism

import matplotlib.pyplot as plt
import numpy as np

from .dataset import get_dataloaders
from .model import get_model
from .visualization import plot_orthogonal_slices


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    step = 0
    
    for batch_data in loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / max(step, 1)


def validate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
    dice_metric: DiceMetric,
    post_trans: Compose,
    device: torch.device
) -> float:
    """Run whole-volume sliding-window validation and compute foreground Dice score."""
    model.eval()
    dice_metric.reset()
    
    with torch.no_grad():
        for batch_data in loader:
            val_images, val_labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            # Sliding window inference on full-resolution 3D volume
            val_outputs = sliding_window_inference(
                inputs=val_images,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=0.5,
            )
            
            # Apply post transforms to convert logits -> one-hot predictions
            val_outputs_list = [post_trans(i) for i in torch.unbind(val_outputs, dim=0)]
            # Convert integer labels to one-hot for Dice metric comparison
            val_labels_list = [
                AsDiscrete(to_onehot=4)(i) for i in torch.unbind(val_labels, dim=0)
            ]
            
            dice_metric(y_pred=val_outputs_list, y=val_labels_list)
            
    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    return metric


def run_training(
    data_dir: Path,
    output_dir: Path,
    model_name: str = "custom_unet",
    epochs: int = 25,
    batch_size: int = 2,
    lr: float = 2e-4,
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 2,
    val_interval: int = 1,
    max_samples: Optional[int] = None,
    seed: int = 42
):
    """Full training pipeline."""
    set_determinism(seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 1. DataLoaders
    print(f"[>] Preparing BraTS DataLoaders from {data_dir}...")
    train_loader, val_loader, metadata = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        roi_size=roi_size,
        val_ratio=0.2,
        num_samples_per_volume=2,
        max_samples=max_samples
    )
    
    # 2. Model & Optimization
    in_channels = len(metadata.get("modality", {"0": "FLAIR", "1": "T1", "2": "T1ce", "3": "T2"}))
    out_channels = len(metadata.get("labels", {"0": "BG", "1": "NCR", "2": "ED", "3": "ET"}))
    print(f"[+] Initializing {model_name} (in_channels={in_channels}, out_channels={out_channels})...")
    model = get_model(model_name=model_name, in_channels=in_channels, out_channels=out_channels).to(device)
    
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=4)])
    
    best_metric = -1.0
    best_epoch = -1
    epoch_loss_values = []
    val_metric_values = []
    
    print("\n" + "=" * 65)
    print(f" Starting 3D Segmentation Training ({epochs} Epochs)")
    print("=" * 65)
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_function, device)
        lr_scheduler.step()
        epoch_loss_values.append(train_loss)
        elapsed = time.time() - t0
        
        if epoch % val_interval == 0:
            val_dice = validate_epoch(
                model, val_loader, roi_size, sw_batch_size, dice_metric, post_trans, device
            )
            val_metric_values.append((epoch, val_dice))
            
            is_best = val_dice > best_metric
            if is_best:
                best_metric = val_dice
                best_epoch = epoch
                torch.save(model.state_dict(), output_dir / "best_metric_model.pth")
                
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Mean Dice: {val_dice:.4f} ({'* New Best' if is_best else '          '}) | Time: {elapsed:.1f}s"
            )
        else:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Time: {elapsed:.1f}s")
            
    total_time = time.time() - start_time
    print("=" * 65)
    print(f"[+] Training completed in {total_time/60:.2f} minutes.")
    print(f"[*] Best Validation Dice Score: {best_metric:.4f} at epoch {best_epoch}")
    
    # Save training metrics plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), epoch_loss_values, label="Train Loss", color="royalblue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Dice + CE)")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    val_eps, val_dices = zip(*val_metric_values) if val_metric_values else ([], [])
    plt.plot(val_eps, val_dices, label="Val Dice", color="forestgreen", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice Score")
    plt.title("Validation Dice Score")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()
    
    # Save training log JSON
    history = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "roi_size": roi_size,
        "best_epoch": best_epoch,
        "best_dice": float(best_metric),
        "total_training_seconds": total_time,
        "train_loss_history": epoch_loss_values,
        "val_dice_history": val_metric_values,
    }
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
        
    print(f"[+] Model checkpoint and curves saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Brain Tumor Segmentation Model")
    parser.add_argument("--data_dir", type=str, default="data/Task01_BrainTumour", help="Path to Task01_BrainTumour")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save model checkpoints")
    parser.add_argument("--model", type=str, default="custom_unet", choices=["custom_unet", "monai_unet", "segresnet"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset samples for fast testing")
    
    args = parser.parse_args()
    run_training(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples
    )
