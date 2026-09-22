"""
3D Volumetric Segmentation Architectures: Custom 3D U-Net and MONAI SegResNet.
"""

import torch
import torch.nn as nn
from typing import Sequence, Optional
from monai.networks.nets import UNet, SegResNet


class DoubleConv3D(nn.Module):
    """(Conv3D -> InstanceNorm3d -> LeakyReLU) * 2 with optional residual connection."""
    
    def __init__(self, in_channels: int, out_channels: int, residual: bool = True):
        super().__init__()
        self.residual = residual and (in_channels == out_channels)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return x + self.conv(x)
        return self.conv(x)


class DownBlock3D(nn.Module):
    """Downscaling with MaxPool3D then DoubleConv3D."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(in_channels, out_channels, residual=False),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock3D(nn.Module):
    """Upscaling with ConvTranspose3d then DoubleConv3D with skip connections."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels, residual=False)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Concatenate along channel dimension
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class CustomUNet3D(nn.Module):
    """
    Modular 3D U-Net for medical volumetric segmentation.
    
    Args:
        in_channels: Number of input modalities (e.g. 4 for BraTS: FLAIR, T1, T1ce, T2).
        out_channels: Number of output segmentation classes (e.g. 4: BG, NCR, ED, ET).
        features: Feature channels across hierarchy levels (default: [16, 32, 64, 128]).
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        features: Sequence[int] = (16, 32, 64, 128)
    ):
        super().__init__()
        self.inc = DoubleConv3D(in_channels, features[0], residual=False)
        
        self.down1 = DownBlock3D(features[0], features[1])
        self.down2 = DownBlock3D(features[1], features[2])
        self.down3 = DownBlock3D(features[2], features[3])
        
        self.up1 = UpBlock3D(features[3], features[2])
        self.up2 = UpBlock3D(features[2], features[1])
        self.up3 = UpBlock3D(features[1], features[0])
        
        self.outc = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


def get_model(
    model_name: str = "custom_unet",
    in_channels: int = 4,
    out_channels: int = 4,
    spatial_dims: int = 3
) -> nn.Module:
    """
    Model factory to build either Custom 3D UNet or MONAI UNet / SegResNet.
    
    Args:
        model_name: "custom_unet", "monai_unet", or "segresnet".
        in_channels: Number of input channels.
        out_channels: Number of output classes.
        spatial_dims: 3 for 3D volumetric data.
        
    Returns:
        nn.Module
    """
    if model_name == "custom_unet":
        return CustomUNet3D(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "monai_unet":
        return UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
            norm="INSTANCE",
        )
    elif model_name == "segresnet":
        return SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=16,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
        )
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Choose from 'custom_unet', 'monai_unet', 'segresnet'.")
