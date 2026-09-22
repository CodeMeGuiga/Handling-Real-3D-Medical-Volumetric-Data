"""
Visualization utilities for 3D medical volumetric data.
Includes publication-quality static multi-planar orthogonal reconstruction (MPR),
axial volumetric slice montages, multi-parametric MRI contrast comparisons,
static 3D mesh surface reconstructions (multi-angle), maximum intensity projections (MIP),
affine coordinate space diagrams, and segmentation evaluation dashboards.
"""

from typing import Optional, Tuple, List, Union, Dict, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Try importing 3D mesh & Plotly tools
try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import ipywidgets as widgets
    from IPython.display import display
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False


# Standard BraTS color mapping for labels:
# 0 = Background
# 1 = Necrotic and Non-Enhancing Tumor Core (NCR/NET)
# 2 = Peritumoral Edema (ED)
# 3 = GD-Enhancing Tumor (ET)
BRATS_LABEL_NAMES = {
    0: "Background",
    1: "Necrotic Core (NCR)",
    2: "Peritumoral Edema (ED)",
    3: "Enhancing Tumor (ET)"
}

# Publication-grade colors:
# Transparent for background, Crimson Red for NCR, Vibrant Green for ED, Bright Gold for ET
BRATS_COLORS = [
    (0.0, 0.0, 0.0, 0.0),       # 0: Background (transparent)
    (0.90, 0.15, 0.15, 0.75),    # 1: NCR - Crimson Red
    (0.12, 0.75, 0.25, 0.70),    # 2: ED  - Emerald Green
    (1.00, 0.80, 0.05, 0.85)     # 3: ET  - Bright Gold/Amber
]
BRATS_CMAP = ListedColormap(BRATS_COLORS)

# Hex colors for legends
BRATS_HEX_COLORS = {
    1: "#e62626",  # Crimson Red
    2: "#1ebf40",  # Emerald Green
    3: "#ffcc0d"   # Bright Gold
}


def _extract_3d_volume(volume: np.ndarray, channel: int = 0) -> np.ndarray:
    """Helper to extract a 3D numpy array from a 3D or 4D volume."""
    if volume.ndim == 4:
        if volume.shape[0] <= 10:  # Channel-first (C, X, Y, Z)
            return volume[channel]
        else:  # Channel-last (X, Y, Z, C)
            return volume[..., channel]
    elif volume.ndim == 3:
        return volume
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {volume.shape}")


def _find_tumor_centroid(mask: np.ndarray, default_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Calculate the centroid (sx, sy, sz) of foreground voxels in a 3D mask."""
    if mask is not None and np.any(mask > 0):
        fg_coords = np.argwhere(mask > 0)
        sx, sy, sz = np.round(fg_coords.mean(axis=0)).astype(int)
        nx, ny, nz = mask.shape[:3]
        return (
            int(np.clip(sx, 0, nx - 1)),
            int(np.clip(sy, 0, ny - 1)),
            int(np.clip(sz, 0, nz - 1))
        )
    nx, ny, nz = default_shape[:3]
    return nx // 2, ny // 2, nz // 2


def plot_orthogonal_slices(
    volume: np.ndarray,
    slices: Optional[Tuple[int, int, int]] = None,
    channel: int = 0,
    mask: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
    title: str = "3D Multi-Planar Orthogonal Reconstruction (MPR)",
    cmap_img: str = "gray",
    figsize: Tuple[int, int] = (16, 5.5),
    alpha_mask: float = 0.65,
    show_crosshairs: bool = True,
    show_labels: bool = True,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Publication-quality Orthogonal Multi-Planar Reconstruction (Sagittal, Coronal, Axial).
    Automatically centers on the pathology centroid if slices is not specified,
    adds radiological orientation annotations (S/I, A/P, L/R), mm physical coordinates,
    and a discrete BraTS subregion legend.
    """
    vol_3d = _extract_3d_volume(volume, channel)
    nx, ny, nz = vol_3d.shape

    if slices is None:
        if mask is not None:
            sx, sy, sz = _find_tumor_centroid(mask, vol_3d.shape)
        else:
            sx, sy, sz = nx // 2, ny // 2, nz // 2
    else:
        sx, sy, sz = slices

    # Compute physical coordinates in mm if affine matrix is provided
    if affine is not None:
        origin_voxel = np.array([sx, sy, sz, 1.0])
        world_pt = affine @ origin_voxel
        x_mm, y_mm, z_mm = world_pt[:3]
        coord_text = f"Focal Point: Voxel ({sx}, {sy}, {sz}) | Scanner: ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm"
    else:
        coord_text = f"Focal Point: Voxel ({sx}, {sy}, {sz})"

    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="#0e1117")

    # Styling helper for anatomical orientation text
    text_kwargs = dict(color="#00f0ff", fontsize=11, fontweight="bold", alpha=0.9)

    # 1. Sagittal View (YZ plane, fixed X) - rotate 90 deg for standard anatomical orientation
    sag_img = np.rot90(vol_3d[sx, :, :])
    axes[0].imshow(sag_img, cmap=cmap_img, aspect='equal')
    axes[0].set_title(f"Sagittal View (X = {sx} / {nx})", fontsize=12, fontweight="bold", color="white", pad=8)
    axes[0].axis('off')
    if mask is not None:
        sag_mask = np.rot90(mask[sx, :, :])
        axes[0].imshow(sag_mask, cmap=BRATS_CMAP, vmin=0, vmax=3, alpha=alpha_mask, interpolation='nearest')
    if show_crosshairs:
        axes[0].axvline(sy, color='#00e5ff', linestyle='--', linewidth=1.2, alpha=0.7)
        axes[0].axhline(nz - 1 - sz, color='#ffea00', linestyle='--', linewidth=1.2, alpha=0.7)
    if show_labels:
        axes[0].text(0.5, 0.96, "S", transform=axes[0].transAxes, ha="center", **text_kwargs)
        axes[0].text(0.5, 0.04, "I", transform=axes[0].transAxes, ha="center", **text_kwargs)
        axes[0].text(0.04, 0.5, "A", transform=axes[0].transAxes, va="center", **text_kwargs)
        axes[0].text(0.96, 0.5, "P", transform=axes[0].transAxes, va="center", **text_kwargs)

    # 2. Coronal View (XZ plane, fixed Y)
    cor_img = np.rot90(vol_3d[:, sy, :])
    axes[1].imshow(cor_img, cmap=cmap_img, aspect='equal')
    axes[1].set_title(f"Coronal View (Y = {sy} / {ny})", fontsize=12, fontweight="bold", color="white", pad=8)
    axes[1].axis('off')
    if mask is not None:
        cor_mask = np.rot90(mask[:, sy, :])
        axes[1].imshow(cor_mask, cmap=BRATS_CMAP, vmin=0, vmax=3, alpha=alpha_mask, interpolation='nearest')
    if show_crosshairs:
        axes[1].axvline(sx, color='#ff007f', linestyle='--', linewidth=1.2, alpha=0.7)
        axes[1].axhline(nz - 1 - sz, color='#ffea00', linestyle='--', linewidth=1.2, alpha=0.7)
    if show_labels:
        axes[1].text(0.5, 0.96, "S", transform=axes[1].transAxes, ha="center", **text_kwargs)
        axes[1].text(0.5, 0.04, "I", transform=axes[1].transAxes, ha="center", **text_kwargs)
        axes[1].text(0.04, 0.5, "R", transform=axes[1].transAxes, va="center", **text_kwargs)
        axes[1].text(0.96, 0.5, "L", transform=axes[1].transAxes, va="center", **text_kwargs)

    # 3. Axial View (XY plane, fixed Z)
    ax_img = np.rot90(vol_3d[:, :, sz])
    axes[2].imshow(ax_img, cmap=cmap_img, aspect='equal')
    axes[2].set_title(f"Axial View (Z = {sz} / {nz})", fontsize=12, fontweight="bold", color="white", pad=8)
    axes[2].axis('off')
    if mask is not None:
        ax_mask = np.rot90(mask[:, :, sz])
        axes[2].imshow(ax_mask, cmap=BRATS_CMAP, vmin=0, vmax=3, alpha=alpha_mask, interpolation='nearest')
    if show_crosshairs:
        axes[2].axvline(sx, color='#ff007f', linestyle='--', linewidth=1.2, alpha=0.7)
        axes[2].axhline(ny - 1 - sy, color='#00e5ff', linestyle='--', linewidth=1.2, alpha=0.7)
    if show_labels:
        axes[2].text(0.5, 0.96, "A", transform=axes[2].transAxes, ha="center", **text_kwargs)
        axes[2].text(0.5, 0.04, "P", transform=axes[2].transAxes, ha="center", **text_kwargs)
        axes[2].text(0.04, 0.5, "R", transform=axes[2].transAxes, va="center", **text_kwargs)
        axes[2].text(0.96, 0.5, "L", transform=axes[2].transAxes, va="center", **text_kwargs)

    # Legend for tumor labels if mask is present
    if mask is not None and np.any(mask > 0):
        legend_elements = [
            Patch(facecolor=BRATS_HEX_COLORS[1], edgecolor='white', label='Necrotic Core (NCR)', alpha=0.85),
            Patch(facecolor=BRATS_HEX_COLORS[2], edgecolor='white', label='Peritumoral Edema (ED)', alpha=0.85),
            Patch(facecolor=BRATS_HEX_COLORS[3], edgecolor='white', label='Enhancing Tumor (ET)', alpha=0.85)
        ]
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=3,
            fontsize=10,
            frameon=True,
            facecolor='#1b202c',
            edgecolor='#374151',
            labelcolor='white',
            bbox_to_anchor=(0.5, -0.04)
        )

    fig.suptitle(f"{title}\n{coord_text}", fontsize=14, fontweight="bold", color="white", y=1.04)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=180, facecolor=fig.get_facecolor())

    return fig


def plot_volumetric_montage(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    channel: int = 0,
    affine: Optional[np.ndarray] = None,
    num_slices: int = 10,
    z_range: Optional[Tuple[int, int]] = None,
    title: str = "Axial Volumetric Slice Montage Across Brain Depth",
    cmap_img: str = "gray",
    alpha_mask: float = 0.65,
    figsize: Tuple[int, int] = (18, 7.5),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Static multi-slice axial gallery covering the full 3D extent of the brain/tumor.
    Provides complete anatomical context without requiring slider navigation.
    """
    vol_3d = _extract_3d_volume(volume, channel)
    nx, ny, nz = vol_3d.shape

    # Determine slice range
    if z_range is not None:
        z_min, z_max = z_range
    elif mask is not None and np.any(mask > 0):
        tumor_z = np.argwhere(mask > 0)[:, 2]
        pad = max(4, int(0.2 * (tumor_z.max() - tumor_z.min() + 1)))
        z_min = max(0, tumor_z.min() - pad)
        z_max = min(nz - 1, tumor_z.max() + pad)
    else:
        z_min = int(0.2 * nz)
        z_max = int(0.8 * nz)

    slice_indices = np.linspace(z_min, z_max, num_slices, dtype=int)
    cols = min(5, num_slices)
    rows = int(np.ceil(num_slices / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize, facecolor="#0e1117")
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, z in enumerate(slice_indices):
        ax = axes_flat[idx]
        img_slice = np.rot90(vol_3d[:, :, z])
        ax.imshow(img_slice, cmap=cmap_img, aspect='equal')

        has_tumor_in_slice = False
        if mask is not None:
            mask_slice = np.rot90(mask[:, :, z])
            if np.any(mask_slice > 0):
                has_tumor_in_slice = True
                ax.imshow(mask_slice, cmap=BRATS_CMAP, vmin=0, vmax=3, alpha=alpha_mask, interpolation='nearest')

        # Annotation text with mm coordinate if affine exists
        if affine is not None:
            z_mm = (affine @ np.array([0, 0, z, 1.0]))[2]
            label = f"Z={z} ({z_mm:.1f} mm)"
        else:
            label = f"Axial Z={z}"

        title_color = "#ffdd57" if has_tumor_in_slice else "#a0aec0"
        if has_tumor_in_slice:
            label += " ★ Tumor"
        ax.set_title(label, fontsize=10, fontweight="bold", color=title_color, pad=4)
        ax.axis('off')

    # Turn off unused subplots
    for j in range(num_slices, len(axes_flat)):
        axes_flat[j].axis('off')

    if mask is not None and np.any(mask > 0):
        legend_elements = [
            Patch(facecolor=BRATS_HEX_COLORS[1], edgecolor='white', label='Necrotic Core (NCR)', alpha=0.85),
            Patch(facecolor=BRATS_HEX_COLORS[2], edgecolor='white', label='Peritumoral Edema (ED)', alpha=0.85),
            Patch(facecolor=BRATS_HEX_COLORS[3], edgecolor='white', label='Enhancing Tumor (ET)', alpha=0.85)
        ]
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=3,
            fontsize=10,
            frameon=True,
            facecolor='#1b202c',
            edgecolor='#374151',
            labelcolor='white',
            bbox_to_anchor=(0.5, -0.03)
        )

    fig.suptitle(title, fontsize=14, fontweight="bold", color="white", y=1.01)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=180, facecolor=fig.get_facecolor())

    return fig


def plot_multimodal_comparison(
    volume_4d: np.ndarray,
    slice_z: Optional[int] = None,
    modality_names: Optional[List[str]] = None,
    mask: Optional[np.ndarray] = None,
    title: str = "Multi-Parametric MRI Modality Contrast Matrix",
    figsize: Tuple[int, int] = (19, 4.5),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Publication-grade side-by-side comparison across all MRI modalities (FLAIR, T1, T1ce, T2),
    plus anatomical tumor overlay and isolated multi-class segmentation mask with clinical notes.
    """
    if volume_4d.ndim != 4:
        raise ValueError(f"Expected 4D array, got shape {volume_4d.shape}")

    if volume_4d.shape[-1] <= 10 and volume_4d.shape[0] > 10:
        volume_4d = np.moveaxis(volume_4d, -1, 0)

    num_channels, _, _, nz = volume_4d.shape

    if slice_z is None:
        if mask is not None and np.any(mask > 0):
            # Pick slice with maximum tumor content
            tumor_per_z = [np.sum(mask[:, :, z] > 0) for z in range(nz)]
            slice_z = int(np.argmax(tumor_per_z))
        else:
            slice_z = nz // 2

    # Standard default names and diagnostic clinical roles
    clinical_subtitles = [
        "Peritumoral Edema / Hyperintensity",
        "Anatomical Architecture / Ventricles",
        "Vascular Enhancing Core (BBB Breach)",
        "Lesion Boundary / Water Content"
    ]
    if modality_names is None:
        modality_names = ["FLAIR", "T1 Native", "T1ce (Contrast)", "T2"]

    # If mask is provided, create 6 panels: 4 modalities + T1ce Overlay + Ground Truth Mask
    num_cols = num_channels + (2 if mask is not None else 0)
    fig, axes = plt.subplots(1, num_cols, figsize=figsize, facecolor="#0e1117")

    for c in range(num_channels):
        ax = axes[c]
        slice_data = np.rot90(volume_4d[c, :, :, slice_z])
        ax.imshow(slice_data, cmap='gray')
        name = modality_names[c] if c < len(modality_names) else f"Channel {c}"
        role = clinical_subtitles[c] if c < len(clinical_subtitles) else ""
        ax.set_title(f"{name}\n({role})", fontsize=10, fontweight="bold", color="white", pad=6)
        ax.axis('off')

    if mask is not None:
        # Panel 5: T1ce with Tumor Overlay
        t1ce_idx = min(2, num_channels - 1)
        t1ce_slice = np.rot90(volume_4d[t1ce_idx, :, :, slice_z])
        mask_slice = np.rot90(mask[:, :, slice_z])

        ax_overlay = axes[num_channels]
        ax_overlay.imshow(t1ce_slice, cmap='gray')
        ax_overlay.imshow(mask_slice, cmap=BRATS_CMAP, vmin=0, vmax=3, alpha=0.7, interpolation='nearest')
        ax_overlay.set_title(f"T1ce + Tumor Overlay\n(Multiclass Segmentation)", fontsize=10, fontweight="bold", color="#ffdd57", pad=6)
        ax_overlay.axis('off')

        # Panel 6: Isolated Segmentation Mask
        ax_mask = axes[num_channels + 1]
        ax_mask.imshow(np.zeros_like(mask_slice), cmap='gray', vmin=0, vmax=1)
        ax_mask.imshow(mask_slice, cmap=BRATS_CMAP, vmin=0, vmax=3, alpha=0.9, interpolation='nearest')
        ax_mask.set_title("BraTS Labels\n(1=NCR, 2=ED, 3=ET)", fontsize=10, fontweight="bold", color="#00ffcc", pad=6)
        ax_mask.axis('off')

        legend_elements = [
            Patch(facecolor=BRATS_HEX_COLORS[1], edgecolor='white', label='Necrotic Core (NCR)', alpha=0.85),
            Patch(facecolor=BRATS_HEX_COLORS[2], edgecolor='white', label='Peritumoral Edema (ED)', alpha=0.85),
            Patch(facecolor=BRATS_HEX_COLORS[3], edgecolor='white', label='Enhancing Tumor (ET)', alpha=0.85)
        ]
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=3,
            fontsize=10,
            frameon=True,
            facecolor='#1b202c',
            edgecolor='#374151',
            labelcolor='white',
            bbox_to_anchor=(0.5, -0.04)
        )

    fig.suptitle(f"{title} (Axial Z={slice_z} / {nz})", fontsize=13, fontweight="bold", color="white", y=1.04)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=180, facecolor=fig.get_facecolor())

    return fig


def plot_affine_coordinate_space(
    affine: np.ndarray,
    shape: Tuple[int, int, int],
    mask: Optional[np.ndarray] = None,
    title: str = "Mathematical Affine Transformation & 3D Physical Scanner Space",
    figsize: Tuple[int, int] = (16, 6.5),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Static visualization of the 4x4 NIfTI affine matrix mapping.
    Displays:
    1. 3D physical coordinate bounding box with patient orientation axes (RAS: Right, Anterior, Superior).
    2. Comprehensive landmark coordinate mapping card comparing discrete voxel grid indices (i, j, k)
       with continuous physical scanner space (x, y, z) in mm.
    """
    nx, ny, nz = shape[:3]
    fig = plt.figure(figsize=figsize, facecolor="#0e1117")
    ax_3d = fig.add_subplot(1, 2, 1, projection='3d', facecolor="#0e1117")
    ax_txt = fig.add_subplot(1, 2, 2, facecolor="#0e1117")

    # Corner voxels of the 3D bounding box
    corners_voxel = np.array([
        [0, 0, 0], [nx, 0, 0], [nx, ny, 0], [0, ny, 0],
        [0, 0, nz], [nx, 0, nz], [nx, ny, nz], [0, ny, nz]
    ])
    homo_corners = np.hstack([corners_voxel, np.ones((8, 1))])
    world_corners = (homo_corners @ affine.T)[:, :3]

    # Wireframe edges for bounding box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for e in edges:
        p1, p2 = world_corners[e[0]], world_corners[e[1]]
        ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='#4a5568', linestyle='--', linewidth=1.2)

    # Origin and center
    origin_world = (np.array([0, 0, 0, 1.0]) @ affine.T)[:3]
    center_world = (np.array([nx / 2, ny / 2, nz / 2, 1.0]) @ affine.T)[:3]

    ax_3d.scatter(*origin_world, color='#ff007f', s=70, label='Origin (0,0,0)', depthshade=False)
    ax_3d.scatter(*center_world, color='#00e5ff', s=70, label='Brain Center', depthshade=False)

    # Orientation vectors from origin (Right, Anterior, Superior in mm)
    v_r = (np.array([40, 0, 0, 1.0]) @ affine.T)[:3] - origin_world
    v_a = (np.array([0, 40, 0, 1.0]) @ affine.T)[:3] - origin_world
    v_s = (np.array([0, 0, 40, 1.0]) @ affine.T)[:3] - origin_world

    ax_3d.quiver(*origin_world, *v_r, color='#ff4d4d', arrow_length_ratio=0.15, linewidth=2, label='X / Right (R)')
    ax_3d.quiver(*origin_world, *v_a, color='#00ff7f', arrow_length_ratio=0.15, linewidth=2, label='Y / Anterior (A)')
    ax_3d.quiver(*origin_world, *v_s, color='#00c8ff', arrow_length_ratio=0.15, linewidth=2, label='Z / Superior (S)')

    # Tumor centroid if available
    if mask is not None and np.any(mask > 0):
        t_sx, t_sy, t_sz = _find_tumor_centroid(mask, shape)
        tumor_world = (np.array([t_sx, t_sy, t_sz, 1.0]) @ affine.T)[:3]
        ax_3d.scatter(*tumor_world, color='#ffd700', s=120, marker='*', label='Tumor Centroid', depthshade=False)

    ax_3d.set_xlabel('Physical X (mm)', color='white', labelpad=8)
    ax_3d.set_ylabel('Physical Y (mm)', color='white', labelpad=8)
    ax_3d.set_zlabel('Physical Z (mm)', color='white', labelpad=8)
    ax_3d.tick_params(colors='white')
    ax_3d.grid(color='#2d3748', linestyle=':', linewidth=0.7)
    ax_3d.view_init(elev=20, azim=45)
    ax_3d.legend(loc='upper right', facecolor='#1a202c', edgecolor='#4a5568', labelcolor='white', fontsize=8)
    ax_3d.set_title("3D Scanner Physical Bounding Box (RAS mm)", color="white", fontsize=11, fontweight="bold")

    # Right Panel: Mathematical Affine Mapping Table & Details
    ax_txt.axis('off')
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))

    landmarks = [
        ("Voxel Grid Origin", (0, 0, 0)),
        ("Volume Center", (nx // 2, ny // 2, nz // 2)),
        ("Bounding Max Corner", (nx - 1, ny - 1, nz - 1)),
    ]
    if mask is not None and np.any(mask > 0):
        landmarks.append(("Tumor Centroid", (t_sx, t_sy, t_sz)))

    text_lines = [
        "NIfTI 4x4 Spatial Affine Coordinate Mapping",
        "━" * 54,
        f"• Voxel Grid Dimensions (Nx, Ny, Nz)  : {nx} x {ny} x {nz}",
        f"• Physical Voxel Spacing (dx, dy, dz) : {spacing[0]:.2f} x {spacing[1]:.2f} x {spacing[2]:.2f} mm",
        f"• Physical Field of View (FOV)       : {nx*spacing[0]:.1f} x {ny*spacing[1]:.1f} x {nz*spacing[2]:.1f} mm",
        "",
        "Affine Transformation Formula:",
        " [ x ]   [ M00  M01  M02  M03 ]   [ i ]",
        " [ y ] = [ M10  M11  M12  M13 ] * [ j ]",
        " [ z ]   [ M20  M21  M22  M23 ]   [ k ]",
        " [ 1 ]   [   0    0    0    1 ]   [ 1 ]",
        "",
        "Affine Transformation Matrix M:",
    ]
    for row in affine:
        text_lines.append(f"  [{row[0]:8.2f}  {row[1]:8.2f}  {row[2]:8.2f}  {row[3]:9.2f} ]")

    text_lines.extend([
        "",
        "Landmark Coordinate Transformation Map:",
        " Landmark                  Voxel (i, j, k)       World Space (x, y, z) mm",
        " " + "─" * 65
    ])
    for name, v_idx in landmarks:
        w_coord = (np.array([v_idx[0], v_idx[1], v_idx[2], 1.0]) @ affine.T)[:3]
        v_str = f"({v_idx[0]:3d}, {v_idx[1]:3d}, {v_idx[2]:3d})"
        w_str = f"({w_coord[0]:7.2f}, {w_coord[1]:7.2f}, {w_coord[2]:7.2f}) mm"
        text_lines.append(f" {name:<24}  {v_str:<18}  {w_str}")

    formatted_text = "\n".join(text_lines)
    ax_txt.text(
        0.02, 0.98, formatted_text,
        transform=ax_txt.transAxes,
        fontfamily='monospace',
        fontsize=9.5,
        color='#e2e8f0',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#161b26', edgecolor='#2d3748', alpha=0.95)
    )

    fig.suptitle(title, fontsize=13, fontweight="bold", color="white", y=0.99)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=180, facecolor=fig.get_facecolor())

    return fig


def plot_3d_tumor_mesh_static(
    mask: np.ndarray,
    affine: Optional[np.ndarray] = None,
    step_size: int = 2,
    title: str = "Publication 3D Brain Tumor Mesh Surface Reconstruction",
    figsize: Tuple[int, int] = (18, 5),
    save_path: Optional[Union[str, Path]] = None
) -> Optional[plt.Figure]:
    """
    Publication-grade 4-view static 3D tumor surface mesh reconstruction using Marching Cubes
    and Matplotlib 3D Poly3DCollection (no WebGL or interactive widgets required).
    Displays 4 canonical medical views simultaneously:
    1. Isometric Perspective (elev=25, azim=45)
    2. Superior / Axial View (elev=90, azim=-90)
    3. Anterior / Coronal View (elev=0, azim=-90)
    4. Lateral / Sagittal View (elev=0, azim=0)
    """
    if not HAS_SKIMAGE:
        print("[!] scikit-image is required for Marching Cubes surface extraction.")
        return None

    subregions = [
        {"val": 2, "name": "Peritumoral Edema (ED)", "facecolor": "#1ebf40", "edgecolor": "#15802c", "alpha": 0.22},
        {"val": 1, "name": "Necrotic Core (NCR)", "facecolor": "#e62626", "edgecolor": "#991414", "alpha": 0.85},
        {"val": 3, "name": "Enhancing Tumor (ET)", "facecolor": "#ffcc0d", "edgecolor": "#b88f00", "alpha": 0.85},
    ]

    # Pre-extract meshes
    meshes = []
    all_verts = []
    for sub in subregions:
        binary_vol = (mask == sub["val"]).astype(np.float32)
        if np.sum(binary_vol) < 20:
            continue
        try:
            verts, faces, _, _ = marching_cubes(binary_vol, level=0.5, step_size=step_size)
            if affine is not None:
                homo_verts = np.hstack([verts, np.ones((verts.shape[0], 1))])
                world_verts = (homo_verts @ affine.T)[:, :3]
                verts = world_verts
            meshes.append({
                "verts": verts,
                "faces": faces,
                "cfg": sub
            })
            all_verts.append(verts)
        except Exception as e:
            continue

    if not meshes:
        print("[!] No foreground tumor voxels found to extract 3D mesh.")
        return None

    all_verts_cat = np.vstack(all_verts)
    min_b = all_verts_cat.min(axis=0)
    max_b = all_verts_cat.max(axis=0)
    center_b = (min_b + max_b) / 2.0
    max_range = max(max_b - min_b) / 2.0

    camera_views = [
        ("Isometric 3D View", 25, 45),
        ("Superior (Axial) View", 89, -90),
        ("Anterior (Coronal) View", 0, -90),
        ("Lateral (Sagittal) View", 0, 0)
    ]

    fig = plt.figure(figsize=figsize, facecolor="#0e1117")

    unit = "mm" if affine is not None else "voxels"

    for i, (v_name, elev, azim) in enumerate(camera_views, 1):
        ax = fig.add_subplot(1, 4, i, projection='3d', facecolor="#0e1117")

        for m in meshes:
            verts = m["verts"]
            faces = m["faces"]
            cfg = m["cfg"]
            poly = verts[faces]
            mesh_collection = Poly3DCollection(
                poly,
                facecolors=cfg["facecolor"],
                edgecolors=cfg["edgecolor"],
                linewidths=0.2,
                alpha=cfg["alpha"]
            )
            ax.add_collection3d(mesh_collection)

        ax.set_xlim(center_b[0] - max_range, center_b[0] + max_range)
        ax.set_ylim(center_b[1] - max_range, center_b[1] + max_range)
        ax.set_zlim(center_b[2] - max_range, center_b[2] + max_range)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(v_name, fontsize=11, fontweight="bold", color="white", pad=8)
        ax.tick_params(colors='#a0aec0', labelsize=8)
        ax.grid(color='#2d3748', linestyle=':', linewidth=0.5)

        if i == 1:
            ax.set_xlabel(f"X ({unit})", color='#e2e8f0', fontsize=9, labelpad=4)
            ax.set_ylabel(f"Y ({unit})", color='#e2e8f0', fontsize=9, labelpad=4)
            ax.set_zlabel(f"Z ({unit})", color='#e2e8f0', fontsize=9, labelpad=4)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

    legend_elements = [
        Patch(facecolor=BRATS_HEX_COLORS[1], edgecolor='white', label='Necrotic Core (NCR)', alpha=0.85),
        Patch(facecolor=BRATS_HEX_COLORS[2], edgecolor='white', label='Peritumoral Edema (ED)', alpha=0.35),
        Patch(facecolor=BRATS_HEX_COLORS[3], edgecolor='white', label='Enhancing Tumor (ET)', alpha=0.85)
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=10,
        frameon=True,
        facecolor='#1b202c',
        edgecolor='#374151',
        labelcolor='white',
        bbox_to_anchor=(0.5, -0.05)
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", color="white", y=1.04)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=180, facecolor=fig.get_facecolor())

    return fig


def plot_triplanar_mip(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    channel: int = 0,
    title: str = "Tri-Planar Maximum Intensity Projection (MIP)",
    cmap_img: str = "gray",
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Generate static Maximum Intensity Projections (MIP) along Sagittal, Coronal, and Axial planes.
    A standard non-interactive 3D radiological technique to appreciate 3D depth and vascularity.
    """
    vol_3d = _extract_3d_volume(volume, channel)

    # 1. Sagittal MIP (project along X / axis 0)
    sag_mip = np.rot90(np.max(vol_3d, axis=0))
    # 2. Coronal MIP (project along Y / axis 1)
    cor_mip = np.rot90(np.max(vol_3d, axis=1))
    # 3. Axial MIP (project along Z / axis 2)
    ax_mip = np.rot90(np.max(vol_3d, axis=2))

    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="#0e1117")

    axes[0].imshow(sag_mip, cmap=cmap_img, aspect='equal')
    axes[0].set_title("Sagittal MIP (Project X)", fontsize=11, fontweight="bold", color="white")
    axes[0].axis('off')

    axes[1].imshow(cor_mip, cmap=cmap_img, aspect='equal')
    axes[1].set_title("Coronal MIP (Project Y)", fontsize=11, fontweight="bold", color="white")
    axes[1].axis('off')

    axes[2].imshow(ax_mip, cmap=cmap_img, aspect='equal')
    axes[2].set_title("Axial MIP (Project Z)", fontsize=11, fontweight="bold", color="white")
    axes[2].axis('off')

    if mask is not None:
        sag_mask_mip = np.rot90(np.max(mask, axis=0))
        cor_mask_mip = np.rot90(np.max(mask, axis=1))
        ax_mask_mip = np.rot90(np.max(mask, axis=2))

        axes[0].imshow(sag_mask_mip, cmap=BRATS_CMAP, vmin=0, vmax=3, alpha=0.5, interpolation='nearest')
        axes[1].imshow(cor_mask_mip, cmap=BRATS_CMAP, vmin=0, vmax=3, alpha=0.5, interpolation='nearest')
        axes[2].imshow(ax_mask_mip, cmap=BRATS_CMAP, vmin=0, vmax=3, alpha=0.5, interpolation='nearest')

        legend_elements = [
            Patch(facecolor=BRATS_HEX_COLORS[1], edgecolor='white', label='Necrotic Core (NCR)', alpha=0.85),
            Patch(facecolor=BRATS_HEX_COLORS[2], edgecolor='white', label='Peritumoral Edema (ED)', alpha=0.85),
            Patch(facecolor=BRATS_HEX_COLORS[3], edgecolor='white', label='Enhancing Tumor (ET)', alpha=0.85)
        ]
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=3,
            fontsize=10,
            frameon=True,
            facecolor='#1b202c',
            edgecolor='#374151',
            labelcolor='white',
            bbox_to_anchor=(0.5, -0.04)
        )

    fig.suptitle(title, fontsize=13, fontweight="bold", color="white", y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=180, facecolor=fig.get_facecolor())

    return fig


def compute_brats_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray, voxel_spacing: Optional[Tuple[float, float, float]] = None) -> Dict[str, Any]:
    """
    Compute official BraTS hierarchical sub-region metrics:
    1. Whole Tumor (WT): Labels 1 + 2 + 3
    2. Tumor Core (TC): Labels 1 + 3
    3. Enhancing Tumor (ET): Label 3
    Computes Dice coefficient, IoU (Jaccard), and estimated volume in cm^3.
    """
    def dice_coef(gt_bin, pred_bin):
        inter = np.logical_and(gt_bin, pred_bin).sum()
        total = gt_bin.sum() + pred_bin.sum()
        if total == 0:
            return 1.0
        return float((2.0 * inter) / (total + 1e-8))

    def iou_coef(gt_bin, pred_bin):
        inter = np.logical_and(gt_bin, pred_bin).sum()
        union = np.logical_or(gt_bin, pred_bin).sum()
        if union == 0:
            return 1.0
        return float(inter / (union + 1e-8))

    # Binary masks for BraTS sub-regions
    gt_wt = (gt_mask >= 1)
    pred_wt = (pred_mask >= 1)

    gt_tc = np.isin(gt_mask, [1, 3])
    pred_tc = np.isin(pred_mask, [1, 3])

    gt_et = (gt_mask == 3)
    pred_et = (pred_mask == 3)

    # Voxel volume in cm^3
    if voxel_spacing is not None:
        vox_vol_cm3 = (voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]) / 1000.0
    else:
        vox_vol_cm3 = 1.0 / 1000.0  # 1 mm^3 default

    metrics = {
        "WT_Dice": dice_coef(gt_wt, pred_wt),
        "WT_IoU": iou_coef(gt_wt, pred_wt),
        "WT_Vol_GT_cm3": float(gt_wt.sum() * vox_vol_cm3),
        "WT_Vol_Pred_cm3": float(pred_wt.sum() * vox_vol_cm3),

        "TC_Dice": dice_coef(gt_tc, pred_tc),
        "TC_IoU": iou_coef(gt_tc, pred_tc),
        "TC_Vol_GT_cm3": float(gt_tc.sum() * vox_vol_cm3),
        "TC_Vol_Pred_cm3": float(pred_tc.sum() * vox_vol_cm3),

        "ET_Dice": dice_coef(gt_et, pred_et),
        "ET_IoU": iou_coef(gt_et, pred_et),
        "ET_Vol_GT_cm3": float(gt_et.sum() * vox_vol_cm3),
        "ET_Vol_Pred_cm3": float(pred_et.sum() * vox_vol_cm3),
    }
    metrics["Mean_Dice"] = float((metrics["WT_Dice"] + metrics["TC_Dice"] + metrics["ET_Dice"]) / 3.0)
    return metrics


def plot_segmentation_evaluation_dashboard(
    image_4d: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    modality_idx: int = 2,
    slices_z: Optional[List[int]] = None,
    num_slices: int = 4,
    metrics: Optional[Dict[str, Any]] = None,
    title: str = "3D U-Net Volumetric Segmentation & Spatial Error Matrix",
    figsize: Tuple[int, int] = (19, 14),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Publication-grade multi-slice segmentation evaluation matrix.
    Rows: Selected axial cross-sections through the tumor (inferior to superior).
    Columns:
      1. Structural MRI Scan (T1ce)
      2. Ground Truth Expert Annotation
      3. 3D U-Net Model Prediction
      4. Spatial Error / Confusion Map:
         - Green: True Positive (Hit)
         - Red: False Positive (Over-segmentation)
         - Blue: False Negative (Missed Tumor)
    Annotates slice-level Dice scores and includes overall BraTS benchmark scorecard.
    """
    vol_3d = _extract_3d_volume(image_4d, modality_idx)
    nz = vol_3d.shape[2]

    if slices_z is None:
        tumor_per_z = np.array([np.sum(gt_mask[:, :, z] > 0) for z in range(nz)])
        if np.any(tumor_per_z > 0):
            z_indices_with_tumor = np.where(tumor_per_z > 0)[0]
            slices_z = np.linspace(z_indices_with_tumor[0], z_indices_with_tumor[-1], num_slices, dtype=int).tolist()
        else:
            slices_z = np.linspace(int(0.3 * nz), int(0.7 * nz), num_slices, dtype=int).tolist()

    num_rows = len(slices_z)
    fig, axes = plt.subplots(num_rows, 4, figsize=figsize, facecolor="#0e1117")
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    col_headers = [
        "1. MRI Structural (T1ce)",
        "2. Ground Truth Annotation",
        "3. 3D U-Net Prediction",
        "4. Spatial Error Map (TP/FP/FN)"
    ]

    for row_idx, z in enumerate(slices_z):
        img_slice = np.rot90(vol_3d[:, :, z])
        gt_slice = np.rot90(gt_mask[:, :, z])
        pred_slice = np.rot90(pred_mask[:, :, z])

        # Slice-level Dice
        gt_fg = (gt_slice > 0)
        pred_fg = (pred_slice > 0)
        tp = np.logical_and(gt_fg, pred_fg)
        fp = np.logical_and(np.logical_not(gt_fg), pred_fg)
        fn = np.logical_and(gt_fg, np.logical_not(pred_fg))

        dice_slice = (2.0 * tp.sum()) / (gt_fg.sum() + pred_fg.sum() + 1e-8) if (gt_fg.sum() + pred_fg.sum()) > 0 else 1.0

        # Col 1: MRI
        axes[row_idx, 0].imshow(img_slice, cmap='gray')
        axes[row_idx, 0].axis('off')
        if row_idx == 0:
            axes[row_idx, 0].set_title(col_headers[0], fontsize=11, fontweight="bold", color="white", pad=8)
        axes[row_idx, 0].text(0.03, 0.92, f"Z = {z}", transform=axes[row_idx, 0].transAxes, color="#00ffff", fontsize=10, fontweight="bold", bbox=dict(facecolor="#161b26", alpha=0.8, edgecolor="none"))

        # Col 2: Ground Truth
        axes[row_idx, 1].imshow(img_slice, cmap='gray')
        axes[row_idx, 1].imshow(gt_slice, cmap=BRATS_CMAP, vmin=0, vmax=3, alpha=0.75, interpolation='nearest')
        axes[row_idx, 1].axis('off')
        if row_idx == 0:
            axes[row_idx, 1].set_title(col_headers[1], fontsize=11, fontweight="bold", color="#1ebf40", pad=8)

        # Col 3: Prediction
        axes[row_idx, 2].imshow(img_slice, cmap='gray')
        axes[row_idx, 2].imshow(pred_slice, cmap=BRATS_CMAP, vmin=0, vmax=3, alpha=0.75, interpolation='nearest')
        axes[row_idx, 2].axis('off')
        if row_idx == 0:
            axes[row_idx, 2].set_title(col_headers[2], fontsize=11, fontweight="bold", color="#ffcc0d", pad=8)

        # Col 4: Error Map
        # Create RGB overlay: TP = Green, FP = Red, FN = Cyan/Blue
        h, w = img_slice.shape
        error_rgb = np.zeros((h, w, 3), dtype=np.float32)
        error_rgb[tp] = [0.12, 0.85, 0.25]  # Green: TP
        error_rgb[fp] = [0.95, 0.15, 0.15]  # Red: FP (Over-segmentation)
        error_rgb[fn] = [0.10, 0.55, 0.95]  # Blue: FN (Under-segmentation)

        error_alpha = np.zeros((h, w), dtype=np.float32)
        error_alpha[tp | fp | fn] = 0.85

        axes[row_idx, 3].imshow(img_slice, cmap='gray')
        axes[row_idx, 3].imshow(error_rgb, alpha=error_alpha, interpolation='nearest')
        axes[row_idx, 3].axis('off')
        dice_color = "#1ebf40" if dice_slice > 0.8 else ("#ffcc0d" if dice_slice > 0.5 else "#e62626")
        axes[row_idx, 3].set_title(f"Slice Dice: {dice_slice:.3f}", fontsize=10, fontweight="bold", color=dice_color, pad=4)
        if row_idx == 0:
            axes[row_idx, 3].set_title(f"{col_headers[3]}\nSlice Dice: {dice_slice:.3f}", fontsize=11, fontweight="bold", color="white", pad=8)

    # Scorecard legend at bottom
    legend_elements = [
        Patch(facecolor='#1ebf40', edgecolor='white', label='True Positive (Hit)'),
        Patch(facecolor='#e62626', edgecolor='white', label='False Positive (Over-seg)'),
        Patch(facecolor='#1a8cff', edgecolor='white', label='False Negative (Missed)'),
        Patch(facecolor=BRATS_HEX_COLORS[1], edgecolor='white', label='NCR Core', alpha=0.85),
        Patch(facecolor=BRATS_HEX_COLORS[2], edgecolor='white', label='ED Edema', alpha=0.85),
        Patch(facecolor=BRATS_HEX_COLORS[3], edgecolor='white', label='ET Enhancing', alpha=0.85)
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=6,
        fontsize=9.5,
        frameon=True,
        facecolor='#1b202c',
        edgecolor='#374151',
        labelcolor='white',
        bbox_to_anchor=(0.5, -0.02)
    )

    metrics_text = ""
    if metrics is not None:
        metrics_text = f" | WT Dice: {metrics.get('WT_Dice', 0):.3f} | TC Dice: {metrics.get('TC_Dice', 0):.3f} | ET Dice: {metrics.get('ET_Dice', 0):.3f} | Mean Dice: {metrics.get('Mean_Dice', 0):.3f}"

    fig.suptitle(f"{title}{metrics_text}", fontsize=13, fontweight="bold", color="white", y=1.01)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=180, facecolor=fig.get_facecolor())

    return fig


def plot_3d_segmentation_comparison_static(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    affine: Optional[np.ndarray] = None,
    step_size: int = 2,
    title: str = "3D Isosurface Comparison: Expert Ground Truth vs. 3D U-Net Prediction",
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[Union[str, Path]] = None
) -> Optional[plt.Figure]:
    """
    Direct side-by-side static 3D mesh rendering comparing:
    - Left: Ground Truth 3D Tumor Mesh
    - Right: 3D U-Net Predicted 3D Tumor Mesh
    Both viewed at identical angles and scale for direct morphological comparison.
    """
    if not HAS_SKIMAGE:
        print("[!] scikit-image is required for Marching Cubes.")
        return None

    subregions = [
        {"val": 2, "name": "Edema (ED)", "facecolor": "#1ebf40", "edgecolor": "#15802c", "alpha": 0.22},
        {"val": 1, "name": "Necrotic Core (NCR)", "facecolor": "#e62626", "edgecolor": "#991414", "alpha": 0.85},
        {"val": 3, "name": "Enhancing Tumor (ET)", "facecolor": "#ffcc0d", "edgecolor": "#b88f00", "alpha": 0.85},
    ]

    def extract_surfaces(mask_vol):
        meshes = []
        all_verts = []
        for sub in subregions:
            b_vol = (mask_vol == sub["val"]).astype(np.float32)
            if np.sum(b_vol) < 20:
                continue
            try:
                verts, faces, _, _ = marching_cubes(b_vol, level=0.5, step_size=step_size)
                if affine is not None:
                    homo = np.hstack([verts, np.ones((verts.shape[0], 1))])
                    verts = (homo @ affine.T)[:, :3]
                meshes.append({"verts": verts, "faces": faces, "cfg": sub})
                all_verts.append(verts)
            except Exception:
                continue
        return meshes, all_verts

    gt_meshes, gt_all = extract_surfaces(gt_mask)
    pred_meshes, pred_all = extract_surfaces(pred_mask)

    all_verts_combined = gt_all + pred_all
    if not all_verts_combined:
        print("[!] No foreground voxels found for 3D comparison.")
        return None

    cat_all = np.vstack(all_verts_combined)
    min_b = cat_all.min(axis=0)
    max_b = cat_all.max(axis=0)
    center_b = (min_b + max_b) / 2.0
    max_range = max(max_b - min_b) / 2.0

    fig = plt.figure(figsize=figsize, facecolor="#0e1117")

    panels = [
        ("Expert Ground Truth (GT) 3D Model", gt_meshes, 1),
        ("3D U-Net Model Prediction 3D Model", pred_meshes, 2)
    ]
    unit = "mm" if affine is not None else "voxels"

    for p_title, meshes, col_idx in panels:
        ax = fig.add_subplot(1, 2, col_idx, projection='3d', facecolor="#0e1117")
        for m in meshes:
            verts = m["verts"]
            faces = m["faces"]
            cfg = m["cfg"]
            poly = verts[faces]
            mesh_coll = Poly3DCollection(
                poly,
                facecolors=cfg["facecolor"],
                edgecolors=cfg["edgecolor"],
                linewidths=0.2,
                alpha=cfg["alpha"]
            )
            ax.add_collection3d(mesh_coll)

        ax.set_xlim(center_b[0] - max_range, center_b[0] + max_range)
        ax.set_ylim(center_b[1] - max_range, center_b[1] + max_range)
        ax.set_zlim(center_b[2] - max_range, center_b[2] + max_range)

        ax.view_init(elev=25, azim=45)
        ax.set_title(p_title, fontsize=12, fontweight="bold", color="white", pad=8)
        ax.set_xlabel(f"X ({unit})", color='#e2e8f0', fontsize=9, labelpad=4)
        ax.set_ylabel(f"Y ({unit})", color='#e2e8f0', fontsize=9, labelpad=4)
        ax.set_zlabel(f"Z ({unit})", color='#e2e8f0', fontsize=9, labelpad=4)
        ax.tick_params(colors='#a0aec0', labelsize=8)
        ax.grid(color='#2d3748', linestyle=':', linewidth=0.5)

    legend_elements = [
        Patch(facecolor=BRATS_HEX_COLORS[1], edgecolor='white', label='Necrotic Core (NCR)', alpha=0.85),
        Patch(facecolor=BRATS_HEX_COLORS[2], edgecolor='white', label='Peritumoral Edema (ED)', alpha=0.35),
        Patch(facecolor=BRATS_HEX_COLORS[3], edgecolor='white', label='Enhancing Tumor (ET)', alpha=0.85)
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=10,
        frameon=True,
        facecolor='#1b202c',
        edgecolor='#374151',
        labelcolor='white',
        bbox_to_anchor=(0.5, -0.04)
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", color="white", y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=180, facecolor=fig.get_facecolor())

    return fig


# =========================================================================
# Backward Compatibility Wrappers for Interactive Functions
# (Kept so legacy code calling them does not fail, but delegates or runs)
# =========================================================================

def interactive_orthogonal_viewer(volume: np.ndarray, mask: Optional[np.ndarray] = None, modality_names: Optional[List[str]] = None):
    """Interactive orthogonal viewer using ipywidgets (legacy wrapper)."""
    if not HAS_WIDGETS:
        return plot_orthogonal_slices(volume, mask=mask)
    # Forward to plot_orthogonal_slices directly if running non-interactively
    return plot_orthogonal_slices(volume, mask=mask)


def interactive_affine_calculator(affine: np.ndarray, shape: Tuple[int, int, int]):
    """Interactive affine calculator (legacy wrapper)."""
    return plot_affine_coordinate_space(affine, shape)


def plot_3d_tumor_mesh_plotly(mask: np.ndarray, affine: Optional[np.ndarray] = None, step_size: int = 2, title: str = "3D Brain Tumor Mesh"):
    """Plotly 3D mesh viewer (legacy wrapper)."""
    if not (HAS_PLOTLY and HAS_SKIMAGE):
        return None
    fig = go.Figure()
    subregions = [
        {"val": 2, "name": "Peritumoral Edema (ED)", "color": "limegreen", "opacity": 0.3},
        {"val": 1, "name": "Necrotic Core (NCR)", "color": "crimson", "opacity": 0.8},
        {"val": 3, "name": "Enhancing Tumor (ET)", "color": "gold", "opacity": 0.9},
    ]
    has_surfaces = False
    for sub in subregions:
        binary_vol = (mask == sub["val"]).astype(np.float32)
        if np.sum(binary_vol) < 20:
            continue
        verts, faces, _, _ = marching_cubes(binary_vol, level=0.5, step_size=step_size)
        if affine is not None:
            homo_verts = np.hstack([verts, np.ones((verts.shape[0], 1))])
            world_verts = homo_verts @ affine.T
            verts = world_verts[:, :3]
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=sub["color"], opacity=sub["opacity"], name=sub["name"]))
        has_surfaces = True
    if not has_surfaces:
        return None
    fig.update_layout(title=title, scene=dict(aspectmode='data'))
    return fig


def interactive_segmentation_comparison(image_4d: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, modality_idx: int = 2):
    """Interactive segmentation dashboard (legacy wrapper)."""
    return plot_segmentation_evaluation_dashboard(image_4d, gt_mask, pred_mask, modality_idx=modality_idx)
