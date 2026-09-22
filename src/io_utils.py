"""
I/O utilities for handling NIfTI medical volumetric images and affine transformations.
"""

from pathlib import Path
from typing import Dict, Tuple, Any, Optional, Union
import numpy as np
import nibabel as nib


def load_nifti(file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """
    Load a NIfTI file (.nii or .nii.gz) using Nibabel.
    
    Args:
        file_path: Path to the NIfTI file.
        
    Returns:
        tuple: (data_array, affine_4x4_matrix, header)
    """
    img_obj = nib.load(str(file_path))
    # get_fdata() loads volume as float64 array (or can be cast as needed)
    data = img_obj.get_fdata()
    affine = img_obj.affine
    header = img_obj.header
    return data, affine, header


def inspect_nifti_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract key spatial, structural, and anatomical metadata from a NIfTI file.
    
    Args:
        file_path: Path to the NIfTI file.
        
    Returns:
        dict: Summary of spatial parameters (shape, voxel sizes, orientation, origin, etc.)
    """
    img = nib.load(str(file_path))
    header = img.header
    affine = img.affine
    
    # Voxel sizes (spacing / pixel dimensions in mm)
    zooms = header.get_zooms()
    
    # Anatomical orientation codes (e.g. ('R', 'A', 'S') or ('L', 'P', 'S'))
    orientation = nib.aff2axcodes(affine)
    
    # Coordinate origin (world coordinates corresponding to voxel (0, 0, 0))
    origin = affine[:3, 3]
    
    # Rotation & scaling matrix (top-left 3x3)
    rot_zoom = affine[:3, :3]
    
    metadata = {
        "file_name": Path(file_path).name,
        "shape": img.shape,
        "ndim": img.ndim,
        "data_dtype": str(img.get_data_dtype()),
        "voxel_spacing_mm": zooms,
        "orientation_codes": orientation,
        "coordinate_origin_mm": origin.tolist(),
        "affine_matrix": affine.tolist(),
        "qform_code": int(header['qform_code']),
        "sform_code": int(header['sform_code']),
    }
    return metadata


def voxel_to_world(voxel_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Transform voxel indices (i, j, k) to physical scanner/world coordinates (x, y, z) in mm.
    
    Args:
        voxel_coords: Array of shape (N, 3) or (3,) with (i, j, k) indices.
        affine: 4x4 affine matrix.
        
    Returns:
        np.ndarray: Physical coordinates in mm.
    """
    voxel_coords = np.asarray(voxel_coords)
    is_single = (voxel_coords.ndim == 1)
    if is_single:
        voxel_coords = voxel_coords[np.newaxis, :]
        
    # Append homogeneous coordinate 1
    homo = np.hstack([voxel_coords, np.ones((voxel_coords.shape[0], 1))])
    world = homo @ affine.T
    world_coords = world[:, :3]
    
    return world_coords[0] if is_single else world_coords


def world_to_voxel(world_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Transform physical scanner/world coordinates (x, y, z) in mm to nearest voxel indices (i, j, k).
    
    Args:
        world_coords: Array of shape (N, 3) or (3,) with physical coords in mm.
        affine: 4x4 affine matrix.
        
    Returns:
        np.ndarray: Voxel indices (integer).
    """
    world_coords = np.asarray(world_coords)
    is_single = (world_coords.ndim == 1)
    if is_single:
        world_coords = world_coords[np.newaxis, :]
        
    inv_affine = np.linalg.inv(affine)
    homo = np.hstack([world_coords, np.ones((world_coords.shape[0], 1))])
    voxels = homo @ inv_affine.T
    voxel_coords = np.round(voxels[:, :3]).astype(int)
    
    return voxel_coords[0] if is_single else voxel_coords


def print_metadata_summary(metadata: Dict[str, Any]) -> None:
    """Pretty-print metadata dictionary."""
    print("=" * 60)
    print(f" NIfTI Volume Metadata: {metadata['file_name']}")
    print("=" * 60)
    print(f" - Dimensions / Shape    : {metadata['shape']}")
    print(f" - Data Type            : {metadata['data_dtype']}")
    print(f" - Voxel Spacing (dx,dy,dz): {[round(z, 3) for z in metadata['voxel_spacing_mm']]} mm")
    print(f" - Anatomical Axes      : {''.join(metadata['orientation_codes'])} ({metadata['orientation_codes']})")
    print(f" - World Origin (0,0,0) : {[round(o, 2) for o in metadata['coordinate_origin_mm']]} mm")
    print(" - Affine Matrix (4x4)  :")
    for row in metadata['affine_matrix']:
        print(f"   [{' '.join(f'{val:9.3f}' for val in row)}]")
    print("=" * 60)
