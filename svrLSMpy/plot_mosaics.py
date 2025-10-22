import numpy as np
import nibabel as nib
from nilearn.image import load_img
from nilearn.plotting import plot_img

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

bg_img_path = files("svrLSMpy.resources").joinpath("mni152.nii.gz")
bg_img = str(bg_img_path)

# Axis configuration for different slice orientations
AXIS_CONFIG = {
    'axial': {'dim': 2, 'axis_tuple': (0, 1), 'voxel_template': [0, 0, None], 'display_mode': 'z'},
    'coronal': {'dim': 1, 'axis_tuple': (0, 2), 'voxel_template': [0, None, 0], 'display_mode': 'y'},
    'sagittal': {'dim': 0, 'axis_tuple': (1, 2), 'voxel_template': [None, 0, 0], 'display_mode': 'x'}
}


def get_slice_coordinates(nii_file_path, num_slices, orientation='axial', adjustment=3.0):
    """
    Calculate slice cut coordinates (in world space) for a NIfTI file.

    :param nii_file_path: Path to the NIfTI (.nii) file.
    :param num_slices: Number of slices to generate.
    :param orientation: Slice orientation ('axial', 'coronal', or 'sagittal').
    :param adjustment: Amount to adjust min/max bounds (default: 3.0).
    :return: A list of cut coordinates in world space.
    """
    if orientation not in AXIS_CONFIG:
        raise ValueError(f"Orientation must be one of {list(AXIS_CONFIG.keys())}")
    
    config = AXIS_CONFIG[orientation]
    img = load_img(nii_file_path)
    affine = img.affine
    data = img.get_fdata()
    
    # Find non-empty slices along the specified axis
    non_empty_slices = np.any(data > 0, axis=config['axis_tuple'])
    non_empty_indices = np.where(non_empty_slices)[0]
    
    if len(non_empty_indices) == 0:
        raise ValueError("The NIfTI image has no non-empty slices.")
    
    min_index, max_index = non_empty_indices[0], non_empty_indices[-1]
    
    # Convert voxel indices to world coordinates
    voxel_min = config['voxel_template'].copy()
    voxel_max = config['voxel_template'].copy()
    voxel_min[config['dim']] = min_index
    voxel_max[config['dim']] = max_index
    
    world_coords = nib.affines.apply_affine(affine, [voxel_min, voxel_max])
    coord_min, coord_max = world_coords[:, config['dim']]
    
    # Adjust bounds
    coord_min_adjusted = coord_min - adjustment
    coord_max_adjusted = coord_max + adjustment
    
    # Generate evenly spaced cut coordinates
    cut_coords = np.linspace(coord_min_adjusted, coord_max_adjusted, num_slices).tolist()
    
    print(f"{orientation.capitalize()} cut slice coordinates:", cut_coords)
    
    return cut_coords


def save_slice_mosaic(nii_file_path, cut_coords, output_image_path, 
                      orientation='axial', max_activation=None, 
                      cmap='jet', threshold=0, black_bg=False, colorbar=True):
    """
    Save a mosaic of brain slices to an image file.

    :param nii_file_path: Path to the NIfTI (.nii) file.
    :param cut_coords: List of cut coordinates in world space.
    :param output_image_path: Path to save the output image.
    :param orientation: Slice orientation ('axial', 'coronal', or 'sagittal').
    :param max_activation: Maximum activation value for color scale (symmetric around 0).
    :param cmap: Colormap to use (default: 'jet').
    :param threshold: Threshold for displaying values (default: 0).
    :param black_bg: Use black background (default: False).
    :param colorbar: Show colorbar (default: True).
    """
    if orientation not in AXIS_CONFIG:
        raise ValueError(f"Orientation must be one of {list(AXIS_CONFIG.keys())}")
    
    img = load_img(nii_file_path)
    display_mode = AXIS_CONFIG[orientation]['display_mode']
    
    vmin = -max_activation if max_activation is not None else None
    vmax = max_activation if max_activation is not None else None
    
    plot_img(img, cut_coords, output_image_path, 
             display_mode=display_mode, threshold=threshold, 
             bg_img=bg_img, vmin=vmin, vmax=vmax, 
             black_bg=black_bg, colorbar=colorbar, cmap=cmap)


# Convenience functions for backward compatibility
def get_axial_slices(nii_file_path, num_slices):
    return get_slice_coordinates(nii_file_path, num_slices, 'axial')

def get_coronal_slices(nii_file_path, num_slices):
    return get_slice_coordinates(nii_file_path, num_slices, 'coronal')

def get_sagittal_slices(nii_file_path, num_slices):
    return get_slice_coordinates(nii_file_path, num_slices, 'sagittal')

def save_axial_mosaic(nii_file_path, cut_coords, output_image_path, max_activation=None):
    save_slice_mosaic(nii_file_path, cut_coords, output_image_path, 'axial', max_activation)

def save_coronal_mosaic(nii_file_path, cut_coords, output_image_path, max_activation=None):
    save_slice_mosaic(nii_file_path, cut_coords, output_image_path, 'coronal', max_activation)

def save_sagittal_mosaic(nii_file_path, cut_coords, output_image_path, max_activation=None):
    save_slice_mosaic(nii_file_path, cut_coords, output_image_path, 'sagittal', max_activation)
