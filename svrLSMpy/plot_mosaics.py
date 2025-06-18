from nilearn.image import load_img
import numpy as np
import nibabel as nib
from nilearn.plotting import plot_img


def get_axial_slices(nii_file_path, num_slices):
    """
    Calculate axial cut coordinates (in world space) for a NIfTI file using Nilearn.

    :param nii_file_path: Path to the NIfTI (.nii) file.
    :param num_slices: Number of slices to generate.
    :return: A sorted list of axial cut coordinates in world space.
    """
    # Load the NIfTI image
    img = load_img(nii_file_path)
    affine = img.affine  # Get the affine matrix
    data = img.get_fdata()

    # Determine the range of the axial dimension in world space
    axial_dim = 2  # Axial slices are along the 3rd dimension (z-axis)
    non_empty_slices = np.any(data > 0, axis=(0, 1))  # Find non-empty slices along the z-axis
    non_empty_indices = np.where(non_empty_slices)[0]

    if len(non_empty_indices) == 0:
        raise ValueError("The NIfTI image has no non-empty slices.")

    z_min_index, z_max_index = non_empty_indices[0], non_empty_indices[-1]

    # Convert voxel indices to world coordinates
    voxel_indices = [[0, 0, z_min_index], [0, 0, z_max_index]]
    world_coords = nib.affines.apply_affine(affine, voxel_indices)
    z_min, z_max = world_coords[:, axial_dim]  # Extract the range in world space

    # Adjust z_min and z_max slightly, ensuring they are within valid range
    adjustment = 3.0  # Adjust by 3 units (or any small value)
    z_min_adjusted = z_min - adjustment
    z_max_adjusted = z_max + adjustment

    # Generate evenly spaced cut coordinates in world space
    cut_coords = np.linspace(z_min_adjusted, z_max_adjusted, num_slices)

    cut_coords = cut_coords.tolist()

    print("Axial cut slice coordinates:", cut_coords)

    return cut_coords

def save_axial_mosaic(nii_file_path, cut_coords, output_image_path):
    img = load_img(nii_file_path)
    plot_img(img, cut_coords, output_image_path, display_mode="z", threshold=0, bg_img="resources/mni152.nii.gz", black_bg=False,colorbar="True", cmap="jet")

import numpy as np
import nibabel as nib
from nilearn import plotting
from nilearn.image import load_img

def get_coronal_slices(nii_file_path, num_slices):
    """
    Calculate coronal cut coordinates (in world space) for a NIfTI file using Nilearn.

    :param nii_file_path: Path to the NIfTI (.nii) file.
    :param num_slices: Number of slices to generate.
    :return: A sorted list of coronal cut coordinates in world space.
    """
    # Load the NIfTI image
    img = load_img(nii_file_path)
    affine = img.affine  # Get the affine matrix
    data = img.get_fdata()

    # Coronal slices are along the 2nd dimension (y-axis)
    coronal_dim = 1
    non_empty_slices = np.any(data > 0, axis=(0, 2))  # Find non-empty slices along the y-axis
    non_empty_indices = np.where(non_empty_slices)[0]

    if len(non_empty_indices) == 0:
        raise ValueError("The NIfTI image has no non-empty slices.")

    y_min_index, y_max_index = non_empty_indices[0], non_empty_indices[-1]

    # Convert voxel indices to world coordinates
    voxel_indices = [[0, y_min_index, 0], [0, y_max_index, 0]]
    world_coords = nib.affines.apply_affine(affine, voxel_indices)
    y_min, y_max = world_coords[:, coronal_dim]  # Extract the range in world space

    # Adjust y_min and y_max slightly, ensuring they are within valid range
    adjustment = 3.0  # Adjust by 3 units (or any small value)
    y_min_adjusted = y_min - adjustment
    y_max_adjusted = y_max + adjustment

    # Generate evenly spaced cut coordinates in world space
    cut_coords = np.linspace(y_min_adjusted, y_max_adjusted, num_slices)

    cut_coords = cut_coords.tolist()

    print("Coronal cut slice coordinates:", cut_coords)

    return cut_coords

def save_coronal_mosaic(nii_file_path, cut_coords, output_image_path):
    """
    Save a mosaic of coronal slices from the NIfTI file.

    :param nii_file_path: Path to the NIfTI (.nii) file.
    :param cut_coords: List of coronal slice coordinates in world space.
    :param output_image_path: Path to save the output image.
    """
    img = load_img(nii_file_path)
    plotting.plot_img(img, cut_coords, output_image_path, display_mode="y", threshold=0, bg_img="resources/mni152.nii.gz", black_bg=False, colorbar=True, cmap="jet")


def get_sagittal_slices(nii_file_path, num_slices):
    """
    Calculate sagittal cut coordinates (in world space) for a NIfTI file using Nilearn.

    :param nii_file_path: Path to the NIfTI (.nii) file.
    :param num_slices: Number of slices to generate.
    :return: A sorted list of sagittal cut coordinates in world space.
    """
    # Load the NIfTI image
    img = load_img(nii_file_path)
    affine = img.affine  # Get the affine matrix
    data = img.get_fdata()

    # Sagittal slices are along the 1st dimension (x-axis)
    sagittal_dim = 0
    non_empty_slices = np.any(data > 0, axis=(1, 2))  # Find non-empty slices along the x-axis
    non_empty_indices = np.where(non_empty_slices)[0]

    if len(non_empty_indices) == 0:
        raise ValueError("The NIfTI image has no non-empty slices.")

    x_min_index, x_max_index = non_empty_indices[0], non_empty_indices[-1]

    # Convert voxel indices to world coordinates
    voxel_indices = [[x_min_index, 0, 0], [x_max_index, 0, 0]]
    world_coords = nib.affines.apply_affine(affine, voxel_indices)
    x_min, x_max = world_coords[:, sagittal_dim]  # Extract the range in world space

    # Adjust x_min and x_max slightly, ensuring they are within valid range
    adjustment = 3.0  # Adjust by 3 units (or any small value)
    x_min_adjusted = x_min - adjustment
    x_max_adjusted = x_max + adjustment

    # Generate evenly spaced cut coordinates in world space
    cut_coords = np.linspace(x_min_adjusted, x_max_adjusted, num_slices)

    cut_coords = cut_coords.tolist()

    print("Sagittal cut slice coordinates:", cut_coords)

    return cut_coords

def save_sagittal_mosaic(nii_file_path, cut_coords, output_image_path):
    """
    Save a mosaic of sagittal slices from the NIfTI file.

    :param nii_file_path: Path to the NIfTI (.nii) file.
    :param cut_coords: List of sagittal slice coordinates in world space.
    :param output_image_path: Path to save the output image.
    """
    img = load_img(nii_file_path)
    plotting.plot_img(img, cut_coords, output_image_path, display_mode="x", threshold=0, bg_img="resources/mni152.nii.gz", black_bg=False, colorbar=True, cmap="jet")


'''
# Example usage
nii_file_path = "lesion_overlap_filtered.nii.gz"
output_image_path = "lesion_overlap_filtered.png"
num_slices = 10

cut_coords = get_axial_slices(nii_file_path, num_slices)
save_axial_mosaic(nii_file_path,cut_coords,output_image_path)
'''
