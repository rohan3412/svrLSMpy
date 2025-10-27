import pandas as pd
import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from .util import normalize_file_name

def load_lesions_and_behaviors(
    lesion_folder,
    csv_file,
    max_score,
    do_regress_out_lesion_volume,
    do_regress_out_covariates
):
    
    print("Loading behavioral data and lesion files...")
    df = pd.read_csv(csv_file)

    # Validate required columns
    required_columns = ['filename', 'behavior']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"CSV file must contain '{col}' column.")

    # Match lesion files to CSV filenames
    lesion_files = []
    matched_pairs = []

    for filename in df['filename']:
        norm_filename = normalize_file_name(filename)
        matches = [
            file for file in os.listdir(lesion_folder)
            if norm_filename in normalize_file_name(file)
        ]

        if not matches:
            raise FileNotFoundError(
                f"No lesion files found in '{lesion_folder}' containing '{filename}' "
                f"(normalized as '{norm_filename}')"
            )
        if len(matches) > 1:
            raise ValueError(
                f"CSV entry '{filename}' matches multiple lesion files in '{lesion_folder}':\n"
                f"{'\n'.join(matches)}"
            )

        lesion_file = os.path.join(lesion_folder, matches[0])
        if lesion_file in [pair[1] for pair in matched_pairs]:
            prev_entry = next(pair[0] for pair in matched_pairs if pair[1] == lesion_file)
            raise ValueError(
                f"Multiple CSV entries map to the same lesion file:\n"
                f" - '{prev_entry}' and '{filename}' both map to '{lesion_file}'"
            )

        matched_pairs.append((filename, lesion_file))
        lesion_files.append(lesion_file)

    # Process behaviors
    behaviors = df['behavior'].values
    print("\nBehavior before normalization:\n", behaviors)
    behaviors = behaviors / max_score
    print("Behavior after normalization:\n", behaviors)

    # Display temporary DataFrame for verification
    temp_df = pd.DataFrame({
        "Lesion": lesion_files,
        "Score": df['filename']
    })
    print("\nTemporary DataFrame (Lesion ↔ Filename from CSV):")
    print(temp_df.to_string(index=False))
    print(f"\nNumber of lesions: {len(temp_df['Lesion'])}")
    print(f"Number of scores: {len(temp_df['Score'])}\n")

    # Initialize outputs
    lesion_volumes = None
    covariates = None

    # Process covariates and lesion volumes only if do_regress_out_covariates is True
    if do_regress_out_covariates:
        # Load covariates from CSV
        covariates = df.drop(columns=["filename", "behavior"]).select_dtypes(include=["number"])
        has_covariates = covariates.shape[1] > 0

        # Handle covariates and lesion volumes
        if has_covariates:
            # Handle NaNs
            if covariates.isna().any().any():
                covariates = covariates.fillna(covariates.mean())
                print("Filled NaN values in covariates with column means.")
            covariates = np.array(covariates)  # Convert to NumPy array for consistency

        if do_regress_out_lesion_volume:
            # Compute lesion volumes
            lesion_volumes = []
            for file in tqdm(lesion_files, desc="Loading lesions and computing volumes"):
                lesion_img = nib.load(file)
                lesion_data = lesion_img.get_fdata()
                voxel_volume = np.prod(lesion_img.header.get_zooms())
                volume = np.sum(lesion_data > 0) * voxel_volume
                lesion_volumes.append(volume)

            lesion_volumes = np.array(lesion_volumes).reshape(-1, 1)
            print(
                f"Lesion volumes summary: min={np.nanmin(lesion_volumes):.3f}, "
                f"max={np.nanmax(lesion_volumes):.3f}, NaNs={np.isnan(lesion_volumes).sum()}"
            )

            # Combine with covariates or use lesion volumes alone
            if has_covariates:
                print(f"Loaded {covariates.shape[1]} additional covariates and lesion volume as covariate.")
                covariates = np.hstack([lesion_volumes, covariates])
            else:
                print("No additional covariates found in the CSV, using lesion volume as covariate.")
                covariates = lesion_volumes
        else:
            # No lesion volumes
            if has_covariates:
                print(f"Loaded {covariates.shape[1]} additional covariates.")
            else:
                print("No additional covariates found in the CSV (and regress_out_lesion_volume is False).")
                covariates = None

        # Z-transform covariates
        if covariates is not None:
            covariates = StandardScaler().fit_transform(covariates)
    else:
        print("SKIPPED Covariate processing and lesion volume computation.")

    return lesion_files, behaviors, covariates, lesion_volumes
