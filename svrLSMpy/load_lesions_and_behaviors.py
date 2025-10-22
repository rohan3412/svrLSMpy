from sklearn.preprocessing import StandardScaler
from pandas import read_csv
import numpy as np
import nibabel as nib
import os
from tqdm import tqdm

def load_lesions_and_behaviors(lesion_folder, csv_file, max_score, do_regress_out_lesion_volume, do_regress_out_covariates):
    """
    Load lesion files, behavioral data, and covariates from CSV, and compute lesion volumes.
    """
    print("Loading behavioral data and lesion files...")
    df = read_csv(csv_file)

    # Check for required columns
    required_columns = ['filename', 'behavior']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"CSV file must contain '{col}' column.")

    def normalize(name):
        # Lowercase, remove underscores, hyphens, and spaces
        return re.sub(r'[_\-\s]', '', name.lower())

    lesion_files = []
    
    for f in df['filename']:
        norm_f = normalize(f)
        # Find all files in lesion_folder where normalized filename contains normalized f
        matches = [file for file in os.listdir(lesion_folder) if norm_f in normalize(file)]
    
        if len(matches) == 0:
            raise FileNotFoundError(f"No files found in '{lesion_folder}' containing '{f}' (normalized as '{norm_f}') as a substring.")
        elif len(matches) > 1:
            raise ValueError(
                f"Multiple files found in '{lesion_folder}' containing '{f}' (normalized as '{norm_f}') as a substring:\n" +
                "\n".join(matches)
            )
        else:
            lesion_files.append(os.path.join(lesion_folder, matches[0]))
    
    behaviors = df['behavior'].values

    print('\nBehavior before:\n', behaviors)

    behaviors = behaviors / max_score

    print('\nBehavior after:\n', behaviors)

    # Compute lesion volumes

    print("\n")
    lesion_volumes = []
    for file in tqdm(lesion_files, desc="Loading lesions and computing volumes..."):
        lesion_img = nib.load(file)
        lesion_data = lesion_img.get_fdata()
        voxel_volume = np.prod(lesion_img.header.get_zooms())
        lesion_volumes.append(np.sum(lesion_data > 0) * voxel_volume)
    lesion_volumes = np.array(lesion_volumes).reshape(-1, 1)

    # Load additional covariates
    covariates = df.drop(columns=["filename", "behavior"])
    covariates = covariates.select_dtypes(include=["number"])

    if covariates.isna().any().any():
        covariates = covariates.fillna(covariates.mean())
        print("Filled NaN values in covariates with column means.")

    if covariates.shape[1] > 0:
        if do_regress_out_covariates:
            print(f"Loaded {covariates.shape[1]} additional covariates",end='')
            # Combine lesion volumes with additional covariates
            if do_regress_out_lesion_volume:
                print(" and lesion volume as covariate")
                covariates = np.hstack([lesion_volumes, covariates])
            else:
                print("\nand lesion volume not regressed out as covariate")
                covariates = np.hstack([covariates])

    elif do_regress_out_lesion_volume:
            print("Loaded lesion volume only as covariate")
            covariates = np.hstack([lesion_volumes])
    else:
        print("No additional covariates found in the CSV.")
        covariates = None

    if covariates is not None:
        # Z-transform covariates
        scaler = StandardScaler()

        covariates = scaler.fit_transform(covariates)

    return lesion_files, behaviors, covariates, lesion_volumes



