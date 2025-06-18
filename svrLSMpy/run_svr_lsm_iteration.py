from modules.atlas_read_cluster import atlas_read_cluster
from modules.time_func import get_current_datetime_for_filename
from modules.time_func import easy_time
from modules.load_lesions_and_behaviors import load_lesions_and_behaviors
from modules.filter_voxels_by_patient_count import filter_voxels_by_patient_count
from modules.regress_covariates_from_behavior import regress_covariates_from_behavior
from modules.svr_lsm import svr_lsm
from modules.save_report import save_report

import time
from pathlib import Path
import numpy as np

def run_svr_lsm_iteration(symptom_folder, csv_name,behaviour_name,do_regress_out_lesion_volume, normalize_vector, max_score,min_patient_count, param_grid, n_permutations, alpha, n_splits, num_slices):
    # base_folder = Path.cwd()  # CURRENT DIRECTORY
    start_time = time.time()

    #Make folder to save all results
    symptom = symptom_folder.name


    # Load lesions and behaviors
    lesion_folder = symptom_folder / 'data'
    csv_file = symptom_folder / csv_name

    lesion_files, behaviors, covariates, lesion_volumes = load_lesions_and_behaviors(lesion_folder, csv_file, max_score, do_regress_out_lesion_volume)
    print("\n\tTIME ELAPSED : ", easy_time(int(time.time() - start_time)),end="\n\n")

    behaviors = regress_covariates_from_behavior(behaviors, covariates)
    print("\n\tTIME ELAPSED : ", easy_time(int(time.time() - start_time)), end="\n\n")
    output_folder = f"outputs/{symptom}_{n_permutations}_results_{get_current_datetime_for_filename()}"
    output_folder = Path(output_folder)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    min_patient_count, features, masker = filter_voxels_by_patient_count(lesion_files, min_patient_count, normalize_vector,output_folder)
    print("\n\tTIME ELAPSED : ", easy_time(int(time.time() - start_time)), end="\n\n")
    # Perform SVR-based lesion-symptom mapping
    svr_params, coef_map, nifti_zmap, zmap = svr_lsm(features=features,
                                                      behaviors=behaviors,
                                                      masker=masker,
                                                      output_folder=output_folder,
                                                      param_grid=param_grid,
                                                      n_permutations=n_permutations,
                                                      alpha=alpha,
                                                      n_splits=n_splits)

    # Dataset statistics
    num_lesions = len(lesion_files)
    num_patients = len(behaviors)
    mean_lesion_volume = np.mean(lesion_volumes)

    # Check for covariate information
    if covariates is not None:
        covariate_info = f"{covariates.shape[1]} additional covariates"
    else:
        covariate_info = None

    # Compute the zmap range
    zmap_range = (np.min(zmap), np.max(zmap))

    zmap_atlas_output_dir = output_folder / "atlasreader_output"

    atlas_read_cluster(nifti_zmap, zmap_atlas_output_dir)

    svr_lsm_report_path = output_folder / "svr_lsm_report.html"

    time_taken = easy_time(time.time() - start_time)


    # Save the report
    save_report(svr_lsm_report_path,
                svr_params,
                behaviour_name,
                n_permutations,
                alpha,
                zmap_range,
                zmap,
                min_patient_count,
                num_patients,
                num_slices,
                nifti_zmap,
                zmap_atlas_output_dir,
                time_taken,
                num_lesions,
                mean_lesion_volume)

    print("\n\tTOTAL TIME TAKEN : ", easy_time(int(time.time() - start_time)))
