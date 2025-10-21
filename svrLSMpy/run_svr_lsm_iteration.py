from .atlas_read_cluster import atlas_read_cluster
from .util import get_current_datetime_for_filename, easy_time
from .load_lesions_and_behaviors import load_lesions_and_behaviors
from .filter_voxels_by_patient_count import filter_voxels_by_patient_count
from .regress_covariates_from_behavior import regress_covariates_from_behavior
from .regress_covariates_from_lesions import regress_covariates_from_lesions
from .svr_lsm import svr_lsm
from .save_report import save_report

import time
from pathlib import Path
import numpy as np


def run_svr_lsm_iteration(symptom_folder,
                          csv_path,
                          max_score,
                          output_path,
                          behaviour_name="behavioural_deficit",
                          regress_out_lesion_volume=True,
                          regress_out_covariates_on_scores=True,
                          regress_out_covariates_on_lesions=True,
                          normalize_vector=False,
                          min_patient_count='10%', 
                          param_grid = {
                                        'C': [50, 40, 30, 20, 10, 5, 1],
                                        'gamma': [50, 10, 5, 4, 3, 2, 1, 'scale', 'auto'],
                                        'epsilon': [0.1, 0.05, 0.01]
                                        },
                          n_permutations=1000, 
                          alpha=0.05, 
                          n_splits=5, 
                          num_slices=7):
    # base_folder = Path.cwd()  # CURRENT DIRECTORY
    start_time = time.time()

    symptom = behaviour_name
    output_folder = f"{output_path}/{symptom}_{n_permutations}_results_{get_current_datetime_for_filename()}"
    output_folder = Path(output_folder)
                              
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    #Make folder to save all results

    # Load lesions and behaviors
    lesion_folder = Path(symptom_folder)

    lesion_files, behaviors, covariates, lesion_volumes = load_lesions_and_behaviors(lesion_folder, csv_path, max_score, regress_out_lesion_volume, regress_out_covariates_on_scores)
    print("\n\tTIME ELAPSED : ", easy_time(int(time.time() - start_time)),end="\n\n")

    min_patient_count, features, masker = filter_voxels_by_patient_count(lesion_files, min_patient_count, normalize_vector, output_folder)
    print("\n\tTIME ELAPSED : ", easy_time(int(time.time() - start_time)), end="\n\n")

    if covariates is not None:
        if regress_out_covariates_on_scores:
            behaviors = regress_covariates_from_behavior(behaviors, covariates)
            print("\n\tTIME ELAPSED : ", easy_time(int(time.time() - start_time)), end="\n\n")
        else:
            print("covariates not regressed from behavioral score")
      
        if regress_out_covariates_on_lesions:
            non_regressed_features = features
            try:
                print("Running... Press Ctrl+C to stop")
                features = regress_covariates_from_lesions(features, covariates)
                print("\n\tTIME ELAPSED : ", easy_time(int(time.time() - start_time)), end="\n\n")
                  
            except KeyboardInterrupt:
                print("Lesion file covariate regression cancelled by user!")
                regress_out_covariates_on_lesions = False
                features = non_regressed_features


          
        else:
            print("covariates not regressed from lesion file")
    else:
        print("\n\nNo covariates present\n\n")

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

    # Compute the zmap range
    zmap_range = (np.min(zmap), np.max(zmap))

    zmap_atlas_output_dir = output_folder / "atlasreader_output"

    atlas_read_cluster(nifti_zmap, zmap_atlas_output_dir)

    svr_lsm_report_path = output_folder / "svr_lsm_report.html"

    time_taken = easy_time(time.time() - start_time)

  
    # Print for covariate information
    if covariates is None:
        print(f"covariates is None")
    else:
        print(f"covariates={covariates}")
      
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
                mean_lesion_volume,
                covariates,
                regress_out_lesion_volume,
                regress_out_covariates_on_scores,
                regress_out_covariates_on_lesions,
                normalize_vector)

    print("\n\tTOTAL TIME TAKEN : ", easy_time(int(time.time() - start_time)))









