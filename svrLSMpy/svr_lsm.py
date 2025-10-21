from sklearn.model_selection import KFold
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from nilearn import masking
import nibabel as nib
from itertools import product
import pickle
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from nilearn.image import threshold_img
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
import time
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from .util import easy_time


def svr_lsm(features, behaviors, masker, output_folder, param_grid, n_permutations=1, alpha=0.05, n_splits=5):
    """
    Perform SVR-based lesion-symptom mapping with K-fold cross-validation and permutation testing.
    """
    print("Running SVR analysis...")

    # Perform grid search with K-fold cross-validation
    print("Performing grid search with K-fold cross-validation...")
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    param_combinations = list(product(param_grid['C'], param_grid['gamma'], param_grid['epsilon']))

    best_params = {'C': param_grid['C'][0], 'gamma': param_grid['gamma'][0], 'epsilon': param_grid['epsilon'][0]}

    best_score = float('inf')
    best_iteration = 1

    patient_count = len(features)
    all_scores = []
    # Iterate over all combinations of hyperparameters
    i = 1
    num_iter = len(param_combinations)

    for idx, (C, gamma, epsilon) in enumerate(param_combinations, 1):
        print(f"\nIteration: {idx}/{num_iter}, Testing parameters: C={C}, gamma={gamma}, epsilon={epsilon}")
        iter_time = time.time()

        scores = []
        no_of_sv = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(features), 1):
            print(f"Split :{fold_idx}/{n_splits}")
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = behaviors[train_idx], behaviors[test_idx]

            svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
            svr.fit(X_train, y_train)

            print(f"\tno. of support vectors : {len(svr.support_)}/{patient_count}", )
            no_of_sv.append(len(svr.support_))

            predictions = svr.predict(X_test)
            score = mean_squared_error(y_test, predictions)

            # default coef of determination metrics
            print(f"\tscore : {score}", )

            scores.append(score)

        # Average score across all folds
        avg_score = np.mean(scores)
        avg_no_of_sv = np.mean(no_of_sv)

        all_scores.append((i, C, gamma, epsilon, avg_score, scores, avg_no_of_sv, no_of_sv))

        print(f"\nAverage score for mse: {avg_score:.4f}")
        print(scores, "\n")

        # Update best parameters if current score is better
        if avg_score < best_score:
            best_iteration = i
            best_score = avg_score
            best_params = {'C': C, 'gamma': gamma, 'epsilon': epsilon}

        i = i + 1
        print(f"Best iteration: {best_iteration}, Best score: {best_score:.4f}->0, Current Score: {avg_score:.4f}")
        print(f"Iteration {best_iteration} parameters: C = {best_params['C']}, gamma = {best_params['gamma']}, epsilon = {best_params['epsilon']}")

        print(f"Iteration time: {easy_time(int(time.time() - iter_time))}")


    columns = [
        "Iteration", "C", "Gamma", "Epsilon", "Avg_Score", "Scores", "Avg_Support_Vectors", "Support_Vectors"
    ]

    df = pd.DataFrame(all_scores, columns=columns)
    del all_scores

    # Save to CSV
    output_path = output_folder / 'results_and_scores.csv'
    df.to_csv(output_path, index=False)
    print(f"Results and scores saved to {output_path}")
    del df

    print(f"Best parameters found: {best_params} with score {best_score:.4f} in iteration {best_iteration}/{num_iter}")
    # Train SVR with the best parameters
    svr_best = SVR(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], epsilon=best_params['epsilon'])
    svr_best.fit(features, behaviors)
    coef_map = svr_best.support_vectors_.mean(axis=0)

    
    #save model for later predictions
    model_path = output_folder / 'svr_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(svr_best, f)
    print(f"Trained SVR model saved to {model_path}")
    
    masker_path = output_folder / 'masker.pkl'
    with open(masker_path, 'wb') as f:
        pickle.dump(masker, f)
    print(f"Masker saved to {masker_path}")

    
    #saving beta map
    nifti_coef_map = masking.unmask(coef_map, masker)
    nifti_coef_path = output_folder / 'beta_map.nii.gz'
    nib.save(nifti_coef_map, nifti_coef_path)

    # Permutation testing
    print("\nPerforming permutation testing...")

    null_params = best_params

    results_file = output_folder / "null_distributions.pkl"
    # Open the results file in binary write mode
    with open(results_file, 'wb') as f:
        permute_time = time.time()

        # Wrap the range with tqdm to show the progress bar
        time.sleep(1)
        with tqdm(range(n_permutations), desc="Running permutations", unit="permutation", mininterval=1, ncols=100, dynamic_ncols=True, leave=True) as pbar:
            for i in pbar:
                # Shuffle the behaviors for this permutation
                perm_behaviors = shuffle(behaviors, random_state=None)

                svr_permutation = SVR(kernel='rbf',
                                      C=null_params['C'],
                                      gamma=null_params['gamma'],
                                      epsilon=null_params['epsilon'])

                svr_permutation.fit(features, perm_behaviors)

                # Compute the mean of support vectors for this permutation
                vector_mean = svr_permutation.support_vectors_.mean(axis=0)

                # Save the result to the file incrementally
                pickle.dump(vector_mean, f)

                # Clear the perm_result from memory
                del vector_mean

                # Update the progress bar, displaying elapsed time and ETA
                elapsed_time = time.time() - permute_time
                pbar.set_postfix(elapsed=f"{easy_time(elapsed_time)}",eta=f"{easy_time((elapsed_time / (i + 1)) * (n_permutations - i - 1))}")

    print(f"Permutations completed. Null distribution saved to {results_file}\n")

    # Z-map Calculations
    # Incremental calculation of mean and std from saved null distributions
    sum_null = None
    sum_null_squared = None
    num_permutations = 0

    with open(results_file, 'rb') as f:
        pbar = tqdm()  # No total specified
        while True:
            try:
                perm_result = pickle.load(f)
    
                if sum_null is None:
                    sum_null = np.zeros_like(perm_result)
                    sum_null_squared = np.zeros_like(perm_result)
    
                sum_null += perm_result
                sum_null_squared += perm_result ** 2
                num_permutations += 1
    
                pbar.update(1)  # Update progress bar by 1
            except EOFError:
                break
        pbar.close()

    # Compute mean and standard deviation
    mean_null = sum_null / num_permutations

    # saving null map
    nifti_null_map = masking.unmask(mean_null, masker)
    nifti_null_path = output_folder / 'null_map.nii.gz'
    nib.save(nifti_null_map, nifti_null_path)

    variance = (sum_null_squared / num_permutations) - (mean_null ** 2)
    std_null = np.sqrt(np.maximum(variance, 0) + 1e-8)

    # Compute z-map
    zmap = (coef_map - mean_null) / std_null

    # Plot the Histogram

    zmap_flat = zmap[zmap != 0]

    plt.hist(zmap_flat, bins=50, density=True, alpha=0.6, color='blue', label='Z-map distribution')

    # Fit a normal distribution (optional)
    mean, std = np.mean(zmap_flat), np.std(zmap_flat)
    x = np.linspace(min(zmap_flat), max(zmap_flat), 100)
    pdf = norm.pdf(x, mean, std)

    # Overlay the normal distribution
    plt.plot(x, pdf, 'r-', label=f'Normal dist (μ={mean:.2f}, σ={std:.2f})')

    # Set x-axis to be symmetric around 0
    plt.xlim(left=-max(abs(min(zmap_flat)), abs(max(zmap_flat))), right=max(abs(min(zmap_flat)), abs(max(zmap_flat))))

    # Add labels and legend
    plt.title('Z-Map Distribution')
    plt.xlabel('Z-score')
    plt.ylabel('Density')
    plt.legend()

    plt.savefig(output_folder / 'z_value_distribution.png')
    del zmap_flat

    zmap_threshold_output_folder = output_folder / "thresholded_zmaps"
    Path(zmap_threshold_output_folder).mkdir(parents=True, exist_ok=True)

    # Unmask the z-map back to a 3D image
    print("Unmasking z-map...")
    nifti_zmap = masking.unmask(zmap, masker)
    nifti_zmap_path = output_folder / 'zmap.nii.gz'
    nib.save(nifti_zmap, nifti_zmap_path)

    thresholds = [
        (1.644854, 'p05', 'p<0.05'),
        (2.326348, 'p01', 'p<0.01'),
        (2.575829, 'p005', 'p<0.005'),
        (3.090232, 'p001', 'p<0.001')
    ]
    
    for thresh_val, label, desc in thresholds:
        print(f"Thresholding z-map at {desc}...")
        nifti_zmap_thresh = threshold_img(nifti_zmap, threshold=thresh_val, cluster_threshold=30)
        nib.save(nifti_zmap_thresh, zmap_threshold_output_folder / f'zmap_{label}.nii.gz')

    return best_params, coef_map, nifti_zmap, zmap





