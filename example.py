# svrLSMpy/examples.py (or svrLSMpy/main.py)
from pathlib import Path
from .run_svr_lsm_iteration import run_svr_lsm_iteration

def run_example():
    base_folder = Path.cwd() / "symptoms"

    symptom_folder = base_folder / 'VAST'
    csv_name = "VAST_Data_Fluency.csv"

    param_grid = {
        'C': [50],
        'gamma': [2],
        'epsilon': [0.1]
    }

    behaviour_name = "Aphasia"
    run_svr_lsm_iteration(
        symptom_folder=symptom_folder,
        csv_name=csv_name,
        behaviour_name=behaviour_name,
        normalize_vector=True,
        do_regress_out_lesion_volume=True,
        max_score=4,
        min_patient_count="10%",
        param_grid=param_grid,
        n_permutations=1,
        alpha=0.05,
        n_splits=2,
        num_slices=7
    )
