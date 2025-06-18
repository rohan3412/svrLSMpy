from svrLSMpy import run_svr_lsm_iteration

run_svr_lsm_iteration(
    symptom_folder="proj/data-final",
    csv_path="proj/data-no-covariates.csv",
    max_score=100,
    output_path="test-package-output"
)
