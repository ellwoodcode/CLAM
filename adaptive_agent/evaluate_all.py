# evaluate_all.py

import subprocess
import os
import pandas as pd

DEFAULT_EVALS = [
    ("Georgia", "Georgia_filtered"),
    ("Fondaz", "Lombardy-Italy_filtered"),
    ("UNSW", "UNSW_filtered"),
    ("Milan_Fondaz", "Milan-Italy_filtered"),
    ("Minnesota", "Minnesota_filtered"),
    ("Carolina", "N.Carolina_filtered"),
    ("Ohio", "Ohio_filtered"),
    ("Utah", "Utah_filtered"),
]

def evaluate_all_cohorts(models_exp_code, results_dir, save_exp_prefix, task, model_type,
                          data_root_dir, embed_dim="1024", k="3", eval_list=DEFAULT_EVALS,
                          eval_script="eval.py", verbose=True):
    """
    Runs evaluation across all external cohorts and returns mean AUC + breakdown.

    Parameters:
        models_exp_code (str): Directory name under results_dir containing the model checkpoints
        results_dir (str): Path to folder containing saved model
        save_exp_prefix (str): Prefix to use for each cohort's experiment result directory
        task (str): Task type, e.g., 'task_1_tumor_vs_normal'
        model_type (str): Type of model, e.g., 'clam_sb', 'clam_mb'
        data_root_dir (str): Base directory of the evaluation feature files
        embed_dim (str): Embedding size
        k (str): Number of folds
        eval_list (list): List of (cohort, label_file) tuples
        eval_script (str): Evaluation script to use (default: 'eval.py')
        verbose (bool): Whether to print progress

    Returns:
        dict: {
            "mean_auc": float,
            "cohort_aucs": dict,
            "success": bool
        }
    """
    cohort_aucs = {}
    for cohort, label_file in eval_list:
        save_exp_code = f"{save_exp_prefix}_{cohort}"
        if verbose:
            print(f"\n>>> Evaluating {models_exp_code} on {cohort}...\n")

        cmd = [
            "python", eval_script,
            "--k", k,
            "--models_exp_code", models_exp_code,
            "--save_exp_code", save_exp_code,
            "--task", task,
            "--model_type", model_type,
            "--results_dir", results_dir,
            "--data_root_dir", data_root_dir,
            "--embed_dim", embed_dim,
            "--cohort", cohort,
            "--cohort_labels", label_file
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"❌ ERROR during eval for {save_exp_code}")
            continue

        summary_path = os.path.join("eval_results", f"EVAL_{save_exp_code}", "summary.csv")
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            cohort_auc = df['test_auc'].mean()
            cohort_aucs[cohort] = cohort_auc
        else:
            print(f" No summary.csv found for {cohort}")
            cohort_aucs[cohort] = None

    valid_aucs = [auc for auc in cohort_aucs.values() if auc is not None]
    mean_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else 0.0

    return {
        "mean_auc": mean_auc,
        "cohort_aucs": cohort_aucs,
        "success": bool(valid_aucs)
    }
