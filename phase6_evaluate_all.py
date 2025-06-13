import subprocess
import os

# --- Configuration ---
# This section contains the paths to your trained models and data directories.

# 1. Paths to your two "expert" models.
CLAM_MODEL_PATH = "C:/Users/Mahon/Documents/Research/CLAM/IMS1_Baseline_Model/IMS1_Baseline_s1/s_0_checkpoint.pt"
RNA_MODEL_PATH = "C:/Users/Mahon/Documents/Research/CLAM/rna_mlp_model/rna_mlp_final.pt"

# 2. Paths to your evaluation data directories.
# These should match the defaults in your eval script.
LABEL_DIR = 'C:/Users/Mahon/Documents/Research/CLAM/Labels/Textual/'
PATCH_DIR = './Phase3A_Baseline_Features/Evals'
TANGLE_DIR = 'C:/Users/Mahon/Documents/Research/CLAM/EVAL_tangle_embeddings'

# --- List of Cohorts for Evaluation ---
# (Folder name for patch/tangle data, Basename of the label CSV file)
eval_cohorts = [
    ("Georgia", "Georgia_filtered"),
    ("Fondaz", "Lombardy-Italy_filtered"),
    ("UNSW", "UNSW_filtered"),
    ("Milan_Fondaz", "Milan-Italy_filtered"),
    ("Minnesota", "Minnesota_filtered"),
    ("Carolina", "N.Carolina_filtered"),
    ("Ohio", "Ohio_filtered"),
    ("Utah", "Utah_filtered"),
]


# --- Evaluation Loop ---
print("--- Starting Late Fusion Evaluation on All Cohorts ---")

for cohort_name, label_file_basename in eval_cohorts:
    print(f"\n{'='*60}")
    print(f">>> Evaluating Cohort: {cohort_name}")
    print(f"{'='*60}")

    # Construct the command to call your evaluation script
    cmd = [
        "python", "./phase6_eval_late_fusion.py",
        "--cohort", cohort_name,
        "--cohort_labels", label_file_basename,
        "--clam_model_path", CLAM_MODEL_PATH,
        "--rna_model_path", RNA_MODEL_PATH,
        "--label_dir", LABEL_DIR,
        "--patch_dir", PATCH_DIR,
        "--tangle_dir", TANGLE_DIR,
    ]

    print(f"Running command: {' '.join(cmd)}")
    
    # Execute the command
    result = subprocess.run(cmd)

    # Check for errors during the evaluation of each cohort
    if result.returncode != 0:
        print(f"\n❌ ERROR during evaluation for cohort: {cohort_name}")
    else:
        print(f"\n✅ Successfully finished evaluation for cohort: {cohort_name}")

print(f"\n{'='*60}")
print("--- All evaluations complete. ---")