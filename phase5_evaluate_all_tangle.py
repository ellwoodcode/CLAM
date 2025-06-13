import subprocess
import os

# --- Configuration ---
# Details of the TANGLE-INTEGRATED model you want to evaluate
models_exp_code = "IMS1_Tangle_Integration_s1" # From your training command, with "_s1" suffix for seed 1
save_exp_prefix = "IMS1_Tangle_on"
task = "task_1_tumor_vs_normal"
model_type = "clam_mb"
results_dir = "./IMS1_Tangle_Model"          # Directory where the Tangle model was saved
data_root_dir = "./Phase3A_Baseline_Features/Evals" # Root dir for patch features of eval cohorts
k = "3"                                      # Number of folds the model was trained on

# Tangle Specific Configuration
# --- MODIFICATION ---
# This is now the BASE directory that CONTAINS the cohort subdirectories.
TANGLE_BASE_DIR = "C:/Users/Mahon/Documents/Research/CLAM/EVAL_tangle_embeddings" 
ORIGINAL_PATCH_FEATURE_DIM = "1024"
TANGLE_EMBEDDING_DIM = "1024"

# --- List of cohorts for evaluation ---
evals = [
    ("Georgia", "Georgia_filtered_cohort"),
    ("Fondaz", "Lombardy-Italy_filtered_cohort"),
    ("UNSW", "UNSW_filtered_cohort"),
    ("Milan_Fondaz", "Milan-Italy_filtered_cohort"),
    ("Minnesota", "Minnesota_filtered_cohort"),
    ("Carolina", "N.Carolina_filtered_cohort"),
    ("Ohio", "Ohio_filtered_cohort"),
    ("Utah", "Utah_filtered_cohort"),
]

# --- Evaluation Loop ---
for cohort, label_file in evals:
    save_exp_code = f"{save_exp_prefix}_{cohort}"
    print(f"\n>>> Evaluating Tangle Model: {models_exp_code} on Cohort: {cohort}\n")

    # --- MODIFICATION ---
    # Dynamically construct the path to the cohort's specific tangle embedding subdirectory.
    cohort_tangle_dir = os.path.join(TANGLE_BASE_DIR, cohort)

    # Check if the subdirectory for tangle embeddings exists
    if not os.path.isdir(cohort_tangle_dir):
        print(f"❌ ERROR: Tangle embedding subdirectory not found for {cohort} at: {cohort_tangle_dir}")
        continue

    cmd = [
        "python", "phase5_eval_tangle.py",
        "--k", k,
        "--models_exp_code", models_exp_code,
        "--save_exp_code", save_exp_code,
        "--task", task,
        "--model_type", model_type,
        "--results_dir", results_dir,
        "--data_root_dir", data_root_dir,
        "--cohort", cohort,
        "--cohort_labels", label_file,
        
        # Tangle-specific arguments
        "--use_tangle_concatenation",
        # --- MODIFICATION ---
        # Pass the dynamically created, cohort-specific path to the evaluation script.
        "--tangle_feature_dir", cohort_tangle_dir,
        "--original_patch_feature_dim", ORIGINAL_PATCH_FEATURE_DIM,
        "--tangle_embedding_dim", TANGLE_EMBEDDING_DIM,
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"❌ ERROR during Tangle model evaluation for {save_exp_code}")
    else:
        print(f"✅ Successfully completed Tangle model evaluation for {save_exp_code}")

print("\n--- All Tangle model evaluations finished ---")