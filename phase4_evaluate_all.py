import subprocess
import os

# --- Configuration ---
# Path to the main directory where CLAM and other related code (like utils, dataset_modules) are.
# This is needed if this script is not in the same top-level directory as 'eval_rna.py' and its imports.
# If they are in the same directory, or PYTHONPATH is set, this might not be strictly needed.
# PROJECT_ROOT = "/path/to/your/clam_project_root" 

# Models and Results (Update these paths and codes for your RNA-trained models)
models_exp_code = "IMS1_RNA_integration_s1"  # Experiment code of your RNA-trained model
save_exp_prefix = "IMS1_RNA_on"       # Prefix for saving evaluation results
task = "task_1_tumor_vs_normal"
model_type = "clam_mb" # Or clam_sb, consistent with your trained RNA model
results_dir = "./IMS1_RNA_Model" # Directory where models_exp_code is located
data_root_dir = "./Phase3A_Baseline_Features/Evals" # Root for patch feature evaluation cohorts
k_folds_for_eval = "3" # Number of folds your model was trained on (used to find checkpoints s_0, s_1, etc.)

# RNA Specific Configuration (MUST BE SET CORRECTLY)
RNA_DATA_BASE_DIR = "C:/Users/Mahon/Documents/Research/CLAM/rna_seq_files_processed_log_transformed" # e.g., "./Processed_RNA_NPYs"
MASTER_RNA_DIM = 19944
ORIGINAL_PATCH_FEATURE_DIM = 1024
# CALCULATED_EMBED_DIM = ORIGINAL_PATCH_FEATURE_DIM + MASTER_RNA_DIM # This will be calculated in eval_rna.py

# List of cohort-label pairs for evaluation
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
for cohort_name, label_file_basename in eval_cohorts:
    current_save_exp_code = f"{save_exp_prefix}_{cohort_name}"
    print(f"\n>>> Evaluating RNA Model: {models_exp_code} on Cohort: {cohort_name}\n")

    # Construct the command for eval_rna.py
    cmd = [
        "python", "phase4_eval_rna.py", # UPDATED to call eval_rna.py
        "--k", k_folds_for_eval,
        "--models_exp_code", models_exp_code,
        "--save_exp_code", current_save_exp_code,
        "--task", task,
        "--model_type", model_type,
        "--results_dir", results_dir,
        "--data_root_dir", data_root_dir,
        # "--embed_dim", str(CALCULATED_EMBED_DIM), # Let eval_rna.py calculate it
        "--cohort", cohort_name,
        "--cohort_labels", label_file_basename,
        
        # RNA-specific arguments
        "--use_rna_concatenation", # Add this flag to enable RNA features
        "--rna_data_base_dir", RNA_DATA_BASE_DIR,
        "--master_rna_dim", str(MASTER_RNA_DIM),
        "--original_patch_feature_dim", str(ORIGINAL_PATCH_FEATURE_DIM)
        # Add other arguments if your eval_rna.py expects them (e.g., --model_size, --drop_out)
        # For example:
        # "--model_size", "small", 
        # "--drop_out", "0.25", 
    ]

    print(f"Running command: {' '.join(cmd)}")
    
    # Ensure the script can find 'eval_rna.py' and its imports.
    # If eval_rna.py and its dependencies (utils, dataset_modules) are not in the current
    # working directory or Python's path, you might need to adjust how it's called
    # or set PYTHONPATH environment variable.
    # env = os.environ.copy()
    # if PROJECT_ROOT:
    #     env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    # result = subprocess.run(cmd, env=env)
    
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"❌ ERROR during RNA model evaluation for {current_save_exp_code} on cohort {cohort_name}")
    else:
        print(f"✅ Successfully completed RNA model evaluation for {current_save_exp_code} on cohort {cohort_name}")

print("\n--- All RNA evaluations finished ---")

