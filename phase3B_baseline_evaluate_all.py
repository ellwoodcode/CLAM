import subprocess

# Configuration
# Change IMSx appropriately
models_exp_code = "IMS2_Baseline_s1"
save_exp_prefix = "IMS2_on"
task = "task_1_tumor_vs_normal"
model_type = "clam_mb"
results_dir = "./IMS2_Baseline_Model"
data_root_dir = "./Phase3A_Baseline_Features/Evals"
embed_dim = "1024"
k = "3"

# List of cohort-label pairs
evals = [
    ("Georgia", "Georgia_filtered"),
    ("Fondaz", "Lombardy-Italy_filtered"),
    ("UNSW", "UNSW_filtered"),
    ("Milan_Fondaz", "Milan-Italy_filtered"),
    ("Minnesota", "Minnesota_filtered"),
    ("Carolina", "N.Carolina_filtered"),
    ("Ohio", "Ohio_filtered"),
    ("Utah", "Utah_filtered"),
]

for cohort, label_file in evals:
    save_exp_code = f"{save_exp_prefix}_{cohort}"
    print(f"\n>>> Evaluating: {models_exp_code} on {cohort}\n")

    cmd = [
        "python", "eval.py",
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
