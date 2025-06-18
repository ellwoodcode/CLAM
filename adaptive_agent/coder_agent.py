import os
import time
import subprocess
import openai
from datetime import datetime

# === CONFIGURATION ===
EXPERIMENT_DIR = "experiments"
MAIN_TEMPLATE_PATH = "main.py"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "codex-mini-latest"  # or replace with best-performing Codex variant

openai.api_key = OPENAI_API_KEY

# Evaluation configuration defaults
DEFAULT_RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
DEFAULT_DATA_ROOT = os.getenv("DATA_ROOT_DIR", "./data")
DEFAULT_TASK = os.getenv("TASK", "task_1_tumor_vs_normal")
DEFAULT_MODEL_TYPE = os.getenv("MODEL_TYPE", "clam_sb")
DEFAULT_EMBED_DIM = os.getenv("EMBED_DIM", "1024")
DEFAULT_K = os.getenv("K_FOLDS", "3")
DEFAULT_MODELS_EXP = os.getenv("MODELS_EXP_CODE", "None_s1")

# === FUNCTIONS ===
def generate_experiment_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"main_experiment_{timestamp}.py"

def get_log_filename(script_name):
    return script_name.replace(".py", ".log")

def build_script_from_template(template_path, strategy_spec):
    with open(template_path, "r") as file:
        template = file.read()

    prompt = (
        f"You are a code-writing agent. Based on the following experiment strategy, rewrite the provided Python training script "
        f"so that it incorporates the proposed changes. Do not change unrelated lines.\n\n"
        f"--- STRATEGY SPECIFICATION ---\n{strategy_spec}\n\n"
        f"--- ORIGINAL SCRIPT ---\n{template}\n\n"
        f"--- MODIFIED SCRIPT ---"
    )

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert machine learning engineer and coder."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

def save_script(script_text, path):
    with open(path, "w") as file:
        file.write(script_text)

def run_script_and_log(script_path):
    log_path = get_log_filename(script_path)
    with open(log_path, "w") as logfile:
        process = subprocess.run(
            ["python", script_path],
            stdout=logfile,
            stderr=subprocess.STDOUT,
            timeout=None  # Remove timeouts due to model training
        )
    return process.returncode == 0, log_path

def retry_until_success(strategy_spec, max_retries=5):
    for attempt in range(max_retries):
        print(f"[Coder Agent] Attempt {attempt + 1}...")
        script_name = generate_experiment_filename()
        script_path = os.path.join(EXPERIMENT_DIR, script_name)

        modified_script = build_script_from_template(MAIN_TEMPLATE_PATH, strategy_spec)
        save_script(modified_script, script_path)

        success, log_path = run_script_and_log(script_path)

        if success:
            print(f"[Coder Agent] Experiment completed successfully. Log saved at {log_path}")
            return True, script_path, log_path
        else:
            print(f"[Coder Agent] Run failed. Log saved at {log_path}. Retrying...\n")

    return False, script_path, log_path

def try_implement_method(method_info, max_retries=5):
    """Attempt to implement and evaluate a method.

    Parameters
    ----------
    method_info : dict
        Dictionary containing ``id``, ``method`` and ``justification`` strings.
    max_retries : int, optional
        Number of attempts for ``retry_until_success``.

    Returns
    -------
    dict
        ``{"success", "auc", "cohort_aucs", "notes", "script_path", "log_path"}``
    """

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    success, script_path, log_path = retry_until_success(
        method_info.get("justification", ""), max_retries=max_retries
    )

    auc = 0.0
    cohort_aucs = {}
    notes = ""

    if success:
        try:
            from adaptive_agent.evaluate_all import evaluate_all_cohorts

            eval_result = evaluate_all_cohorts(
                models_exp_code=DEFAULT_MODELS_EXP,
                results_dir=DEFAULT_RESULTS_DIR,
                save_exp_prefix=method_info.get("id", "exp"),
                task=DEFAULT_TASK,
                model_type=DEFAULT_MODEL_TYPE,
                data_root_dir=DEFAULT_DATA_ROOT,
                embed_dim=DEFAULT_EMBED_DIM,
                k=DEFAULT_K,
            )

            auc = eval_result.get("mean_auc", 0.0)
            cohort_aucs = eval_result.get("cohort_aucs", {})
            success = eval_result.get("success", False)
        except Exception as e:
            notes = f"Evaluation failed: {e}"
            success = False
    else:
        notes = "Training failed"

    return {
        "success": success,
        "auc": auc,
        "cohort_aucs": cohort_aucs,
        "notes": notes,
        "script_path": script_path,
        "log_path": log_path,
    }

# === MAIN ENTRY ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Coder Agent with a strategy spec.")
    parser.add_argument("--strategy", type=str, required=True, help="The strategy spec string.")
    args = parser.parse_args()

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    result = retry_until_success(args.strategy)
    if result[0]:
        print(f"[Coder Agent] Final experiment script: {result[1]}")
    else:
        print("[Coder Agent] All attempts failed.")
