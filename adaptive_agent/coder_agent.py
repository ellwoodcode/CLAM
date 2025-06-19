import os
import time
import subprocess
import openai
from datetime import datetime
import re

# === CONFIGURATION ===
EXPERIMENT_DIR = "experiments"
MAIN_TEMPLATE_PATH = "main.py"
OPENAI_API_KEY = ""
MODEL_NAME = "gpt-4o-2024-08-06"  # or replace with best-performing Codex variant

openai.api_key = OPENAI_API_KEY

# Evaluation configuration defaults
DEFAULT_RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
DEFAULT_DATA_ROOT = os.getenv("DATA_ROOT_DIR", "C:/Users/Mahon/Documents/Research/CLAM/Phase3A_Baseline_Features/Train/TCGA_ims1")
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

def clean_generated_code(code):
    match = re.search(r"```(?:python)?\n(.*?)\n```", code, re.DOTALL)
    return match.group(1).strip() if match else code.strip()

def build_script_from_template(template_path, strategy_spec):
    with open(template_path, "r") as file:
        template = file.read()

    prompt = (
        f"You are a code-writing agent. Based on the following experiment strategy, rewrite the provided Python training script "
        f"so that it incorporates the proposed changes. Do not change unrelated lines. If you modify or regenerate the full script, "
        f"you must preserve all necessary imports. Add the following lines at the top of the script (but below any `__future__` imports) "
        f"to ensure correct module paths:\n\n"
        f"import sys, os\n"
        f"sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))\n\n"
        f"--- STRATEGY SPECIFICATION ---\n{strategy_spec}\n\n"
        f"--- ORIGINAL SCRIPT ---\n{template}\n\n"
        f"--- MODIFIED SCRIPT ---"
    )

    response = openai.chat.completions.create(
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

def _extract_failure_excerpt(log_lines):
    """Return a short excerpt around the first detected failure keyword."""
    keywords = ["traceback", "exception", "nan"]
    for idx, line in enumerate(log_lines):
        lower = line.lower()
        if any(k in lower for k in keywords):
            start = max(0, idx - 2)
            end = min(len(log_lines), idx + 3)
            return "".join(log_lines[start:end]).strip()
    return ""


def run_script_and_log(script_path):
    log_path = get_log_filename(script_path)
    with open(log_path, "w") as logfile:
        process = subprocess.run(
            ["python", script_path],
            stdout=logfile,
            stderr=subprocess.STDOUT,
            timeout=None  # Remove timeouts due to model training
        )

    success = process.returncode == 0
    notes = ""

    try:
        with open(log_path, "r") as f:
            log_lines = f.readlines()

        if not success:
            notes = "".join(log_lines[-10:]).strip()
        else:
            excerpt = _extract_failure_excerpt(log_lines)
            if excerpt:
                success = False
                notes = excerpt
    except Exception as e:
        notes = f"Could not read log: {e}"

    return success, log_path, notes

def retry_until_success(strategy_spec, max_retries=5):
    notes = ""
    for attempt in range(max_retries):
        print(f"[Coder Agent] Attempt {attempt + 1}...")
        script_name = generate_experiment_filename()
        script_path = os.path.join(EXPERIMENT_DIR, script_name)

        modified_script = build_script_from_template(MAIN_TEMPLATE_PATH, strategy_spec)
        modified_script = clean_generated_code(modified_script)
        save_script(modified_script, script_path)

        success, log_path, run_notes = run_script_and_log(script_path)
        if run_notes:
            notes += f"Attempt {attempt + 1}:\n{run_notes}\n"

        if success:
            print(f"[Coder Agent] Experiment completed successfully. Log saved at {log_path}")
            return True, script_path, log_path, notes.strip()
        else:
            print(f"[Coder Agent] Run failed. Log saved at {log_path}. Retrying...\n")

    return False, script_path, log_path, notes.strip()

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

    success, script_path, log_path, run_notes = retry_until_success(
        method_info.get("justification", ""), max_retries=max_retries
    )

    auc = 0.0
    cohort_aucs = {}
    notes = run_notes or ""

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
            notes = (notes + "\n" if notes else "") + f"Evaluation failed: {e}"
            success = False
    else:
         notes = f"Training failed:\n{notes}" if notes else "Training failed"

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

    success, script_path, _log_path, _notes = retry_until_success(args.strategy)
    if success:
        print(f"[Coder Agent] Final experiment script: {script_path}")
    else:
        print("[Coder Agent] All attempts failed.")
