import os
import time
import subprocess
import openai
from datetime import datetime
import re

# === CONFIGURATION ===
EXPERIMENT_DIR = "experiments"
MAIN_TEMPLATE_PATH = "main_template.py"
OPENAI_API_KEY = ""
MODEL_NAME = "gpt-4o-2024-08-06"  # or replace with best-performing Codex variant

openai.api_key = OPENAI_API_KEY

# Evaluation configuration defaults
DEFAULT_RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
DEFAULT_DATA_ROOT = os.getenv("DATA_ROOT_DIR", "C:/Users/Mahon/Documents/Research/CLAM/Phase3A_Baseline_Features/Train/TCGA_ims1")
DEFAULT_TASK = os.getenv("TASK", "task_1_tumor_vs_normal")
DEFAULT_MODEL_TYPE = os.getenv("MODEL_TYPE", "clam_mb")
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

def build_script_from_template(template_path, strategy_spec, first_attempt=True):
    with open(template_path, "r") as file:
        template = file.read()
    if first_attempt:
        prompt = (
            "You are an expert code-writing agent. Based on the following experiment strategy, rewrite the provided Python training script "
            "so that it incorporates the proposed changes. You must ensure the modified script is executable and self-contained. Specific requirements:\n\n"
            "- All new functions (e.g., losses, dataloaders, training routines) must be fully **integrated into the training pipeline**.\n"
            "- If you add optional features (e.g., `use_ssl`, `use_dino`, `ssl_epochs`), you must:\n"
            "  • Add new `argparse` arguments with appropriate default values\n"
            "  • Set any new booleans like `use_ssl` to `True` by default to activate the new feature for testing\n"
            "  • Pass these arguments where needed in the code (e.g., `train()` or dataset loaders)\n"
            "- Do **not** include unused stubs or define functions that are never called\n"
            "- You **may restructure logic**, but must preserve unrelated existing functionality\n"
            "- **Do not** import functions from the same file — reference them directly instead\n"
            "- Keep **all logic in one file** — do not split across modules\n\n"
            "You must also insert the following two lines directly after any `from __future__` imports (if present):\n"
            "import sys, os\n"
            "sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))\n\n"
            "=== CONTEXT ON FILE STRUCTURE AND DATA ACCESS ===\n"
            "Patch features are stored as .pt files in folders named `pt_files/`, and h5 feature files are in `h5_files/` folders. "
            "These are organized under:\n"
            "`C:/Users/Mahon/Documents/Research/CLAM/Phase3A_Baseline_Features/Train/TCGA_ims1` and similar directories.\n"
            "During training, the path passed as `--data_root_dir` may point to either a single cohort (e.g., `TCGA_ims1`) or a structure that includes `pt_files/` and/or `h5_files/`.\n"
            "If your code needs to load patch features for SSL or feature analysis, you may scan the `pt_files` subdirectory inside that `data_root_dir`.\n\n"
            "=== STRATEGY SPECIFICATION ===\n"
            f"{strategy_spec}\n\n"
            "=== ORIGINAL SCRIPT ===\n"
            f"{template}\n\n"
            "Now return the complete, modified script with only the necessary and relevant changes applied. Ensure the script will run end-to-end."
        )
    else:
        prompt = ("You are an expert debugging agent. Based on the following experiment strategy and the previous failed attempt, revise the provided Python training script to resolve the failure and complete the integration successfully. Your task is to correct the script, not regenerate it from scratch, while preserving all functional progress already made. Specific requirements:\n\n"
        "- Diagnose and resolve any issues that caused the previous run to fail, using the traceback or error context provided.\n"
        "- Maintain any working logic from the original script unless it directly contributes to the failure.\n"
        "- If needed, re-integrate or re-implement functions (e.g., custom losses, dataloaders, training routines) so they are fully operational in the training pipeline.\n"
        "- If any optional features (e.g., `use_ssl`, `use_dino`, `ssl_epochs`) are introduced:\n"
        "  • Add appropriate `argparse` arguments with default values\n"
        "  • Set booleans like `use_ssl` to `True` by default\n"
        "  • Ensure arguments are passed through to the training routine and/or dataset as needed\n"
        "- Do **not** include unused stubs, redundant definitions, or unused imports\n"
        "- You **may restructure logic**, but must preserve unrelated existing functionality\n"
        "- **Do not** import functions from the same file — reference them directly instead\n"
        "- Keep **all logic in one file** — do not split across modules\n\n"
        "You must also insert the following two lines directly after any `from __future__` imports (if present):\n"
        "import sys, os\n"
        "sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))\n\n"
        "=== CONTEXT ON FILE STRUCTURE AND DATA ACCESS ===\n"
        "Patch features are stored as .pt files in folders named `pt_files/`, and h5 feature files are in `h5_files/` folders. "
        "These are organized under:\n"
        "`C:/Users/Mahon/Documents/Research/CLAM/Phase3A_Baseline_Features/Train/TCGA_ims1` and similar directories.\n"
        "During training, the path passed as `--data_root_dir` may point to either a single cohort (e.g., `TCGA_ims1`) or a structure that includes `pt_files/` and/or `h5_files/`.\n"
        "If your code needs to load patch features for SSL or feature analysis, you may scan the `pt_files` subdirectory inside that `data_root_dir`.\n\n"
        "=== STRATEGY SPECIFICATION ===\n"
        f"{strategy_spec}\n\n"
        "=== FAILED SCRIPT ===\n"
        f"{template}\n\n"
        "Now return the complete, corrected version of the script. Focus only on fixing the relevant error while preserving valid changes from the prior version. Ensure the script runs end-to-end."
        )

    print("\n========== PROMPT SENT TO API ==========\n")
    print(prompt)  # Print first 5000 characters for brevity
    print("\n========== END PROMPT ==========\n")

    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert machine learning engineer and coder."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    raw_response = response.choices[0].message.content.strip()
    print("\n========== RAW API RESPONSE ==========\n")
    print(raw_response)  # Print first 5000 characters for brevity
    print("\n========== END RESPONSE ==========\n")

    return raw_response

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
    cmd = [
        "python", script_path,
        "--task", DEFAULT_TASK,
        "--data_root_dir", DEFAULT_DATA_ROOT,
        "--results_dir", DEFAULT_RESULTS_DIR,
        "--model_type", DEFAULT_MODEL_TYPE,
        "--embed_dim", DEFAULT_EMBED_DIM,
        "--k", DEFAULT_K,
        "--exp_code", "autogen_exp",
    ]

    log_path = get_log_filename(script_path)
    with open(log_path, "w") as logfile:
        process = subprocess.run(
            cmd,
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

def extract_traceback_from_log(log_path):
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()

        traceback_lines = []
        in_traceback = False

        for line in lines:
            if "Traceback" in line:
                in_traceback = True
                traceback_lines = [line]  # reset and start fresh
            elif in_traceback:
                traceback_lines.append(line)

        if traceback_lines:
            return "".join(traceback_lines).strip()

        return ""  # No traceback found
    except Exception as e:
        return f"Error reading log: {e}"

def retry_until_success(strategy_spec, max_retries=5):
    notes = ""
    last_traceback = ""
    source_path = MAIN_TEMPLATE_PATH  # start with template
    first_attempt = True
    for attempt in range(max_retries):
        first_attempt = (attempt == 0)
        print(f"[Coder Agent] Attempt {attempt + 1}...")
        script_name = generate_experiment_filename()
        script_path = os.path.join(EXPERIMENT_DIR, script_name)

        if last_traceback:
            full_strategy = (
                f"{strategy_spec}\n\n"
                f"=== ERROR FROM PREVIOUS ATTEMPT ===\n"
                f"{last_traceback.strip()}\n"
                f"Please correct this error in the next version."
            )
        else:
            full_strategy = strategy_spec

        modified_script = build_script_from_template(source_path, full_strategy, first_attempt)
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
            last_traceback = extract_traceback_from_log(log_path) 
            print(f"[Coder Agent] Detected error:\n{last_traceback[:500]}")
            source_path = script_path

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
