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
