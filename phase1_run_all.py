import os
import subprocess
import sys
import shutil

# --- Configuration ---

# Path to the Python interpreter you want to use
PYTHON_EXECUTABLE = sys.executable

# 1. Path to the CLAM create_patches.py script (the "old" one that saves image data in H5)
SCRIPT_TO_RUN = "create_patches_fp.py"  # Assumes it's in the same dir or in PATH

# 2. Base directory where your source WSI datasets are located
BASE_SOURCE_WSI_DIR = "C:/Users/Mahon/OneDrive/Documents/CLAM/Datasets/"

# 3. Base directory where all outputs from this phase will be saved.
#    A subdirectory will be created for each dataset.
BASE_MAIN_OUTPUT_DIR = "C:/Users/Mahon/OneDrive/Documents/CLAM/Patches/Phase1/"

# 4. Path to your CLAM preset CSV file
PRESET_CSV_PATH = "C:/Users/Mahon/OneDrive/Documents/CLAM/presets/tcga.csv"

# 5. Common patching parameters for all datasets
PATCH_SIZE = 256
STEP_SIZE = 256
PATCH_LEVEL = 0
# Flags for create_patches.py (these enable the desired actions)
DO_SEGMENTATION = True  # Corresponds to --seg
DO_PATCHING = True      # Corresponds to --patch (this is crucial for H5 with imgs)
DO_STITCHING = False    # Corresponds to --stitch (optional, set to True if you want stitches)

# auto_skip is True by default in create_patches.py (skips if .h5 exists)
# To force re-processing, add "--no_auto_skip" to the command list below.

# 6. Define your datasets
# Each entry is a tuple:
# (relative_path_to_wsi_from_BASE_SOURCE_WSI_DIR, unique_dataset_identifier_for_output)
# The unique_dataset_identifier will be used to create a subfolder in BASE_MAIN_OUTPUT_DIR
DATASETS_TO_PROCESS = [
    ("Train/TCGA_ims1", "Train/TCGA_ims1"),
    # ("Train/TCGA_ims2", "Train/TCGA_ims2"),
    # ("Evals/Carolina", "Evals/Carolina"),
    # ("Evals/Fondaz", "Evals/Fondaz"),
    # ("Evals/Georgia", "Evals/Georgia"),
    # ("Evals/Milan_Fondaz", "Evals/Milan_Fondaz"),
    # ("Evals/Minnesota", "Evals/Minnesota"),
    # ("Evals/Ohio", "Evals/Ohio"),
    # ("Evals/UNSW", "Evals/UNSW"),
    ("Evals/UTAH", "Evals/UTAH")
]

# --- End Configuration ---

def run_script_for_dataset(dataset_info):
    relative_wsi_path, dataset_identifier = dataset_info

    source_wsi_dir = os.path.join(BASE_SOURCE_WSI_DIR, relative_wsi_path)
    # This will be the 'save_dir' for create_patches.py
    # It will create 'patches', 'masks', 'stitches' subdirs inside this
    dataset_output_basedir = os.path.join(BASE_MAIN_OUTPUT_DIR, dataset_identifier)

    print(f"\n--- Processing Dataset Identifier: {dataset_identifier} ---")
    print(f"Source WSIs: {source_wsi_dir}")
    print(f"Target Main Output Directory (save_dir for script): {dataset_output_basedir}")

    if not os.path.isdir(source_wsi_dir):
        print(f"WARNING: Source WSI directory not found: {source_wsi_dir}. Skipping.")
        return False

    # create_patches.py creates subdirs like 'patches', 'masks' inside its 'save_dir'
    # So, we just need to ensure dataset_output_basedir exists.
    os.makedirs(dataset_output_basedir, exist_ok=True)

    command = [
        PYTHON_EXECUTABLE,
        SCRIPT_TO_RUN,
        "--source", source_wsi_dir,
        "--save_dir", dataset_output_basedir, # This is where 'patches', 'masks' will be created by the script
        "--patch_size", str(PATCH_SIZE),
        "--step_size", str(STEP_SIZE),
        "--patch_level", str(PATCH_LEVEL)
    ]

    if PRESET_CSV_PATH and os.path.exists(PRESET_CSV_PATH):
        command.extend(["--preset", PRESET_CSV_PATH])
    elif PRESET_CSV_PATH:
        print(f"WARNING: Preset CSV specified but not found: {PRESET_CSV_PATH}")
        print("The patch extraction script will use its default parameters.")

    if DO_SEGMENTATION:
        command.append("--seg")
    if DO_PATCHING:
        command.append("--patch") # This is crucial for getting H5 with 'imgs'
    if DO_STITCHING:
        command.append("--stitch")
    
    # To force re-processing (disable auto_skip), add: command.append("--no_auto_skip")

    print("\nExecuting command:")
    # Correctly quote paths with spaces for display, subprocess handles it internally
    display_command = []
    for arg in command:
        if " " in arg and not (arg.startswith('"') and arg.endswith('"')):
            display_command.append(f'"{arg}"')
        else:
            display_command.append(arg)
    print(" ".join(display_command))


    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("\nSTDOUT from create_patches.py:")
        print(process.stdout)
        if process.stderr: # Should be empty on pure success from the script
            print("\nSTDERR from create_patches.py:")
            print(process.stderr)
        print(f"Successfully processed dataset: {dataset_identifier}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR processing dataset: {dataset_identifier}")
        print(f"Return code: {e.returncode}")
        print("\nSTDOUT from create_patches.py:")
        print(e.stdout)
        print("\nSTDERR from create_patches.py:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"ERROR: Script to run '{SCRIPT_TO_RUN}' or Python executable '{PYTHON_EXECUTABLE}' not found.")
        return False
    except Exception as e_global:
        print(f"An unexpected error occurred while processing {dataset_identifier}: {e_global}")
        return False

if __name__ == "__main__":
    if not os.path.exists(SCRIPT_TO_RUN):
        # Try to find it in common locations if it's not in current dir or PATH
        potential_script_path = os.path.join(os.path.dirname(__file__), SCRIPT_TO_RUN)
        if os.path.exists(potential_script_path):
            SCRIPT_TO_RUN = potential_script_path
            print(f"Found script at: {SCRIPT_TO_RUN}")
        else:
            print(f"ERROR: The main script '{SCRIPT_TO_RUN}' was not found. Please ensure the path is correct.")
            exit()
    
    # Check if preset file path is relative and adjust
    if PRESET_CSV_PATH and not os.path.isabs(PRESET_CSV_PATH) and not os.path.exists(PRESET_CSV_PATH):
        potential_preset_path = os.path.join(os.path.dirname(__file__), PRESET_CSV_PATH)
        if os.path.exists(potential_preset_path):
            PRESET_CSV_PATH = potential_preset_path
            print(f"Adjusted preset path to: {PRESET_CSV_PATH}")
        else:
            # Check if it's just a filename and CLAM's structure puts it in ./presets/
            clams_preset_dir_path = os.path.join(os.path.dirname(SCRIPT_TO_RUN), "presets", PRESET_CSV_PATH)
            if os.path.exists(clams_preset_dir_path):
                 PRESET_CSV_PATH = clams_preset_dir_path
                 print(f"Found preset in CLAM's presets directory: {PRESET_CSV_PATH}")
            else:
                 print(f"Warning: Preset CSV '{PRESET_CSV_PATH}' not found directly or in typical relative locations. create_patches.py might use defaults.")


    total_datasets = len(DATASETS_TO_PROCESS)
    successful_runs = 0
    failed_datasets = []

    for i, dataset_info in enumerate(DATASETS_TO_PROCESS):
        print(f"\n================================================================================")
        print(f"Batch Script: Starting dataset {i+1}/{total_datasets}: {dataset_info[1]}")
        print(f"================================================================================")
        
        if run_script_for_dataset(dataset_info):
            successful_runs += 1
        else:
            failed_datasets.append(dataset_info[1])
        
        print(f"Batch Script: Finished processing dataset {i+1}/{total_datasets}: {dataset_info[1]}")

    print("\n\n--- Batch Processing Summary ---")
    print(f"Total datasets attempted: {total_datasets}")
    print(f"Successfully processed: {successful_runs}")
    print(f"Failed datasets: {len(failed_datasets)}")
    if failed_datasets:
        print("List of failed datasets:")
        for ds_name in failed_datasets:
            print(f"  - {ds_name}")
    print("Batch processing finished.")