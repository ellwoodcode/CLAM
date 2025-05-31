import os
import subprocess
import sys
import glob
import pandas as pd # Still useful for sanity checking CSVs if needed, but not for creating them

# --- Configuration (Incorporating Your Paths and Label CSVs) ---

PYTHON_EXECUTABLE = sys.executable
SCRIPT_TO_RUN = "extract_features_fp.py" # CLAM's feature extraction script

BASE_SOURCE_WSI_DIR = "C:/Users/Mahon/OneDrive/Documents/CLAM/Datasets/"
BASE_H5_COORD_DIR = "C:/Users/Mahon/OneDrive/Documents/CLAM/Patches/Phase1/" # From your Phase 1
BASE_FEAT_OUTPUT_DIR = "C:/Users/Mahon/OneDrive/Documents/CLAM/Phase3A_Baseline_Features/"
BASE_LABEL_CSV_DIR = "C:/Users/Mahon/OneDrive/Documents/CLAM/Labels/Boolean/"

MODEL_NAME_FOR_FEATURES = "resnet50_trunc"
FEATURE_EXTRACTION_BATCH_SIZE = 512
SLIDE_EXTENSION = ".svs"
TARGET_PATCH_SIZE_FOR_EXTRACTOR = 224
# auto_skip is True by default in extract_features_fp.py (skips if .pt output file exists)
# To force re-processing, add "--no_auto_skip" to the command list.

# Define your datasets
# Each entry: (
#   relative_path_to_wsi_from_BASE_SOURCE_WSI_DIR,
#   relative_path_to_h5_coords_from_BASE_H5_COORD_DIR,
#   unique_dataset_identifier_for_output_subdir,
#   filename_of_label_csv_in_BASE_LABEL_CSV_DIR  <-- NEW
# )

DATASETS_TO_PROCESS = [
    ("Train/TCGA_ims1", "Train/TCGA_ims1", "Train/TCGA_ims1", "labels_ims1.csv"),  
    # ("Train/TCGA_ims2", "Train/TCGA_ims2", "Train/TCGA_ims2", "labels_ims2.csv"),  
    # ("Evals/Carolina",  "Evals/Carolina",  "Evals/Carolina",  "N.Carolina.csv"),    
    # ("Evals/Fondaz",    "Evals/Fondaz",    "Evals/Fondaz",    "Lombardy-Italy.csv"),      
    # ("Evals/Georgia",   "Evals/Georgia",   "Evals/Georgia",   "Georgia.csv"),     
    # ("Evals/Milan_Fondaz", "Evals/Milan_Fondaz", "Evals/Milan_Fondaz", "Milan-Italy.csv"),  
    # ("Evals/Minnesota", "Evals/Minnesota", "Evals/Minnesota", "Minnesota.csv"), 
    # ("Evals/Ohio",      "Evals/Ohio",      "Evals/Ohio",      "Ohio.csv"),        
    # ("Evals/UNSW",      "Evals/UNSW",      "Evals/UNSW",      "UNSW.csv"),        
    ("Evals/UTAH",      "Evals/UTAH",      "Evals/UTAH",      "UTAH.csv")         
]
# --- End Configuration ---

def run_feature_extraction_for_dataset(dataset_info):
    relative_wsi_path, relative_h5_coord_path, output_subdir_identifier, label_csv_filename = dataset_info

    source_wsi_dataset_dir = os.path.join(BASE_SOURCE_WSI_DIR, relative_wsi_path)
    h5_coord_input_dir_for_dataset = os.path.join(BASE_H5_COORD_DIR, relative_h5_coord_path)
    feature_output_basedir_for_dataset = os.path.join(BASE_FEAT_OUTPUT_DIR, output_subdir_identifier)
    full_label_csv_path = os.path.join(BASE_LABEL_CSV_DIR, label_csv_filename)

    print(f"\n--- Processing Dataset for Feature Extraction: {output_subdir_identifier} ---")
    print(f"Source WSIs: {source_wsi_dataset_dir}")
    print(f"Input H5 Coords from: {h5_coord_input_dir_for_dataset}")
    print(f"Using Label CSV for slide list: {full_label_csv_path}")
    print(f"Output Features To (base): {feature_output_basedir_for_dataset}")

    if not os.path.isdir(source_wsi_dataset_dir):
        print(f"WARNING: Source WSI directory not found: {source_wsi_dataset_dir}. Skipping.", file=sys.stderr)
        return False
    if not os.path.isdir(h5_coord_input_dir_for_dataset):
        print(f"WARNING: Input H5 coordinate directory not found: {h5_coord_input_dir_for_dataset}. Skipping.", file=sys.stderr)
        return False
    if not os.path.exists(full_label_csv_path):
        print(f"ERROR: Label CSV not found: {full_label_csv_path} for dataset {output_subdir_identifier}. Skipping.", file=sys.stderr)
        return False

    os.makedirs(feature_output_basedir_for_dataset, exist_ok=True)

    command = [
        PYTHON_EXECUTABLE,
        SCRIPT_TO_RUN,
        "--data_h5_dir", h5_coord_input_dir_for_dataset,
        "--data_slide_dir", source_wsi_dataset_dir,
        "--csv_path", full_label_csv_path, # Using your specified label CSV
        "--feat_dir", feature_output_basedir_for_dataset,
        "--batch_size", str(FEATURE_EXTRACTION_BATCH_SIZE),
        "--slide_ext", SLIDE_EXTENSION
    ]
    # If you want to force re-processing (disable auto_skip):
    # command.append("--no_auto_skip")

    print("\nExecuting command:")
    display_command = [f'"{arg}"' if " " in arg and not (arg.startswith('"') and arg.endswith('"')) else arg for arg in command]
    print(" ".join(display_command))

    success = False
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        print("\nSTDOUT from extract_features_fp.py:")
        print(process.stdout)
        if process.stderr:
            print("\nSTDERR from extract_features_fp.py:")
            print(process.stderr)
        print(f"Successfully extracted features for dataset: {output_subdir_identifier}")
        success = True
    except subprocess.CalledProcessError as e:
        print(f"ERROR extracting features for dataset: {output_subdir_identifier}", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        print("\nSTDOUT from extract_features_fp.py:", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("\nSTDERR from extract_features_fp.py:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
    except FileNotFoundError:
        print(f"ERROR: Script to run '{SCRIPT_TO_RUN}' or Python executable '{PYTHON_EXECUTABLE}' not found.", file=sys.stderr)
    except Exception as e_global:
        print(f"An unexpected error occurred while processing {output_subdir_identifier}: {e_global}", file=sys.stderr)
    
    return success

if __name__ == "__main__":
    script_path_to_check = SCRIPT_TO_RUN
    if not os.path.exists(script_path_to_check) and not shutil.which(script_path_to_check):
        potential_script_path = os.path.join(os.path.dirname(__file__), SCRIPT_TO_RUN)
        if os.path.exists(potential_script_path):
            SCRIPT_TO_RUN = potential_script_path
            print(f"Found feature extraction script at: {SCRIPT_TO_RUN}")
        else:
            print(f"ERROR: The feature extraction script '{SCRIPT_TO_RUN}' was not found. Please ensure the path is correct or it's in your system PATH.")
            exit()

    total_datasets = len(DATASETS_TO_PROCESS)
    successful_runs = 0
    failed_datasets = []

    print("Starting batch feature extraction for baseline CLAM using provided label CSVs...")
    for i, dataset_info in enumerate(DATASETS_TO_PROCESS):
        print(f"\n================================================================================")
        print(f"Batch Script: Starting feature extraction for dataset {i+1}/{total_datasets}: {dataset_info[2]}") # dataset_info[2] is output_subdir_identifier
        print(f"================================================================================")
        
        if run_feature_extraction_for_dataset(dataset_info):
            successful_runs += 1
        else:
            failed_datasets.append(dataset_info[2]) # Log based on output_subdir_identifier
        
        print(f"Batch Script: Finished feature extraction for dataset {i+1}/{total_datasets}: {dataset_info[2]}")

    print("\n\n--- Feature Extraction Batch Processing Summary ---")
    print(f"Total datasets attempted: {total_datasets}")
    print(f"Successfully processed: {successful_runs}")
    print(f"Failed datasets: {len(failed_datasets)}")
    if failed_datasets:
        print("List of failed datasets:")
        for ds_name in failed_datasets:
            print(f"  - {ds_name}")
    print("Batch feature extraction finished.")