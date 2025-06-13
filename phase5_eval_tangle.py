from __future__ import print_function

import numpy as np
import argparse
import torch
import pdb
import os
import pandas as pd
from torch.utils.data import DataLoader

from utils.utils import *
from math import floor
from dataset_modules.dataset_generic_tangle import Generic_MIL_Dataset_Tangle

# --- MODIFIED: Import model and summary function directly ---
from models.model_clam_tangle import CLAM_MB_Tangle, CLAM_SB_Tangle
from utils.core_utils_tangle import summary


# --- Data Loader Collate Function ---
def tangle_collate(batch):
    features, labels, tangle_feats, slide_ids = zip(*batch)
    return list(features), torch.tensor(labels), list(tangle_feats), list(slide_ids)

# --- Evaluation Settings (No changes here) ---
parser = argparse.ArgumentParser(description='CLAM Tangle-Integrated Model Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='Data directory for patch features of evaluation cohorts.')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='Relative path to results folder containing models_exp_code.')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='Experiment code for saving evaluation results.')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='Experiment code of trained models (directory under results_dir).')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='Size of model (default: small).')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_mb', 
                    help='Type of model (default: clam_mb).')
parser.add_argument('--drop_out', type=float, default=0.25, help='Dropout rate for model.')
parser.add_argument('--k', type=int, default=3, help='Number of folds the model was trained on.')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'],
                    default='task_1_tumor_vs_normal')
parser.add_argument('--cohort', type=str, required=True,
                    help='Name of the evaluation cohort folder (e.g., Georgia).')
parser.add_argument('--cohort_labels', type=str, required=True,
                    help='Base name of the label CSV file for the cohort (e.g., Georgia_filtered).')
parser.add_argument('--use_tangle_concatenation', action='store_true', default=False,
                    help='Enable feature concatenation with Tangle embeddings.')
parser.add_argument('--tangle_feature_dir', type=str, default=None,
                    help='Directory containing Tangle embedding .npy files for the evaluation cohort.')
parser.add_argument('--tangle_embedding_dim', type=int, default=1024,
                    help='Dimension of the Tangle embeddings.')
parser.add_argument('--original_patch_feature_dim', type=int, default=1024,
                    help='Dimension of the original patch features.')
parser.add_argument('--embed_dim', type=int, default=1024, 
                    help='This will be calculated and should not be set manually when using Tangle features.')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Calculate effective embed_dim (No changes here) ---
if args.use_tangle_concatenation:
    calculated_embed_dim = args.original_patch_feature_dim + args.tangle_embedding_dim
    args.embed_dim = calculated_embed_dim
    print(f"Tangle concatenation enabled. Effective embed_dim for model: {args.embed_dim}")
else:
    args.embed_dim = args.original_patch_feature_dim
    print(f"Tangle concatenation disabled. Effective embed_dim for model: {args.embed_dim}")

# --- Setup Directories (No changes here) ---
args.save_dir = os.path.join('./eval_results', 'EVAL_TANGLE_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))
os.makedirs(args.save_dir, exist_ok=True)
assert os.path.isdir(args.models_dir), f"Models directory not found: {args.models_dir}"

# --- Log Settings (No changes here) ---
settings = {key: val for key, val in vars(args).items()}
with open(os.path.join(args.save_dir, f'eval_experiment_tangle_{args.save_exp_code}.txt'), 'w') as f:
    for key, val in settings.items():
        f.write(f"{key}: {val}\n")
print("\nEvaluation Settings:", settings)

# --- Dataset Instantiation (No changes here) ---
dataset_kwargs = {
    "csv_path": os.path.join('C:/Users/Mahon/Documents/Research/CLAM/Labels/Tangle/', args.cohort_labels + '.csv'),
    "data_dir": os.path.join(args.data_root_dir, args.cohort),
    "shuffle": False, 
    "print_info": True,
    "label_dict": {'normal_tissue': 0, 'tumor_tissue': 1},
    "patient_strat": False,
    "ignore": [],
    "tangle_feature_dir": args.tangle_feature_dir,
    "tangle_embedding_dim": args.tangle_embedding_dim
}

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes = 2
else:
    raise NotImplementedError(f"Task {args.task} not implemented.")

dataset = Generic_MIL_Dataset_Tangle(**dataset_kwargs)

# --- Evaluation Loop ---
if __name__ == "__main__":
    all_auc = []
    all_acc = []
    folds = range(args.k)
    
    collate_fn = tangle_collate if args.use_tangle_concatenation else None

    for fold in folds:
        # --- MODIFICATION: Replaced `eval_utils.eval()` with direct implementation ---

        # 1. Instantiate the Tangle-aware model
        print(f"\n--- Evaluating Fold: {fold} ---")
        model_dict = {"dropout": args.drop_out, "n_classes": args.n_classes, "embed_dim": args.embed_dim}
        if args.model_type == 'clam_mb':
            model = CLAM_MB_Tangle(**model_dict)
        elif args.model_type == 'clam_sb':
            model = CLAM_SB_Tangle(**model_dict)
        else:
            raise NotImplementedError
        
        # 2. Load the checkpoint weights
        ckpt_path = os.path.join(args.models_dir, f's_{fold}_checkpoint.pt')
        if not os.path.isfile(ckpt_path):
            print(f"Checkpoint not found for fold {fold}, skipping: {ckpt_path}")
            all_auc.append(float('nan'))
            all_acc.append(float('nan'))
            continue
            
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)

        # 3. Create the Tangle-aware DataLoader
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        # 4. Run inference using the summary function from core_utils_tangle
        patient_results, test_error, auc, acc_logger = summary(model, loader, args.n_classes, args.use_tangle_concatenation)
        
        # 5. Create the results DataFrame
        df = pd.DataFrame(patient_results).transpose()
        
        # --- End of MODIFICATION ---

        all_auc.append(auc)
        all_acc.append(1 - test_error)
        df.to_csv(os.path.join(args.save_dir, f'fold_{fold}_results.csv'), index=False)

    # --- Summarize Results (No changes here) ---
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    summary_path = os.path.join(args.save_dir, 'summary.csv')
    final_df.to_csv(summary_path, index=False)
    print(f"\nEvaluation summary saved to {summary_path}")