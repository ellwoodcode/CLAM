from __future__ import print_function

import numpy as np
import argparse
import torch
# import torch.nn as nn # Not directly used here, but likely in eval_utils
import pdb
import os
import pandas as pd
from utils.utils import * # Assuming this has general utilities
from math import floor
# import matplotlib.pyplot as plt # Not used in the provided snippet
from dataset_modules.dataset_generic_rna import Generic_MIL_Dataset_RNA # UPDATED
# import h5py # Not directly used here
from utils.eval_utils import * # This is assumed to contain the main eval function

# Evaluation settings
parser = argparse.ArgumentParser(description='CLAM RNA Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory for patch features')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder containing models_exp_code')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code of trained models (directory under results_dir)')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory (if custom)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout rate for model') # From main.py

parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: 0, from original main.py logic)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: k, from original main.py logic)')
parser.add_argument('--fold', type=int, default=-1, help='specific single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_average for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'],
                    default='task_1_tumor_vs_normal') # Added a default like in main.py

# Arguments for batch processing from original eval.py
parser.add_argument('--cohort', type=str, required=True,
                    help='Name of the cohort folder (e.g., Georgia)')
parser.add_argument('--cohort_labels', type=str, required=True,
                    help='Base name of the label CSV file (e.g., Georgia_filtered)')

# New arguments for RNA concatenation
parser.add_argument('--use_rna_concatenation', action='store_true', default=False,
                    help='Enable feature concatenation with RNA-seq data.')
parser.add_argument('--rna_data_base_dir', type=str, default=None,
                    help='Base directory for processed RNA NPY files.')
parser.add_argument('--master_rna_dim', type=int, default=19944,
                    help='Dimension of the RNA vectors.')
parser.add_argument('--original_patch_feature_dim', type=int, default=1024,
                    help='Dimension of original patch features.')
parser.add_argument('--debug_rna_paths', action='store_true', default=False, 
                    help='Print RNA file paths for debugging.')
# embed_dim will be calculated or taken if provided, ensure it's the concatenated one if RNA is used
parser.add_argument('--embed_dim', type=int, default=1024, 
                    help='Feature embedding dimension. If use_rna_concatenation, this should be sum of patch+RNA dims, or will be calculated.')


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calculate effective embed_dim for the models if RNA is used
if args.use_rna_concatenation:
    calculated_embed_dim = args.original_patch_feature_dim + args.master_rna_dim
    if args.embed_dim != calculated_embed_dim and args.embed_dim != 1024: # 1024 is default, might not have been changed by user
         print(f"Warning: Provided --embed_dim {args.embed_dim} does not match calculated concatenated dim {calculated_embed_dim}. Using calculated dim.")
    args.embed_dim = calculated_embed_dim
    print(f"RNA concatenation enabled. Effective embed_dim for model: {args.embed_dim}")
else:
    # If not using RNA, embed_dim should ideally be original_patch_feature_dim
    if args.embed_dim != args.original_patch_feature_dim:
        # print(f"Warning: --embed_dim {args.embed_dim} provided, but not using RNA. Ensure this matches original_patch_feature_dim if that's intended.")
        # It will use the provided args.embed_dim or its default (1024)
        pass # Keep args.embed_dim as is, or set to original_patch_feature_dim
    args.embed_dim = args.original_patch_feature_dim # Ensuring it's based on patch features if no RNA
    print(f"RNA concatenation disabled. Effective embed_dim for model: {args.embed_dim}")


args.save_dir = os.path.join('./eval_results', 'EVAL_RNA_' + str(args.save_exp_code)) # Added _RNA_
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir), f"Models directory not found: {args.models_dir}"
assert os.path.isdir(args.splits_dir), f"Splits directory not found: {args.splits_dir}"

settings = {
    'task': args.task,
    'split': args.split,
    'save_dir': args.save_dir, 
    'models_dir': args.models_dir,
    'model_type': args.model_type,
    'model_size': args.model_size,
    'drop_out': args.drop_out,
    'cohort': args.cohort,
    'cohort_labels': args.cohort_labels,
    'use_rna_concatenation': args.use_rna_concatenation,
    'rna_data_base_dir': args.rna_data_base_dir,
    'master_rna_dim': args.master_rna_dim,
    'original_patch_feature_dim': args.original_patch_feature_dim,
    'calculated_embed_dim': args.embed_dim # This is the crucial dim for the model
}

with open(os.path.join(args.save_dir, f'eval_experiment_rna_{args.save_exp_code}.txt'), 'w') as f:
    for key, val in settings.items():
        f.write(f"{key}: {val}\n")

print("Evaluation Settings:")
for key, val in settings.items():
    print(f"{key}: {val}")

# Dataset Instantiation
dataset_kwargs = {
    "csv_path": os.path.join('C:/Users/Mahon/Documents/Research/CLAM/Labels/Tangle/', args.cohort_labels + '_cohort.csv'),
    "data_dir": os.path.join(args.data_root_dir, args.cohort), # This is for patch features
    "shuffle": False,
    "print_info": True,
    "patient_strat": False,
    "ignore": [],
    # RNA-specific arguments for Generic_MIL_Dataset_RNA
    "use_rna_concatenation": args.use_rna_concatenation,
    "rna_data_base_dir": args.rna_data_base_dir,
    "master_rna_dim": args.master_rna_dim,
    "original_patch_feature_dim": args.original_patch_feature_dim,
    "debug_rna_paths": args.debug_rna_paths,
}

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes = 2
    dataset_kwargs["label_dict"] = {'normal_tissue': 0, 'tumor_tissue': 1}
elif args.task == 'task_2_tumor_subtyping': # From original eval.py
    args.n_classes = 3
    # Update paths and dicts as per your actual subtyping task setup
    dataset_kwargs["csv_path"] = 'dataset_csv/tumor_subtyping_dummy_clean.csv' # Placeholder
    dataset_kwargs["data_dir"] = os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features') # Placeholder
    dataset_kwargs["label_dict"] = {'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2}
else:
    raise NotImplementedError(f"Task {args.task} not implemented in eval_rna.py")

dataset = Generic_MIL_Dataset_RNA(**dataset_kwargs) # UPDATED dataset class

# Determine folds to evaluate
if args.k_start == -1: start = 0
else: start = args.k_start
if args.k_end == -1: end = args.k
else: end = args.k_end

if args.fold == -1:
    folds_to_eval = range(start, end)
else:
    folds_to_eval = range(args.fold, args.fold + 1)

ckpt_paths = [os.path.join(args.models_dir, f's_{fold}_checkpoint.pt') for fold in folds_to_eval]
# datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1} # Not used in snippet

if __name__ == "__main__":
    all_results_list = [] # Renamed to avoid conflict with outer scope if any
    all_auc_list = []
    all_acc_list = []

    for ckpt_idx, fold_num in enumerate(folds_to_eval):
        current_ckpt_path = ckpt_paths[ckpt_idx]
        if not os.path.exists(current_ckpt_path):
            print(f"Checkpoint not found: {current_ckpt_path}. Skipping fold {fold_num}.")
            all_auc_list.append(float('nan')) # Or handle as desired
            all_acc_list.append(float('nan'))
            # all_results_list.append({}) 
            continue
        
        print(f"\nEvaluating Fold: {fold_num} with checkpoint: {current_ckpt_path}")
        # The dataset object is already filtered by args.split (implicitly by Generic_MIL_Dataset)
        # Or, if eval_utils.eval expects a specific split from the main dataset:
        # train_split, val_split, test_split = dataset.return_splits(from_id=False, csv_path=os.path.join(args.splits_dir, f'splits_{fold_num}.csv'))
        # if args.split == 'val': split_dataset_obj = val_split
        # elif args.split == 'test': split_dataset_obj = test_split
        # else: split_dataset_obj = dataset # Fallback or handle 'all', 'train'
        
        # Assuming 'dataset' is already the data for the desired split (e.g. test set for the cohort)
        # The eval function from eval_utils will handle model loading and prediction
        # It needs to be aware of args.use_rna_concatenation and use args.embed_dim correctly
        model, patient_results, test_error, auc, df = eval(dataset, args, current_ckpt_path) # Pass full dataset for the cohort
        
        # all_results_list.append(patient_results) # Assuming patient_results is the detailed dict
        all_auc_list.append(auc)
        all_acc_list.append(1 - test_error)
        df.to_csv(os.path.join(args.save_dir, f'fold_{fold_num}_results.csv'), index=False)

    final_df_data = {'folds': list(folds_to_eval), 'test_auc': all_auc_list, 'test_acc': all_acc_list}
    final_df = pd.DataFrame(final_df_data)
    
    if len(folds_to_eval) != args.k and args.fold == -1 : # Only if evaluating multiple folds, not just one
        save_name = f'summary_partial_{folds_to_eval[0]}_{folds_to_eval[-1]}.csv'
    elif args.fold != -1:
         save_name = f'summary_fold_{args.fold}.csv'
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name), index=False)
    print(f"\nEvaluation summary saved to {os.path.join(args.save_dir, save_name)}")