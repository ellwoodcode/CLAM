from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl # Assuming these utils are general
from utils.utils import * # Assuming these utils are general (seed_torch, etc.)
# Import RNA-specific core_utils and dataset
from utils.core_utils_rna import train_rna # Renamed train to train_rna
from dataset_modules.dataset_generic_rna import Generic_MIL_Dataset_RNA # RNA Dataset

# pytorch imports
import torch
# DataLoader, sampler, nn, F might be used by utils or core_utils
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, sampler


import pandas as pd
import numpy as np

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)

    # Determine embed_dim based on RNA concatenation
    if args.use_rna_concatenation:
        args.embed_dim = args.original_patch_feature_dim + args.master_rna_dim
        print(f"Using RNA concatenation. Original Patch Dim: {args.original_patch_feature_dim}, RNA Dim: {args.master_rna_dim}, New Embed Dim: {args.embed_dim}")
    else:
        args.embed_dim = args.original_patch_feature_dim # Or use a specific arg for this if original was different from a fixed 1024
        print(f"Not using RNA concatenation. Embed Dim: {args.embed_dim}")


    # Dataset loading (moved outside the loop as it's loaded once)
    print('\nLoad Dataset')
    if args.task == 'task_1_tumor_vs_normal':
        args.n_classes = 2
        dataset = Generic_MIL_Dataset_RNA(
            csv_path='C:/Users/Mahon/Documents/Research/CLAM/Labels/Tangle/labels_ims1_filtered_cohort.csv', # Ensure this CSV has 'cohort' column if use_rna_concatenation
            data_dir=args.data_root_dir, # Patch features
            shuffle=False, # Typically, shuffling is done by DataLoader per epoch for training set
            seed=args.seed,
            print_info=True,
            label_dict={'normal_tissue': 0, 'tumor_tissue': 1},
            patient_strat=False, # Or based on args if needed
            ignore=[],
            # RNA specific args
            rna_data_base_dir=args.rna_data_base_dir,
            use_rna_concatenation=args.use_rna_concatenation,
            master_rna_dim=args.master_rna_dim,
            original_patch_feature_dim=args.original_patch_feature_dim,
            debug_rna_paths=args.debug_rna_paths if hasattr(args, 'debug_rna_paths') else False
        )
    elif args.task == 'task_2_tumor_subtyping':
        args.n_classes = 3
        dataset = Generic_MIL_Dataset_RNA(
            csv_path='dataset_csv/tumor_subtyping_dummy_clean.csv', # Ensure this CSV has 'cohort' column if use_rna_concatenation
            data_dir=os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'), # Patch features
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict={'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2},
            patient_strat=False, # Or based on args
            ignore=[],
            # RNA specific args
            rna_data_base_dir=args.rna_data_base_dir,
            use_rna_concatenation=args.use_rna_concatenation,
            master_rna_dim=args.master_rna_dim,
            original_patch_feature_dim=args.original_patch_feature_dim,
            debug_rna_paths=args.debug_rna_paths if hasattr(args, 'debug_rna_paths') else False
        )
        if args.model_type in ['clam_sb', 'clam_mb']: # This check was in original main
             assert args.subtyping, "Subtyping must be true for task_2 with CLAM models"

    else:
        raise NotImplementedError

    # This was outside the loop in original, seems fine.
    # `dataset` object is now `Generic_MIL_Dataset_RNA`
    # Its `return_splits` method will return splits of `Generic_Split_RNA`
    # which internally handle the feature loading and concatenation.


    for i in folds:
        seed_torch(args.seed) # Seed for each fold for reproducibility
        
        # The dataset object is already initialized. We just get splits.
        # The csv_path for splits is specific to the fold.
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        print(f"\nFold {i}:")
        if train_dataset: print(f"Train dataset embed_dim (from dataset obj): {train_dataset.concatenated_feature_dim if hasattr(train_dataset, 'concatenated_feature_dim') else 'N/A'}")
        
        datasets = (train_dataset, val_dataset, test_dataset)
        # Pass the full args to train_rna, which will use args.embed_dim
        results, test_auc, val_auc, test_acc, val_acc  = train_rna(datasets, i, args)
        
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc,
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# --- Main parsing and setup ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configurations for WSI Training with RNA Concatenation')
    # General arguments from original
    parser.add_argument('--data_root_dir', type=str, default=None, help='data directory for patch features')
    # embed_dim will be calculated based on other args, so removed from direct parsing or made optional
    # parser.add_argument('--embed_dim', type=int, default=1024) # This will be set based on concatenation
    parser.add_argument('--max_epochs', type=int, default=200, help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
    parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, implies 0)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, implies k)')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument('--split_dir', type=str, default=None, help='manually specify the set of splits to use')
    parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
    parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
    parser.add_argument('--drop_out', type=float, default=0.25, help='dropout rate for model (default: 0.25)') # Clarified help
    parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce', help='slide-level classification loss function (default: ce)')
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', help='type of model (default: clam_sb)')
    parser.add_argument('--exp_code', type=str, default='experiment_rna', help='experiment code for saving results (default: experiment_rna)') # Default changed
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil type if not used by it')
    parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'], default='task_1_tumor_vs_normal') # Default added

    # CLAM specific options (from original)
    parser.add_argument('--no_inst_cluster', action='store_true', default=False, help='disable instance-level clustering')
    parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', 'None'], default='None', help='instance-level clustering loss function (default: None)') # Made None a string choice
    parser.add_argument('--subtyping', action='store_true', default=False, help='subtyping problem (typically True for task_2)')
    parser.add_argument('--bag_weight', type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
    parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam instance loss')

    # New arguments for RNA concatenation
    parser.add_argument('--use_rna_concatenation', action='store_true', default=False, help='Enable feature concatenation with RNA-seq data.')
    parser.add_argument('--rna_data_base_dir', type=str, default=None, help='Base directory for processed RNA NPY files (e.g., /path/to/rna_data).')
    parser.add_argument('--master_rna_dim', type=int, default=19944, help='Dimension of the RNA vectors (default: 19944).')
    parser.add_argument('--original_patch_feature_dim', type=int, default=1024, help='Dimension of original patch features (e.g., ResNet50 output, default: 1024).')
    # This will be calculated, not parsed directly for the model, but needed for dataset
    # parser.add_argument('--feat_dim', type=int, help='Concatenated feature dimension, set automatically if use_rna_concatenation is true.')
    parser.add_argument('--debug_rna_paths', action='store_true', default=False, help='Print RNA file paths for debugging.')


    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Handle inst_loss 'None' string
    if args.inst_loss == 'None':
        args.inst_loss = None
        
    # Seed everything
    seed_torch(args.seed)

    # Calculate effective embed_dim for the models (used in core_utils_rna.py)
    if args.use_rna_concatenation:
        args.embed_dim = args.original_patch_feature_dim + args.master_rna_dim
    else:
        args.embed_dim = args.original_patch_feature_dim


    # Settings dictionary for logging (similar to original)
    settings = {'num_splits': args.k,
                'k_start': args.k_start,
                'k_end': args.k_end,
                'task': args.task,
                'max_epochs': args.max_epochs,
                'results_dir': args.results_dir,
                'lr': args.lr,
                'experiment': args.exp_code,
                'reg': args.reg,
                'label_frac': args.label_frac,
                'bag_loss': args.bag_loss,
                'inst_loss': args.inst_loss, # Log the actual value
                'seed': args.seed,
                'model_type': args.model_type,
                'model_size': args.model_size,
                "drop_out": args.drop_out, # Use consistent naming with arg
                'weighted_sample': args.weighted_sample,
                'opt': args.opt,
                'use_rna_concatenation': args.use_rna_concatenation, # New
                'rna_data_base_dir': args.rna_data_base_dir,       # New
                'master_rna_dim': args.master_rna_dim,             # New
                'original_patch_feature_dim': args.original_patch_feature_dim, # New
                'calculated_embed_dim': args.embed_dim             # New (effective dim for model)
                }

    if args.model_type in ['clam_sb', 'clam_mb']:
       settings.update({'bag_weight': args.bag_weight,
                        'B': args.B, # num patches for instance loss
                        'subtyping': args.subtyping,
                        'no_inst_cluster': args.no_inst_cluster})


    print('\nLoad Dataset section moved before the fold loop.')

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.split_dir is None:
        args.split_dir = os.path.join('splits', args.task + '_{}'.format(int(args.label_frac * 100)))
    else:
        # If a custom split_dir is provided, it might be an absolute path or relative to 'splits'
        # For simplicity, let's assume it's a full path or relative to the current working directory if not under 'splits/'
        if not os.path.isabs(args.split_dir) and not args.split_dir.startswith('splits/'):
             args.split_dir = os.path.join('splits', args.split_dir)


    print('Split dir: ', args.split_dir)
    if not os.path.isdir(args.split_dir):
      raise FileNotFoundError(f"Split directory {args.split_dir} not found.")


    settings.update({'split_dir': args.split_dir})

    with open(os.path.join(args.results_dir, 'experiment_{}.txt'.format(args.exp_code)), 'w') as f: # Use os.path.join
        print(settings, file=f)
    # f.close() # with open handles closing

    print("\n################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))

    results = main(args) # Call the main function that contains the k-fold loop
    
    print("finished!")
    print("end script")