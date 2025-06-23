from __future__ import print_function

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_modules.dataset_generic_tangle import Generic_MIL_Dataset_Tangle
from models.model_clam_tangle import CLAM_SB_Tangle, CLAM_MB_Tangle
from utils.core_utils_tangle import summary

# ---------------------------------------------
#  Collate function to handle tangle embeddings
# ---------------------------------------------
def tangle_collate(batch):
    patch_feats, labels, tangle_feats, slide_ids = zip(*batch)
    return list(patch_feats), torch.tensor(labels), list(tangle_feats), list(slide_ids)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a CLAM model trained with Tangle cross-modal fusion')

    parser.add_argument('--data_root_dir', type=str, default='PATH_TO_PATCH_FEATURES',
                        help='Root directory containing patch features for the evaluation cohort.')
    parser.add_argument('--tangle_dir', type=str, default='PATH_TO_TANGLE_EMBEDDINGS',
                        help='Base directory containing Tangle embedding files for each cohort.')
    parser.add_argument('--label_dir', type=str, default='PATH_TO_LABEL_CSV_DIRECTORY',
                        help='Directory containing CSV label files for evaluation cohorts.')
    parser.add_argument('--cohort', type=str, required=True,
                        help='Name of the evaluation cohort (e.g., Georgia).')
    parser.add_argument('--cohort_labels', type=str, required=True,
                        help='Basename of the CSV label file (without extension).')

    parser.add_argument('--results_dir', type=str, default='./results_fusion',
                        help='Directory containing trained models.')
    parser.add_argument('--models_exp_code', type=str, default='tangle_CMF_s1',
                        help='Experiment code of trained models under results_dir.')
    parser.add_argument('--save_exp_code', type=str, default='tangle_CMF_eval',
                        help='Code name for saving evaluation results.')

    parser.add_argument('--k', type=int, default=3,
                        help='Number of folds the model was trained on.')
    parser.add_argument('--fold', type=int, default=-1,
                        help='If set, evaluate only this fold.')

    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb'],
                        default='clam_sb', help='CLAM model type.')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small',
                        help='Size of the CLAM model.')
    parser.add_argument('--drop_out', type=float, default=0.25, help='Dropout used during training.')
    parser.add_argument('--embed_dim', type=int, default=1024,
                        help='Dimension of the patch features fed to the model.')
    parser.add_argument('--tangle_embedding_dim', type=int, default=1024,
                        help='Dimension of the Tangle embeddings.')

    return parser.parse_args()


def load_dataset(args):
    csv_path = os.path.join(args.label_dir, f"{args.cohort_labels}.csv")
    data_dir = os.path.join(args.data_root_dir, args.cohort)
    tangle_feature_dir = os.path.join(args.tangle_dir, args.cohort)

    dataset = Generic_MIL_Dataset_Tangle(
        csv_path=csv_path,
        data_dir=data_dir,
        shuffle=False,
        print_info=True,
        label_dict={'normal_tissue': 0, 'tumor_tissue': 1},
        patient_strat=False,
        ignore=[],
        tangle_feature_dir=tangle_feature_dir,
        tangle_embedding_dim=args.tangle_embedding_dim,
    )
    return dataset


def instantiate_model(args, device):
    model_kwargs = {
        'dropout': args.drop_out,
        'n_classes': 2,
        'embed_dim': args.embed_dim,
        'use_fusion': True,
        'tangle_dim': args.tangle_embedding_dim,
    }

    if args.model_type == 'clam_sb':
        model = CLAM_SB_Tangle(**model_kwargs)
    else:
        model = CLAM_MB_Tangle(**model_kwargs)

    model = model.to(device)
    model.eval()
    return model


def evaluate_fold(model, ckpt_path, loader, device):
    if not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}. Skipping.")
        return float('nan'), float('nan'), pd.DataFrame()

    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    patient_results, test_error, auc, acc_logger = summary(model, loader, n_classes=2, use_tangle=True)
    df = pd.DataFrame(patient_results).transpose()
    return auc, 1 - test_error, df


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_dataset(args)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=tangle_collate)

    args.save_dir = os.path.join('./eval_results', f"EVAL_{args.save_exp_code}")
    args.models_dir = os.path.join(args.results_dir, args.models_exp_code)
    os.makedirs(args.save_dir, exist_ok=True)

    settings = {key: getattr(args, key) for key in vars(args)}
    with open(os.path.join(args.save_dir, f'eval_experiment_{args.save_exp_code}.txt'), 'w') as f:
        for key, val in settings.items():
            f.write(f"{key}: {val}\n")
    print("\nEvaluation settings:")
    print(settings)

    folds = range(args.k) if args.fold == -1 else range(args.fold, args.fold + 1)

    all_auc = []
    all_acc = []
    for fold in folds:
        ckpt_path = os.path.join(args.models_dir, f's_{fold}_checkpoint.pt')
        auc, acc, df = evaluate_fold(instantiate_model(args, device), ckpt_path, loader, device)
        all_auc.append(auc)
        all_acc.append(acc)
        df.to_csv(os.path.join(args.save_dir, f'fold_{fold}.csv'), index=False)

    summary_df = pd.DataFrame({'fold': list(folds), 'test_auc': all_auc, 'test_acc': all_acc})
    summary_df.to_csv(os.path.join(args.save_dir, 'summary.csv'), index=False)
    print(f"\nEvaluation summary saved to {os.path.join(args.save_dir, 'summary.csv')}")


if __name__ == '__main__':
    main()