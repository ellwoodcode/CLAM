from __future__ import print_function

import argparse
import os
import numpy as np
import pandas as pd

# internal imports
from utils.file_utils import save_pkl
from utils.utils import seed_torch
from utils.core_utils_tangle import train
from dataset_modules.dataset_generic_tangle import Generic_MIL_Dataset_Tangle

# pytorch imports
import torch


def main(args):
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)

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
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False,
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc,
                             'val_auc': all_val_auc,
                             'test_acc': all_test_acc, 'val_acc': all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))


def parse_args():
    parser = argparse.ArgumentParser(
        description='WSI training with Tangle embeddings and cross-modal fusion')
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='directory containing patch features')
    parser.add_argument('--tangle_feature_dir', type=str, required=True,
                        help='directory containing Tangle embedding .npy files')
    parser.add_argument('--tangle_embedding_dim', type=int, default=1024,
                        help='dimension of the Tangle embeddings')
    parser.add_argument('--embed_dim', type=int, default=1024,
                        help='dimension of base patch features')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--label_frac', type=float, default=1.0)
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--k_start', type=int, default=-1)
    parser.add_argument('--k_end', type=int, default=-1)
    parser.add_argument('--results_dir', default='./results_fusion')
    parser.add_argument('--split_dir', type=str, required=True,
                        help='directory with data splits')
    parser.add_argument('--log_data', action='store_true', default=False)
    parser.add_argument('--testing', action='store_true', default=False)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--drop_out', type=float, default=0.25)
    parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce')
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'],
                        default='clam_sb')
    parser.add_argument('--exp_code', type=str, default='fusion_experiment')
    parser.add_argument('--weighted_sample', action='store_true', default=False)
    parser.add_argument('--model_size', type=str, choices=['small', 'big'],
                        default='small')
    parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'],
                        required=True)
    # CLAM specific
    parser.add_argument('--no_inst_cluster', action='store_true', default=False)
    parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None)
    parser.add_argument('--subtyping', action='store_true', default=False)
    parser.add_argument('--bag_weight', type=float, default=0.7)
    parser.add_argument('--B', type=int, default=8)
    args = parser.parse_args()
    return args


def prepare_dataset(args):
    tangle_kwargs = {
        'tangle_feature_dir': args.tangle_feature_dir,
        'tangle_embedding_dim': args.tangle_embedding_dim,
    }

    if args.task == 'task_1_tumor_vs_normal':
        args.n_classes = 2
        csv_path = 'dataset_csv/labels_ims1_filtered.csv'
        dataset = Generic_MIL_Dataset_Tangle(
            csv_path=csv_path,
            data_dir=args.data_root_dir,
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict={'normal_tissue': 0, 'tumor_tissue': 1},
            patient_strat=False,
            ignore=[],
            **tangle_kwargs)
    elif args.task == 'task_2_tumor_subtyping':
        args.n_classes = 3
        csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv'
        dataset = Generic_MIL_Dataset_Tangle(
            csv_path=csv_path,
            data_dir=os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict={'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2},
            patient_strat=False,
            ignore=[],
            **tangle_kwargs)
        if args.model_type in ['clam_sb', 'clam_mb']:
            assert args.subtyping
    else:
        raise NotImplementedError

    return dataset


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_torch(args.seed)

    # prepare dataset
    dataset = prepare_dataset(args)

    if args.split_dir is None:
        args.split_dir = os.path.join('splits', args.task + '_{}'.format(int(args.label_frac * 100)))
    else:
        args.split_dir = os.path.join('splits', args.split_dir)

    if not os.path.isdir(args.split_dir):
        raise FileNotFoundError('Split directory not found: {}'.format(args.split_dir))

    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    os.makedirs(args.results_dir, exist_ok=True)

    settings = {
        'num_splits': args.k,
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
        'seed': args.seed,
        'model_type': args.model_type,
        'model_size': args.model_size,
        'use_drop_out': args.drop_out,
        'weighted_sample': args.weighted_sample,
        'opt': args.opt,
        'use_tangle_fusion': True,
        'tangle_feature_dir': args.tangle_feature_dir,
        'embed_dim': args.embed_dim,
    }
    if args.model_type in ['clam_sb', 'clam_mb']:
        settings.update({'bag_weight': args.bag_weight,
                         'inst_loss': args.inst_loss,
                         'B': args.B})

    with open(os.path.join(args.results_dir, 'experiment_{}.txt'.format(args.exp_code)), 'w') as f:
        print(settings, file=f)

    print('################# Settings ###################')
    for key, val in settings.items():
        print('{}:  {}'.format(key, val))

    # dataset variable defined earlier
    globals()['dataset'] = dataset

    results = main(args)
    print('finished!')
    print('end script')

