import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb # Keep for compatibility if user uses it
import pickle
from scipy import stats

from torch.utils.data import Dataset
import h5py

# Assuming utils.utils will be available in the same environment
# If not, these might need to be defined or imported differently
try:
    from utils.utils import generate_split, nth 
except ImportError:
    print("Warning: utils.utils not found, stubs for generate_split/nth might be needed if create_splits/set_splits are used.")
    def generate_split(*args, **kwargs): raise NotImplementedError("generate_split not available")
    def nth(iterable, n, default=None): raise NotImplementedError("nth not available")


def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index = True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

    df.to_csv(filename)
    print()

class Generic_WSI_Classification_Dataset_RNA(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv',
        shuffle = False,
        seed = 7,
        print_info = True,
        label_dict = {},
        filter_dict = {},
        ignore=[],
        patient_strat=False,
        label_col = None,
        patient_voting = 'max',
        # New arguments for RNA concatenation
        rna_data_base_dir=None,
        use_rna_concatenation=False,
        master_rna_dim=19944,
        original_patch_feature_dim=1024,
        debug_rna_paths=False
        ):
        
        self.label_dict = label_dict
        if not isinstance(self.label_dict, dict): # Ensure label_dict is a dictionary
            raise TypeError(f"label_dict must be a dictionary, got {type(label_dict)}")

        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None # This will be set in Generic_MIL_Dataset_RNA
        if not label_col:
            label_col = 'label'
        self.label_col = label_col

        # New RNA-specific attributes
        self.rna_data_base_dir = rna_data_base_dir
        self.use_rna_concatenation = use_rna_concatenation
        self.master_rna_dim = master_rna_dim
        self.original_patch_feature_dim = original_patch_feature_dim
        self.debug_rna_paths = debug_rna_paths
        self.concatenated_feature_dim = self.original_patch_feature_dim + self.master_rna_dim if self.use_rna_concatenation else self.original_patch_feature_dim


        slide_data_df = pd.read_csv(csv_path) # Renamed to avoid conflict with self.slide_data
        
        if self.use_rna_concatenation and 'cohort' not in slide_data_df.columns:
            raise ValueError("CSV file must contain a 'cohort' column when use_rna_concatenation is True.")

        slide_data_df = self.filter_df(slide_data_df, filter_dict)
        slide_data_df = self.df_prep(slide_data_df, self.label_dict, ignore, self.label_col)
        
        original_dtypes = slide_data_df.dtypes # Store dtypes before potential numpy conversion

        if shuffle:
            np.random.seed(seed)
            slide_data_np = slide_data_df.to_numpy()
            np.random.shuffle(slide_data_np)
            slide_data_df = pd.DataFrame(slide_data_np, columns=slide_data_df.columns)
            # Preserve original dtypes as much as possible
            for col in slide_data_df.columns:
                try:
                    slide_data_df[col] = slide_data_df[col].astype(original_dtypes[col])
                except Exception as e:
                    print(f"Warning: Could not preserve dtype for column {col} after shuffle: {e}")


        self.slide_data = slide_data_df # Assign to self.slide_data

        self.patient_data_prep(patient_voting)
        self.cls_ids_prep()

        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self, patient_voting='max'):
        patients = np.unique(np.array(self.slide_data['case_id'])) 
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            if not locations: # Handle case where patient has no slides after filtering
                print(f"Warning: Patient {p} has no slides after filtering. Skipping.")
                continue

            slide_labels_for_patient = self.slide_data['label'][locations].values
            if len(slide_labels_for_patient) == 0: # Should not happen if locations is not empty
                 print(f"Warning: Patient {p} has slides but no labels. Skipping.")
                 continue

            if patient_voting == 'max':
                label = slide_labels_for_patient.max() 
            elif patient_voting == 'maj':
                mode_result = stats.mode(slide_labels_for_patient)
                # stats.mode can return ModeResult object, access [0] for mode array, then [0] for value if array
                label = mode_result[0][0] if isinstance(mode_result[0], np.ndarray) and len(mode_result[0]) > 0 else mode_result[0]

            else:
                raise NotImplementedError(f"Patient voting strategy '{patient_voting}' not implemented.")
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

    @staticmethod
    def df_prep(data, label_dict, ignore, label_col):
        if label_col != 'label' and label_col in data.columns:
            data['label'] = data[label_col].copy()
        elif 'label' not in data.columns and label_col not in data.columns:
            raise ValueError(f"Label column '{label_col}' or 'label' not found in dataframe.")
        elif 'label' not in data.columns and label_col == 'label': # No 'label' col, default also 'label'
             raise ValueError(f"Default label column 'label' not found in dataframe.")


        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        
        # Ensure all keys in data['label'] are in label_dict before mapping
        # unique_labels_in_data = data['label'].unique()
        # for key in unique_labels_in_data:
        #     if key not in label_dict:
        #         raise ValueError(f"Label '{key}' found in data but not in label_dict: {label_dict}")
        
        # Safer mapping: apply or map
        try:
            data['label'] = data['label'].apply(lambda x: label_dict.get(x, x)) # Keep original if not in dict, or error
            # Or strict: data['label'] = data['label'].map(label_dict) -> will produce NaN for missing keys
        except Exception as e:
            raise ValueError(f"Error mapping labels with label_dict {label_dict}: {e}")


        # Ensure all labels are now numeric as per label_dict values
        # for x in data['label']:
        #    if not isinstance(x, (int, np.integer)):
        #        raise ValueError(f"Label '{x}' was not correctly mapped to an integer by label_dict. Current labels: {data['label'].unique()}")
        return data

    def filter_df(self, df, filter_dict={}):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, dtype=bool) # Use dtype=bool
            for key, val in filter_dict.items():
                if key not in df.columns:
                    print(f"Warning: Filter key '{key}' not in DataFrame columns. Skipping this filter.")
                    continue
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        if not self.slide_data.empty and 'label' in self.slide_data.columns:
            print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        else:
            print("Slide data is empty or 'label' column missing.")

        for i in range(self.num_classes):
            if len(self.patient_cls_ids) > i and self.patient_cls_ids[i] is not None:
                 print('Patient-LVL; Number of samples registered in class %d: %d' % (i, len(self.patient_cls_ids[i])))
            if len(self.slide_cls_ids) > i and self.slide_cls_ids[i] is not None:
                 print('Slide-LVL; Number of samples registered in class %d: %d' % (i, len(self.slide_cls_ids[i])))

        if self.use_rna_concatenation:
            print(f"RNA concatenation is enabled.")
            print(f"Original patch feature dimension: {self.original_patch_feature_dim}")
            print(f"Master RNA dimension: {self.master_rna_dim}")
            print(f"Concatenated feature dimension: {self.concatenated_feature_dim}")


    def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
        settings = {
            'n_splits' : k,
            'val_num' : val_num,
            'test_num': test_num,
            'label_frac': label_frac,
            'seed': self.seed,
            'custom_test_ids': custom_test_ids
        }

        if self.patient_strat:
            settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
        else:
            settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

        self.split_gen = generate_split(**settings)

    def set_splits(self,start_from=None):
        if start_from:
            ids = nth(self.split_gen, start_from)
        else:
            ids = next(self.split_gen)

        if ids is None:
            raise ValueError("Split generation did not return valid IDs.")


        if self.patient_strat:
            slide_ids = [[] for i in range(len(ids))]
            for split_idx in range(len(ids)):
                for patient_idx in ids[split_idx]:
                    case_id = self.patient_data['case_id'][patient_idx]
                    slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                    slide_ids[split_idx].extend(slide_indices)
            self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]
        else:
            self.train_ids, self.val_ids, self.test_ids = ids

    def get_split_from_df(self, all_splits, split_key='train'):
        if split_key not in all_splits.columns:
             print(f"Warning: Split key '{split_key}' not in all_splits DataFrame. Returning None.")
             return None
        split_ids = all_splits[split_key].dropna().unique().tolist() # Use unique IDs

        if len(split_ids) > 0:
            mask = self.slide_data['slide_id'].isin(split_ids)
            df_slice = self.slide_data[mask].reset_index(drop=True)
            if df_slice.empty:
                print(f"Warning: No slides found for split '{split_key}' after filtering. Returning None.")
                return None
            
            split_dataset = Generic_Split_RNA(df_slice,
                                              data_dir=self.data_dir,
                                              num_classes=self.num_classes,
                                              rna_data_base_dir=self.rna_data_base_dir,
                                              use_rna_concatenation=self.use_rna_concatenation,
                                              master_rna_dim=self.master_rna_dim,
                                              original_patch_feature_dim=self.original_patch_feature_dim,
                                              debug_rna_paths=self.debug_rna_paths)
            return split_dataset
        else:
            print(f"Split key '{split_key}' resulted in an empty list of IDs. Returning None.")
            return None

    def get_merged_split_from_df(self, all_splits, split_keys=['train']):
        merged_slide_ids = []
        for split_key in split_keys:
            if split_key not in all_splits.columns:
                print(f"Warning: Split key '{split_key}' for merging not in all_splits DataFrame. Skipping.")
                continue
            split = all_splits[split_key].dropna().unique().tolist() # Use unique IDs
            merged_slide_ids.extend(split)
        
        merged_slide_ids = list(set(merged_slide_ids)) # Ensure unique IDs in merged list

        if len(merged_slide_ids) > 0:
            mask = self.slide_data['slide_id'].isin(merged_slide_ids)
            df_slice = self.slide_data[mask].reset_index(drop=True)
            if df_slice.empty:
                print(f"Warning: No slides found for merged split keys {split_keys} after filtering. Returning None.")
                return None

            split_dataset = Generic_Split_RNA(df_slice,
                                              data_dir=self.data_dir,
                                              num_classes=self.num_classes,
                                              rna_data_base_dir=self.rna_data_base_dir,
                                              use_rna_concatenation=self.use_rna_concatenation,
                                              master_rna_dim=self.master_rna_dim,
                                              original_patch_feature_dim=self.original_patch_feature_dim,
                                              debug_rna_paths=self.debug_rna_paths)
            return split_dataset
        else:
            print(f"Merged split keys {split_keys} resulted in an empty list of IDs. Returning None.")
            return None


    def return_splits(self, from_id=True, csv_path=None):
        common_params = {
            "data_dir": self.data_dir, "num_classes": self.num_classes,
            "rna_data_base_dir": self.rna_data_base_dir,
            "use_rna_concatenation": self.use_rna_concatenation,
            "master_rna_dim": self.master_rna_dim,
            "original_patch_feature_dim": self.original_patch_feature_dim,
            "debug_rna_paths": self.debug_rna_paths
        }
        train_split, val_split, test_split = None, None, None

        if from_id:
            if self.train_ids is not None and len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
                if not train_data.empty : train_split = Generic_Split_RNA(train_data, **common_params)
            
            if self.val_ids is not None and len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
                if not val_data.empty : val_split = Generic_Split_RNA(val_data, **common_params)
            
            if self.test_ids is not None and len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
                if not test_data.empty : test_split = Generic_Split_RNA(test_data, **common_params)
        else:
            if not csv_path or not os.path.exists(csv_path):
                 raise FileNotFoundError(f"CSV path for splits '{csv_path}' not provided or does not exist.")
            all_splits_df = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype if 'slide_id' in self.slide_data else str) # Robust dtype
            
            train_split = self.get_split_from_df(all_splits_df, 'train')
            val_split = self.get_split_from_df(all_splits_df, 'val')
            test_split = self.get_split_from_df(all_splits_df, 'test')
            
        return train_split, val_split, test_split

    def get_list(self, ids):
        if ids is None: return pd.Series(dtype='object') # Return empty series for None ids
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        if ids is None: return pd.Series(dtype='object')
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        # This method will be overridden by Generic_MIL_Dataset_RNA
        # but to make this class instantiable for testing, provide a basic implementation or raise error
        raise NotImplementedError("This method should be called from a subclass like Generic_MIL_Dataset_RNA")


    def test_split_gen(self, return_descriptor=False):
        df = None
        if return_descriptor:
            # Ensure all keys in label_dict values are represented, even if count is 0
            index_names = ["" for _ in range(self.num_classes)]
            for label_name, label_idx in self.label_dict.items():
                if 0 <= label_idx < self.num_classes:
                    index_names[label_idx] = label_name
                else:
                    print(f"Warning: Label index {label_idx} for {label_name} is out of num_classes bounds {self.num_classes}")

            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.full((self.num_classes, len(columns)), 0, dtype=np.int32), index=index_names, columns=columns)

        for split_name, split_ids in [('train', self.train_ids), ('val', self.val_ids), ('test', self.test_ids)]:
            if split_ids is None:
                print(f'\nNumber of {split_name} samples: 0 (IDs not set)')
                continue

            count = len(split_ids)
            print(f'\nNumber of {split_name} samples: {count}')
            if count > 0:
                labels = self.getlabel(split_ids)
                unique, counts = np.unique(labels, return_counts=True)
                for u_idx, u_label_val in enumerate(unique):
                    print(f'Number of samples in cls {u_label_val}: {counts[u_idx]}')
                    if return_descriptor and df is not None:
                        # Map u_label_val (numeric) back to label_name for df index
                        label_name_for_df = df.index[u_label_val] # Assuming df.index correctly maps numeric to name
                        df.loc[label_name_for_df, split_name] = counts[u_idx]
        
        if self.train_ids is not None and self.test_ids is not None:
            assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        if self.train_ids is not None and self.val_ids is not None:
            assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        if self.val_ids is not None and self.test_ids is not None:
            assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

        return df


    def save_split(self, filename):
        train_split_list = self.get_list(self.train_ids)
        val_split_list = self.get_list(self.val_ids)
        test_split_list = self.get_list(self.test_ids)
        
        # Pad shorter lists with None or NaN to make DataFrame creation possible if lengths differ
        max_len = max(len(train_split_list), len(val_split_list), len(test_split_list))

        df_tr = pd.DataFrame({'train': pd.Series(train_split_list, index=range(max_len))})
        df_v = pd.DataFrame({'val': pd.Series(val_split_list, index=range(max_len))})
        df_t = pd.DataFrame({'test': pd.Series(test_split_list, index=range(max_len))})
        
        df_concat = pd.concat([df_tr, df_v, df_t], axis=1)
        df_concat.to_csv(filename, index = False)


class Generic_MIL_Dataset_RNA(Generic_WSI_Classification_Dataset_RNA):
    def __init__(self,
        data_dir, # This is the patch feature directory
        **kwargs):
    
        super(Generic_MIL_Dataset_RNA, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.use_h5 = False # Default to .pt files

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        # Get slide_id and label from the DataFrame at the given index
        slide_id = self.slide_data['slide_id'].iloc[idx]
        label = self.slide_data['label'].iloc[idx]
        
        # Determine the data directory for patch features
        # (self.data_dir is for patch features, self.rna_data_base_dir for RNA)
        current_patch_data_dir = self.data_dir 
        if type(self.data_dir) == dict:
            # This assumes 'source' column exists if data_dir is a dict
            if 'source' not in self.slide_data.columns:
                raise ValueError(f"Dataset 'source' column not found, but self.data_dir is a dictionary.")
            source = self.slide_data['source'].iloc[idx]
            current_patch_data_dir = self.data_dir[source]

        # Load original patch features
        if not self.use_h5:
            if current_patch_data_dir:
                feature_file_path = os.path.join(current_patch_data_dir, 'pt_files', f'{slide_id}.pt')
                if not os.path.exists(feature_file_path):
                    raise FileNotFoundError(f"Patch feature file not found: {feature_file_path}")
                features = torch.load(feature_file_path) # Expected shape: [N, original_patch_feature_dim]
            else:
                raise ValueError("current_patch_data_dir is not set for .pt files.")
        else: # use_h5 is True
            if current_patch_data_dir:
                feature_file_path = os.path.join(current_patch_data_dir, 'h5_files', f'{slide_id}.h5')
                if not os.path.exists(feature_file_path):
                    raise FileNotFoundError(f"Patch feature file not found: {feature_file_path}")
                with h5py.File(feature_file_path, 'r') as hdf5_file:
                    patch_features_np = hdf5_file['features'][:]
                features = torch.from_numpy(patch_features_np) # Expected shape: [N, original_patch_feature_dim]
            else:
                raise ValueError("current_patch_data_dir is not set for .h5 files.")

        # --- RNA Concatenation Logic ---
        if self.use_rna_concatenation:
            rna_profile_tensor = None
            # 'cohort' column is essential for finding the RNA file
            if 'cohort' not in self.slide_data.columns:
                raise ValueError(f"Missing 'cohort' column in slide_data for slide_id {slide_id}, needed for RNA concatenation.")
            cohort = self.slide_data['cohort'].iloc[idx]
            
            if not self.rna_data_base_dir:
                print(f"Warning: use_rna_concatenation is True but rna_data_base_dir is not set for slide {slide_id}. Using zeros for RNA.")
                # Fall through to use zeros if rna_data_base_dir is None
            else:
                rna_file_path = os.path.join(self.rna_data_base_dir, str(cohort), 'NPYs', f"{slide_id}.npy")

                if self.debug_rna_paths:
                    print(f"Debug: Attempting to load RNA for slide_id: {slide_id}, cohort: {cohort}, path: {rna_file_path}")

                if os.path.exists(rna_file_path):
                    try:
                        rna_profile_np = np.load(rna_file_path)
                        rna_profile_tensor = torch.from_numpy(rna_profile_np).float() # Expected: [master_rna_dim]
                        
                        if rna_profile_tensor.ndim == 0 or rna_profile_tensor.shape[0] != self.master_rna_dim:
                            print(f"Warning: RNA profile for {slide_id} at {rna_file_path} has unexpected shape {rna_profile_tensor.shape}, expected ({self.master_rna_dim},). Using zeros.")
                            rna_profile_tensor = None
                    except Exception as e:
                        print(f"Warning: Could not load or process RNA file {rna_file_path}: {e}. Using zeros.")
                        rna_profile_tensor = None
                elif self.debug_rna_paths: # Only print if debug is on and file not found
                    print(f"Debug: RNA file not found: {rna_file_path}. Using zeros.")
                # If file doesn't exist and not debugging, it will fall through to use zeros.

            num_patches = features.shape[0]
            if rna_profile_tensor is not None:
                # Expand RNA vector to match number of patches: [N, master_rna_dim]
                rna_profile_expanded = rna_profile_tensor.unsqueeze(0).repeat(num_patches, 1)
            else:
                # Create a zero tensor if RNA is not available/loadable: [N, master_rna_dim]
                rna_profile_expanded = torch.zeros(num_patches, self.master_rna_dim, dtype=features.dtype, device=features.device)

            # Concatenate along the feature dimension
            concatenated_features = torch.cat((features, rna_profile_expanded), dim=1)
            
            # Final dimension check for concatenated features
            expected_concatenated_dim = self.original_patch_feature_dim + self.master_rna_dim
            if concatenated_features.shape[1] != expected_concatenated_dim:
                raise ValueError(f"Concatenated feature dimension mismatch for slide {slide_id}. Expected {expected_concatenated_dim}, got {concatenated_features.shape[1]}. Original patch dim: {features.shape[1]}, RNA expanded dim: {rna_profile_expanded.shape[1]}")
            
            return concatenated_features, label
        
        else: # Not using RNA concatenation
            # Dimension check for original features when not concatenating
            if features.shape[1] != self.original_patch_feature_dim:
                 raise ValueError(f"Original feature dimension mismatch for slide {slide_id} (no concatenation). Expected {self.original_patch_feature_dim}, got {features.shape[1]}")
            return features, label


class Generic_Split_RNA(Generic_MIL_Dataset_RNA):
    # This class inherits __getitem__ from Generic_MIL_Dataset_RNA.
    # Its main purpose is to be a wrapper around a DataFrame slice (split).
    def __init__(self, slide_data_df, data_dir=None, num_classes=2,
                 rna_data_base_dir=None, use_rna_concatenation=False,
                 master_rna_dim=19944, original_patch_feature_dim=1024,
                 debug_rna_paths=False):
        
        # We don't call super().__init__() of Generic_MIL_Dataset_RNA's ultimate base (Dataset)
        # as that would re-process CSVs. Instead, we manually set the necessary attributes
        # that Generic_MIL_Dataset_RNA's __getitem__ will need.

        self.slide_data = slide_data_df # This is the crucial part - it gets a slice of the main dataset's df
        self.data_dir = data_dir # Patch feature directory
        self.num_classes = num_classes
        self.use_h5 = False # Default, can be changed by load_from_h5 if that method is called on an instance.

        # RNA-specific attributes, passed through
        self.rna_data_base_dir = rna_data_base_dir
        self.use_rna_concatenation = use_rna_concatenation
        self.master_rna_dim = master_rna_dim
        self.original_patch_feature_dim = original_patch_feature_dim
        self.concatenated_feature_dim = self.original_patch_feature_dim + self.master_rna_dim if self.use_rna_concatenation else self.original_patch_feature_dim
        self.debug_rna_paths = debug_rna_paths

        # Prepare slide_cls_ids for this specific split, if needed by any inherited methods
        # or if the split itself is used for stratification later (unlikely here).
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        if 'label' in self.slide_data.columns:
            for i in range(self.num_classes):
                self.slide_cls_ids[i] = self.slide_data[self.slide_data['label'] == i].index.tolist() # Store indices from this split's df
        else:
            print("Warning: 'label' column not in slide_data for Generic_Split_RNA, slide_cls_ids will be empty.")


    def __len__(self):
        return len(self.slide_data)
    
    # load_from_h5 is inherited from Generic_MIL_Dataset_RNA
    # __getitem__ is inherited from Generic_MIL_Dataset_RNA and will operate on self.slide_data (the split)