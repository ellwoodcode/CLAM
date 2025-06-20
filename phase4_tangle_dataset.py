import os
import torch
import numpy as np
from torch.utils.data import Dataset

class TangleDataset(Dataset):
    """Dataset returning patch-level features and RNA vectors for each slide.

    Parameters
    ----------
    patch_dir : str
        Directory containing `.pt` files of patch features for each slide.
    rna_dir : str
        Directory containing `.npy` files with RNA expression vectors per slide.
    """

    def __init__(self, patch_dir: str, rna_dir: str):
        self.patch_dir = patch_dir
        self.rna_dir = rna_dir

        # Collect slide IDs from patch directory based on `.pt` filenames
        self.slide_ids = []
        if os.path.isdir(self.patch_dir):
            for fname in os.listdir(self.patch_dir):
                if fname.endswith('.pt'):
                    slide_id = os.path.splitext(fname)[0]
                    self.slide_ids.append(slide_id)
        self.slide_ids.sort()

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        pt_path = os.path.join(self.patch_dir, f"{slide_id}.pt")
        rna_path = os.path.join(self.rna_dir, f"{slide_id}.npy")

        patch_features = torch.load(pt_path)
        rna_vector = torch.from_numpy(np.load(rna_path)).float()
        return patch_features, rna_vector
