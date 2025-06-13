import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

# --- MODIFIED: Import the standard CLAM model, not the Tangle one ---
from models.model_clam import CLAM_MB 
from phase6_model_rna_mlp import RnaMlp
from dataset_modules.dataset_generic_tangle import Generic_MIL_Dataset_Tangle

# A collate function to handle the 4-item output from our dataset
def tangle_collate(batch):
    features, labels, tangle_feats, slide_ids = zip(*batch)
    return list(features), torch.tensor(labels), list(tangle_feats), list(slide_ids)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Datasets ---
    print(f"Loading evaluation data for cohort: {args.cohort}")
    dataset = Generic_MIL_Dataset_Tangle(
        csv_path=os.path.join(args.label_dir, f"{args.cohort_labels}.csv"),
        data_dir=os.path.join(args.patch_dir, args.cohort),
        tangle_feature_dir=os.path.join(args.tangle_dir, args.cohort),
        label_dict={'normal_tissue': 0, 'tumor_tissue': 1}
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=tangle_collate)

    # --- 2. Load Trained Models ---
    # --- MODIFIED: Load the BASELINE CLAM model ---
    print(f"Loading CLAM model from: {args.clam_model_path}")
    # This now correctly instantiates the baseline model which expects 1024-dim input
    clam_model = CLAM_MB(n_classes=2) 
    clam_model.load_state_dict(torch.load(args.clam_model_path, map_location=device))
    clam_model.to(device)
    clam_model.eval()

    # Load RNA Model (MLP)
    print(f"Loading RNA MLP model from: {args.rna_model_path}")
    rna_model = RnaMlp(n_classes=2).to(device)
    rna_model.load_state_dict(torch.load(args.rna_model_path, map_location=device))
    rna_model.eval()

    # --- 3. Run Evaluation and Fusion ---
    all_labels = []
    fused_probs = []
    
    with torch.no_grad():
        for patch_features, label, tangle_embedding, slide_id in loader:
            # --- MODIFIED: Get Prediction A from the baseline CLAM model ---
            patch_features = patch_features[0].to(device)
            # The baseline model's forward pass does not take a tangle_feat argument
            _, prob_a, _, _, _ = clam_model(h=patch_features)
            prob_a = torch.softmax(prob_a, dim=1)

            # Get Prediction B: Conditionally run the RNA Model
            tangle_embedding = tangle_embedding[0]
            # Check if the slide has RNA data
            if tangle_embedding is not None and torch.any(tangle_embedding != 0):
                tangle_embedding = tangle_embedding.to(device)
                logit_b = rna_model(tangle_embedding.unsqueeze(0))
                prob_b = torch.softmax(logit_b, dim=1)
                # Fuse by averaging probabilities
                final_prob = (prob_a + prob_b) / 2
            else:
                # If no RNA data, use the CLAM prediction alone
                final_prob = prob_a
            
            fused_probs.append(final_prob.cpu().numpy())
            all_labels.append(label.item())

    # --- 4. Calculate and Print Metrics ---
    fused_probs = np.concatenate(fused_probs)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, fused_probs[:, 1])
    preds = np.argmax(fused_probs, axis=1)
    acc = (preds == all_labels).sum() / len(all_labels)
    
    print("\n--- Late Fusion Evaluation Results ---")
    print(f"Cohort: {args.cohort}")
    print(f"Final Fused AUC: {auc:.4f}")
    print(f"Final Fused Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, preds))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Late Fusion Evaluation Script')
    
    # --- Paths for Evaluation Data ---
    parser.add_argument('--cohort', type=str, required=True, help='Name of the evaluation cohort folder (e.g., Georgia).')
    parser.add_argument('--cohort_labels', type=str, required=True, help='Basename of the label CSV for the cohort (e.g., Georgia_filtered).')
    parser.add_argument('--label_dir', type=str, default='C:/Users/Mahon/Documents/Research/CLAM/Labels/Textual/', help='Directory containing label CSVs.')
    parser.add_argument('--patch_dir', type=str, default='./Phase3A_Baseline_Features/Evals', help='Root directory for patch features.')
    parser.add_argument('--tangle_dir', type=str, default='C:/Users/Mahon/Documents/Research/CLAM/EVAL_tangle_embeddings', help='Base directory for Tangle embeddings.')
    
    # --- Paths for Trained Models ---
    parser.add_argument('--clam_model_path', type=str, required=True, help='Path to the trained CLAM model checkpoint (.pt).')
    parser.add_argument('--rna_model_path', type=str, required=True, help='Path to the trained RNA MLP model checkpoint (.pt).')

    args = parser.parse_args()
    main(args)