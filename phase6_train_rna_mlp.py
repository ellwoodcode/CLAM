import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import argparse
from phase6_model_rna_mlp import RnaMlp # Import the model we just defined

# A simple Dataset to load only the .npy Tangle embeddings and their labels
class RnaDataset(Dataset):
    def __init__(self, csv_path, npy_dir):
        self.df = pd.read_csv(csv_path)
        self.npy_dir = npy_dir
        # Filter out rows where RNA-seq data might be missing if necessary
        # For this example, we assume the training CSV only contains slides with RNA data.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row['slide_id']
        label = row['label']
        
        # Load the .npy Tangle/RNA embedding
        npy_path = os.path.join(self.npy_dir, f"{slide_id}.npy")
        embedding = torch.from_numpy(np.load(npy_path)).float()
        
        return embedding, torch.tensor(label).long()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    dataset = RnaDataset(args.csv_path, args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # --- Model ---
    model = RnaMlp(input_dim=1024, n_classes=2, dropout=args.drop_out).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)

    # --- Training Loop ---
    for epoch in range(args.max_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for embeddings, labels in loader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(embeddings)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1}/{args.max_epochs} -> Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # --- Save Model ---
    os.makedirs(args.results_dir, exist_ok=True)
    save_path = os.path.join(args.results_dir, "rna_mlp_final.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. RNA MLP model saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNA MLP Training Script')
    # --- IMPORTANT: Update these default paths for your training set ---
    parser.add_argument('--csv_path', type=str, default='C:/Users/Mahon/Documents/Research/CLAM/Labels/Boolean/labels_ims1.csv', help='Path to CSV file with slide_ids and labels.')
    parser.add_argument('--data_dir', type=str, default='C:/Users/Mahon/Documents/Research/CLAM/IMS1_tangle_embeddings', help='Directory of .npy Tangle embeddings.')
    parser.add_argument('--results_dir', type=str, default='./rna_mlp_model', help='Directory to save the trained model.')
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--reg', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--drop_out', type=float, default=0.25)
    
    args = parser.parse_args()
    main(args)