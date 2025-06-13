from phase4_tangle_dataset import TangleDataset
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch.optim as optim

# Model Definition
class TangleModel(nn.Module):
    def __init__(self, input_dim=1024, rna_dim=19944):
        super(TangleModel, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, rna_dim)
        )

    def forward(self, x):
        attn_weights = torch.softmax(self.attention(x), dim=0)  # shape: [n_patches, 1]
        attended = torch.sum(attn_weights * x, dim=0)           # shape: [input_dim]
        output = self.decoder(attended)
        return output

# Dataset Paths
save_path = "C:/Users/Mahon/OneDrive/Documents/TANGLE/tangle_model_ims1.pth"
pt_dir = "C:/Users/Mahon/Documents/Research/CLAM/Phase3A_Baseline_Features/Train/TCGA_ims1/pt_files"
npy_dir = "C:/Users/Mahon/OneDrive/Documents/TANGLE/rna_seq_files_processed_log_transformed/IMS1/NPYs"

# Load Data
dataset = TangleDataset(pt_dir, npy_dir)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TangleModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# Training Loop
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    preds_all = []
    targets_all = []
    for slide, rna in loader:
        slide, rna = slide[0].to(device), rna[0].to(device)  # single sample per batch
        optimizer.zero_grad()
        outputs = model(slide)
        loss = loss_fn(outputs, rna)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds_all.append(outputs.detach().cpu().numpy())
        targets_all.append(rna.detach().cpu().numpy())

     # --- Metrics ---
    preds_all = np.vstack(preds_all)
    targets_all = np.vstack(targets_all)
    mse = mean_squared_error(targets_all, preds_all)
    r2 = r2_score(targets_all, preds_all)

    print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.4f} | MSE: {mse:.4f} | R²: {r2:.4f}")

torch.save(model.state_dict(), save_path)
print(f"✅ Model saved to {save_path}")