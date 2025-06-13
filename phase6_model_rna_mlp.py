import torch
import torch.nn as nn

class RnaMlp(nn.Module):
    def __init__(self, input_dim=1024, n_classes=2, dropout=0.25):
        """
        A simple MLP for processing 1D RNA-seq/Tangle embeddings.
        
        Args:
            input_dim (int): The size of the input embedding (e.g., 1024).
            n_classes (int): The number of output classes.
            dropout (float): The dropout rate.
        """
        super(RnaMlp, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.model(x)