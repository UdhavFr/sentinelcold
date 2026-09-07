"""
SentinelCold — Tier 4: Dense Autoencoder (AE-B)
Simpler MLP-based autoencoder alternative.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.logger import tier_logger

log = tier_logger(4)


class DenseAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_dims: list[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32, 16]

        # Encoder
        enc_layers = []
        in_dim = n_features
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (symmetric)
        dec_layers = []
        for h in reversed(hidden_dims[:-1]):
            dec_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, n_features))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DenseAutoencoderTrainer:
    """Train and score the Dense Autoencoder."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 64)
        self.lr = config.get("learning_rate", 0.001)
        self.hidden_dims = config.get("hidden_dims", [64, 32, 16])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_: DenseAutoencoder | None = None
        self.threshold_: float = 0.0

    @property
    def name(self) -> str:
        return "Dense Autoencoder"

    def fit(self, X_normal: np.ndarray) -> "DenseAutoencoderTrainer":
        n_features = X_normal.shape[1]
        self.model_ = DenseAutoencoder(n_features, self.hidden_dims).to(self.device)
        dataset = TensorDataset(torch.FloatTensor(X_normal))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                x_hat = self.model_(batch)
                loss = criterion(x_hat, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)

            if (epoch + 1) % 20 == 0:
                log.info("Dense AE Epoch %d/%d — MSE: %.6f", epoch + 1, self.epochs, total_loss / len(X_normal))

        errors = self.compute_reconstruction_error(X_normal)
        self.threshold_ = float(np.percentile(errors, 95))
        log.info("Dense AE threshold (95th pctl): %.6f", self.threshold_)
        return self

    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            X_hat = self.model_(X_t)
            errors = ((X_t - X_hat) ** 2).mean(dim=1).cpu().numpy()
        return errors

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return self.compute_reconstruction_error(X)

    def predict_anomaly(self, X: np.ndarray) -> np.ndarray:
        return (self.compute_reconstruction_error(X) > self.threshold_).astype(int)
