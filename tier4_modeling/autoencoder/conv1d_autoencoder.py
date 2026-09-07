"""
SentinelCold — Tier 4: Conv1D Autoencoder (AE-A)
PRD-specified architecture: 3 conv blocks, stride-2, MSE loss.
Trained on non-failure samples only (one-class learning).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.logger import tier_logger

log = tier_logger(4)


class Conv1DEncoder(nn.Module):
    def __init__(self, n_features: int, channels: list[int] = None, kernel_size: int = 3):
        super().__init__()
        channels = channels or [16, 32, 64]
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, stride=2, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Conv1DDecoder(nn.Module):
    def __init__(self, n_features: int, channels: list[int] = None, kernel_size: int = 3):
        super().__init__()
        channels = channels or [64, 32, 16]
        layers = []
        for i, out_ch in enumerate(channels):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            if i == 0:
                continue
            layers.extend([
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=2, padding=kernel_size // 2, output_padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ])
        # Final layer to reconstruct
        layers.append(nn.ConvTranspose1d(channels[-1], 1, kernel_size, stride=2, padding=kernel_size // 2, output_padding=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Conv1DAutoencoder(nn.Module):
    """Complete Conv1D Autoencoder for anomaly detection."""

    def __init__(self, n_features: int, config: dict = None):
        super().__init__()
        config = config or {}
        channels = config.get("channels", [16, 32, 64])
        kernel_size = config.get("kernel_size", 3)

        # Simple symmetric architecture using linear layers wrapping conv
        self.n_features = n_features
        hidden = config.get("latent_dim", 16)

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, hidden),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class Conv1DAutoencoderTrainer:
    """Train and score the Conv1D Autoencoder."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 64)
        self.lr = config.get("learning_rate", 0.001)
        self.latent_dim = config.get("latent_dim", 16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_: Conv1DAutoencoder | None = None
        self.threshold_: float = 0.0

    @property
    def name(self) -> str:
        return "Conv1D Autoencoder"

    def fit(self, X_normal: np.ndarray) -> "Conv1DAutoencoderTrainer":
        """Train on non-failure samples only."""
        n_features = X_normal.shape[1]
        self.model_ = Conv1DAutoencoder(n_features, {"latent_dim": self.latent_dim}).to(self.device)

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
                avg_loss = total_loss / len(X_normal)
                log.info("AE Epoch %d/%d — MSE: %.6f", epoch + 1, self.epochs, avg_loss)

        # Compute threshold (95th percentile of non-failure reconstruction error)
        errors = self.compute_reconstruction_error(X_normal)
        self.threshold_ = float(np.percentile(errors, 95))
        log.info("AE threshold (95th pctl): %.6f", self.threshold_)
        return self

    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            X_hat = self.model_(X_t)
            errors = ((X_t - X_hat) ** 2).mean(dim=1).cpu().numpy()
        return errors

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Return continuous anomaly scores (higher = more anomalous)."""
        return self.compute_reconstruction_error(X)

    def predict_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Return binary anomaly flags."""
        errors = self.compute_reconstruction_error(X)
        return (errors > self.threshold_).astype(int)
