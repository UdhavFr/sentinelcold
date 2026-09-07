"""
SentinelCold — Tier 4: Variational Autoencoder (AE-C)
VAE with KL-divergence regularization for better latent space.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.logger import tier_logger

log = tier_logger(4)


class VAE(nn.Module):
    def __init__(self, n_features: int, hidden_dims: list[int] = None, latent_dim: int = 16):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        self.latent_dim = latent_dim

        # Encoder
        enc_layers = []
        in_dim = n_features
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # Decoder
        dec_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, n_features))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


class VAETrainer:
    """Train and score the Variational Autoencoder."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 64)
        self.lr = config.get("learning_rate", 0.001)
        self.latent_dim = config.get("latent_dim", 16)
        self.kl_weight = config.get("kl_weight", 0.001)
        self.hidden_dims = config.get("hidden_dims", [64, 32])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_: VAE | None = None
        self.threshold_: float = 0.0

    @property
    def name(self) -> str:
        return "Variational Autoencoder"

    def fit(self, X_normal: np.ndarray) -> "VAETrainer":
        n_features = X_normal.shape[1]
        self.model_ = VAE(n_features, self.hidden_dims, self.latent_dim).to(self.device)
        dataset = TensorDataset(torch.FloatTensor(X_normal))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                x_hat, mu, logvar = self.model_(batch)
                recon_loss = nn.MSELoss()(x_hat, batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(batch)
                loss = recon_loss + self.kl_weight * kl_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)

            if (epoch + 1) % 20 == 0:
                log.info("VAE Epoch %d/%d — Loss: %.6f", epoch + 1, self.epochs, total_loss / len(X_normal))

        errors = self.compute_reconstruction_error(X_normal)
        self.threshold_ = float(np.percentile(errors, 95))
        log.info("VAE threshold (95th pctl): %.6f", self.threshold_)
        return self

    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            X_hat, _, _ = self.model_(X_t)
            errors = ((X_t - X_hat) ** 2).mean(dim=1).cpu().numpy()
        return errors

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return self.compute_reconstruction_error(X)

    def predict_anomaly(self, X: np.ndarray) -> np.ndarray:
        return (self.compute_reconstruction_error(X) > self.threshold_).astype(int)
