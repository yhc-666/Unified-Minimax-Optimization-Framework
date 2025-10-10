"""
Variational Autoencoder for Collaborative Filtering (VAE-CF) for rating prediction.
Encoder-decoder architecture with optional KL divergence regularization.
No sigmoid - outputs unbounded ratings for regression.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VAECF(nn.Module):
    def __init__(self, user_num, item_num, embedding_size, l2_reg_lambda,
                 hidden_dims=(60, 20), use_kl=True, kl_beta=0.2):
        super(VAECF, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.embedding_size = embedding_size
        self.l2_reg_lambda = l2_reg_lambda
        self.hidden_dims = hidden_dims
        self.use_kl = use_kl
        self.kl_beta = kl_beta

        h1, h2 = hidden_dims

        # Encoder: user interaction history (item_num dimensions) -> latent space
        self.enc_fc1 = nn.Linear(item_num, h1)
        self.enc_fc2 = nn.Linear(h1, h2)
        self.mu = nn.Linear(h2, embedding_size)
        self.logvar = nn.Linear(h2, embedding_size)

        # Item embeddings for decoding
        self.item_embedding = nn.Parameter(torch.randn(item_num, embedding_size))

        # Biases for rating prediction
        self.user_bias = nn.Parameter(torch.randn(user_num))
        self.item_bias = nn.Parameter(torch.randn(item_num))
        self.global_bias = nn.Parameter(torch.randn(1))

        # User interaction history storage (sparse or dense)
        self._user_hist_dense = None
        self._kl = torch.tensor(0.0)

    def set_user_hist_from_pairs(self, x_pairs, y_labels):
        """
        Build user interaction history from (user, item) pairs and labels.

        Args:
            x_pairs: (N, 2) array of (user_id, item_id)
            y_labels: (N,) array of ratings or binary labels
        """
        # Create dense user-item matrix
        user_hist = torch.zeros(self.user_num, self.item_num)

        for i in range(len(x_pairs)):
            user_id = int(x_pairs[i, 0])
            item_id = int(x_pairs[i, 1])
            user_hist[user_id, item_id] = float(y_labels[i])

        self._user_hist_dense = user_hist

        # Move to same device as model parameters
        device = next(self.parameters()).device
        self._user_hist_dense = self._user_hist_dense.to(device)

    def encode(self, user_hist):
        """
        Encode user interaction history to latent representation.

        Args:
            user_hist: (batch_size, item_num) user interaction vectors

        Returns:
            z: (batch_size, embedding_size) latent representation
        """
        h = F.relu(self.enc_fc1(user_hist))
        h = F.relu(self.enc_fc2(h))

        mu = self.mu(h)
        logvar = self.logvar(h)

        if self.use_kl and self.training:
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std

            # Compute KL divergence
            self._kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        else:
            z = mu
            self._kl = torch.tensor(0.0)

        return z

    def decode(self, z, item_ids):
        """
        Decode latent representation to rating predictions.

        Args:
            z: (batch_size, embedding_size) latent user representation
            item_ids: (batch_size,) item indices

        Returns:
            ratings: (batch_size,) predicted ratings
        """
        # Get item embeddings
        item_emb = self.item_embedding[item_ids]

        # Dot product between user latent and item embedding
        prediction = torch.sum(z * item_emb, dim=1)

        # Add biases (extract user_ids from z's batch indices - passed separately)
        # Note: user_ids will be passed via forward()

        return prediction

    def forward(self, user_id, item_id):
        """
        Forward pass for rating prediction.

        Args:
            user_id: (batch_size,) user indices
            item_id: (batch_size,) item indices

        Returns:
            prediction: (batch_size,) predicted ratings
        """
        # Get user interaction history
        if self._user_hist_dense is None:
            raise ValueError("Must call set_user_hist_from_pairs() before forward()")

        user_hist = self._user_hist_dense[user_id]

        # Encode to latent space
        z = self.encode(user_hist)

        # Decode to rating prediction
        item_emb = self.item_embedding[item_id]
        prediction = torch.sum(z * item_emb, dim=1)

        # Add biases
        b_u = self.user_bias[user_id]
        b_i = self.item_bias[item_id]
        prediction = prediction + b_u + b_i + self.global_bias

        return prediction

    def loss(self, prediction, y):
        """
        Compute total loss: MSE + KL divergence + L2 regularization.

        Args:
            prediction: (batch_size,) predicted ratings
            y: (batch_size,) true ratings

        Returns:
            total_loss: scalar loss
            mse: scalar MSE (for monitoring)
        """
        # MSE loss
        mse = torch.mean(torch.square(prediction - y))

        # KL divergence (computed in encode())
        kl_loss = self._kl if self.use_kl else torch.tensor(0.0)

        # L2 regularization
        l2_regularization = torch.sum(torch.square(self.item_embedding))
        l2_regularization += torch.sum(torch.square(self.user_bias))
        l2_regularization += torch.sum(torch.square(self.item_bias))
        l2_regularization += torch.sum(torch.square(self.global_bias))
        l2_regularization += torch.sum(torch.square(self.mu.weight))
        l2_regularization += torch.sum(torch.square(self.logvar.weight))
        l2_regularization += torch.sum(torch.square(self.enc_fc1.weight))
        l2_regularization += torch.sum(torch.square(self.enc_fc2.weight))

        total_loss = mse + self.kl_beta * kl_loss + self.l2_reg_lambda * l2_regularization

        return total_loss, mse
