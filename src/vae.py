"""
VAE Models for Music Clustering.

Contains:
  - BasicVAE: MLP-based Variational Autoencoder (Easy task)
  - ConvVAE: Convolutional VAE for Mel-spectrogram features (Medium task)
  - BetaVAE: Beta-VAE for disentangled representations (Hard task)
  - CVAE: Conditional VAE conditioned on language + genre (Hard task)
  - MultiModalVAE: Dual-encoder VAE for audio + lyrics fusion (Hard task)
  - vae_loss: shared loss function (supports beta weighting)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class BasicVAE(nn.Module):
    """
    Basic MLP Variational Autoencoder.

    Architecture:
        Encoder: input_dim → hidden_dims → (mu, logvar) of latent_dim
        Decoder: latent_dim → hidden_dims(reversed) → input_dim

    Loss: MSE Reconstruction + KL Divergence
    """

    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: List[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.hidden_dims = hidden_dims

        # ----- Encoder -----
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.2))
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space: mu and logvar
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # ----- Decoder -----
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims))
        in_dim = latent_dim
        for h_dim in reversed_dims:
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.2))
            in_dim = h_dim
        decoder_layers.append(nn.Linear(reversed_dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * sigma."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass.

        Returns dict with:
            reconstruction: reconstructed input
            mu: latent mean
            logvar: latent log-variance
            z: sampled latent vector
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return {
            "reconstruction": reconstruction,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (mu) without sampling."""
        mu, _ = self.encode(x)
        return mu


# ============================================================
# ConvVAE — Medium Task
# ============================================================

class ConvVAE(nn.Module):
    """
    Convolutional VAE for 1-D Mel-spectrogram features.

    Treats the flat mel feature vector (256-dim = 2*N_MELS) as a 1-D
    sequence and applies Conv1d layers for encoding.

    Architecture:
        Encoder: (B, 1, input_dim) → Conv1d ×3 → GlobalAvgPool → [mu, logvar]
        Decoder: z → Linear → Reshape → ConvTranspose1d ×3 → (B, input_dim)

    Same interface as BasicVAE (forward returns same dict keys,
    get_latent() returns mu) so train_vae() works without modification.
    """

    def __init__(self, input_dim: int, latent_dim: int = 32,
                 channels: List[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        if channels is None:
            channels = [32, 64, 128]
        self.channels = channels

        # ----- Encoder -----
        enc_layers = []
        in_ch = 1
        for out_ch in channels:
            enc_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.conv_encoder = nn.Sequential(*enc_layers)
        # GlobalAvgPool → (B, channels[-1])
        self.fc_mu = nn.Linear(channels[-1], latent_dim)
        self.fc_logvar = nn.Linear(channels[-1], latent_dim)

        # ----- Decoder -----
        # Project z back to (B, channels[-1] * input_dim)
        self.fc_decode = nn.Linear(latent_dim, channels[-1] * input_dim)
        dec_layers = []
        rev_channels = list(reversed(channels))
        in_ch = rev_channels[0]
        for out_ch in rev_channels[1:]:
            dec_layers += [
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        dec_layers.append(
            nn.ConvTranspose1d(in_ch, 1, kernel_size=3, padding=1)
        )
        self.conv_decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, input_dim) → (B, 1, input_dim)
        h = self.conv_encoder(x.unsqueeze(1))  # (B, channels[-1], input_dim)
        h = h.mean(dim=2)                       # GlobalAvgPool → (B, channels[-1])
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)                               # (B, channels[-1]*input_dim)
        h = h.view(z.size(0), self.channels[-1], self.input_dim)  # (B, C, L)
        out = self.conv_decoder(h)                          # (B, 1, input_dim)
        return out.squeeze(1)                               # (B, input_dim)

    def forward(self, x: torch.Tensor) -> dict:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return {"reconstruction": reconstruction, "mu": mu, "logvar": logvar, "z": z}

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu


# ============================================================
# BetaVAE — Hard Task
# ============================================================

class BetaVAE(BasicVAE):
    """
    Beta-VAE: BasicVAE with a beta > 1 KL weight for disentanglement.

    Identical architecture to BasicVAE. The beta parameter is stored
    as an attribute and passed as kl_weight to vae_loss() at training time:

        losses = vae_loss(recon, x, mu, logvar, kl_weight=model.beta)

    Args:
        input_dim: input feature dimension
        latent_dim: latent space dimension
        hidden_dims: hidden layer sizes
        beta: KL divergence weight (default 4.0; try 2, 4, 8, 10)
    """

    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: List[int] = None, beta: float = 4.0):
        super().__init__(input_dim, latent_dim, hidden_dims)
        self.beta = beta


# ============================================================
# CVAE — Hard Task
# ============================================================

class CVAE(nn.Module):
    """
    Conditional VAE conditioned on language + genre labels.

    Architecture:
        Encoder: concat(x, one_hot(c)) → hidden → (mu, logvar)
        Decoder: concat(z, one_hot(c)) → hidden → reconstruction

    The condition vector c is a one-hot encoding of a combined
    (language, genre) class index. Use condition_dim = n_languages * n_genres.

    Args:
        input_dim: audio feature dimension
        latent_dim: latent space dimension
        condition_dim: length of condition one-hot vector (default 12 = 2*6)
        hidden_dims: hidden layer sizes
    """

    def __init__(self, input_dim: int, latent_dim: int = 32,
                 condition_dim: int = 12,
                 hidden_dims: List[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.hidden_dims = hidden_dims

        # ----- Encoder takes (x || c) -----
        enc_in = input_dim + condition_dim
        enc_layers = []
        in_dim = enc_in
        for h_dim in hidden_dims:
            enc_layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ]
            in_dim = h_dim
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # ----- Decoder takes (z || c) -----
        dec_in = latent_dim + condition_dim
        dec_layers = []
        in_dim = dec_in
        for h_dim in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ]
            in_dim = h_dim
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor,
               c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode (x, condition) → (mu, logvar)."""
        xc = torch.cat([x, c], dim=-1)
        h = self.encoder(xc)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decode (z, condition) → reconstruction."""
        zc = torch.cat([z, c], dim=-1)
        return self.decoder(zc)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> dict:
        """
        Full forward pass.

        Args:
            x: feature tensor (B, input_dim)
            c: condition one-hot tensor (B, condition_dim)

        Returns dict with: reconstruction, mu, logvar, z
        """
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, c)
        return {"reconstruction": reconstruction, "mu": mu,
                "logvar": logvar, "z": z}

    def get_latent(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Get latent representation (mu) without sampling."""
        mu, _ = self.encode(x, c)
        return mu


# ============================================================
# MultiModalVAE — Hard Task
# ============================================================

class MultiModalVAE(nn.Module):
    """
    Multi-modal VAE with separate encoders for audio features and lyrics
    embeddings, fused at the latent level.

    Architecture:
        Audio encoder:  audio_dim → hidden → h_audio
        Lyrics encoder: lyrics_dim → smaller_hidden → h_lyrics
        Fusion:         cat(h_audio, h_lyrics) → fc_mu / fc_logvar
        Decoder:        z → reversed_hidden → audio_dim  (reconstructs audio only)

    Args:
        audio_dim: audio feature dimension (e.g. 256 for mel features)
        lyrics_dim: lyrics embedding dimension (768 for LaBSE, 384 for MiniLM)
        latent_dim: joint latent space dimension
        hidden_dims: hidden layer sizes for audio encoder
        lyrics_hidden: hidden size for lyrics encoder (scalar, default 256)
    """

    def __init__(self, audio_dim: int, lyrics_dim: int = 768,
                 latent_dim: int = 32,
                 hidden_dims: List[int] = None,
                 lyrics_hidden: int = 256):
        super().__init__()
        self.audio_dim = audio_dim
        self.lyrics_dim = lyrics_dim
        self.latent_dim = latent_dim
        self.input_dim = audio_dim  # for compatibility with train_vae / evaluation
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.hidden_dims = hidden_dims

        # ----- Audio encoder -----
        audio_enc = []
        in_dim = audio_dim
        for h_dim in hidden_dims:
            audio_enc += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ]
            in_dim = h_dim
        self.audio_encoder = nn.Sequential(*audio_enc)

        # ----- Lyrics encoder (lighter) -----
        self.lyrics_encoder = nn.Sequential(
            nn.Linear(lyrics_dim, lyrics_hidden),
            nn.BatchNorm1d(lyrics_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(lyrics_hidden, lyrics_hidden),
            nn.ReLU(),
        )

        # ----- Fusion → latent -----
        fusion_dim = hidden_dims[-1] + lyrics_hidden
        self.fc_mu = nn.Linear(fusion_dim, latent_dim)
        self.fc_logvar = nn.Linear(fusion_dim, latent_dim)

        # ----- Decoder (audio reconstruction only) -----
        dec_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ]
            in_dim = h_dim
        dec_layers.append(nn.Linear(in_dim, audio_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, audio: torch.Tensor,
               lyrics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_a = self.audio_encoder(audio)
        h_l = self.lyrics_encoder(lyrics)
        h = torch.cat([h_a, h_l], dim=-1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, audio: torch.Tensor, lyrics: torch.Tensor) -> dict:
        """
        Args:
            audio: (B, audio_dim)
            lyrics: (B, lyrics_dim)

        Returns dict with: reconstruction (audio), mu, logvar, z
        """
        mu, logvar = self.encode(audio, lyrics)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return {"reconstruction": reconstruction, "mu": mu,
                "logvar": logvar, "z": z}

    def get_latent(self, audio: torch.Tensor,
                   lyrics: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(audio, lyrics)
        return mu


# ============================================================
# Loss function
# ============================================================

def vae_loss(recon_x: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             kl_weight: float = 1.0) -> dict:
    """
    VAE Loss = Reconstruction Loss + KL Divergence.

    Args:
        recon_x: reconstructed input
        x: original input
        mu: latent mean
        logvar: latent log-variance
        kl_weight: weight for KL term (beta for Beta-VAE)

    Returns:
        dict with total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    total_loss = recon_loss + kl_weight * kl_loss

    return {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
    }
