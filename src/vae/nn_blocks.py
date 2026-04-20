"""Shared building blocks for hierarchical VAE variants."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vae.utils.set_transformer import MeanPoolSetEncoder, SetEncoder


class ParentEncoder(nn.Module):
    """q(z1 | x1) — infer global latent from Index only."""

    def __init__(self, x1_dim: int, z1_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(x1_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, z1_dim)
        self.fc_scale = nn.Linear(hidden_dim, z1_dim)

    def forward(self, x1):
        x1 = x1.reshape(-1, self.fc1.in_features)
        hidden = F.softplus(self.fc1(x1))
        z1_loc = self.fc_loc(hidden)
        z1_scale = F.softplus(self.fc_scale(hidden)) + 1e-4
        return z1_loc, z1_scale


class ParentDecoder(nn.Module):
    """p(x1 | z1) — reconstruct Index from global latent."""

    def __init__(self, x1_dim: int, z1_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(z1_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, x1_dim)
        self.log_x_scale = nn.Parameter(torch.zeros(x1_dim))

    def forward(self, z):
        hidden = F.softplus(self.fc1(z))
        x_loc = self.fc_loc(hidden)
        return x_loc, self.log_x_scale.expand_as(x_loc)


class ChildPrior(nn.Module):
    """p(z2 | z1, c2) — learned conditional prior for child latent."""

    def __init__(self, z1_dim: int, c2_dim: int, z2_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(z1_dim + c2_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, z2_dim)
        self.fc_scale = nn.Linear(hidden_dim, z2_dim)

    def forward(self, z1, c2):
        inp = torch.cat([z1, c2], dim=-1)
        hidden = F.softplus(self.fc1(inp))
        z2_loc = self.fc_loc(hidden)
        z2_scale = F.softplus(self.fc_scale(hidden)) + 1e-4
        return z2_loc, z2_scale


class ChildEncoder(nn.Module):
    """q(z2 | x2, z1, c2) — shared child encoder across all names."""

    def __init__(
        self, x2_dim: int, z1_dim: int, c2_dim: int, z2_dim: int, hidden_dim: int
    ):
        super().__init__()
        self.fc1 = nn.Linear(x2_dim + z1_dim + c2_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, z2_dim)
        self.fc_scale = nn.Linear(hidden_dim, z2_dim)

    def forward(self, x2, z1, c2):
        inp = torch.cat([x2, z1, c2], dim=-1)
        hidden = F.softplus(self.fc1(inp))
        z2_loc = self.fc_loc(hidden)
        z2_scale = F.softplus(self.fc_scale(hidden)) + 1e-4
        return z2_loc, z2_scale


class ParentEncoderInput(nn.Module):
    """Compose the parent encoder input from x1 and an optional summary module."""

    def __init__(
        self,
        *,
        x1_dim: int,
        summary_encoder: nn.Module | None = None,
        summary_dim: int = 0,
    ):
        super().__init__()
        if summary_encoder is None and summary_dim != 0:
            raise ValueError(
                "summary_dim must be 0 when no summary_encoder is provided"
            )
        if summary_encoder is not None and summary_dim <= 0:
            raise ValueError(
                "summary_dim must be positive when summary_encoder is provided"
            )

        self.summary_encoder = summary_encoder
        self.summary_dim = summary_dim
        self.encoder_input_dim = x1_dim + self.summary_dim

    @property
    def uses_summary(self):
        return self.summary_encoder is not None

    def forward(self, x1, x2_set=None, child_mask=None):
        """Build input for the parent encoder, optionally including a set summary."""
        if self.summary_encoder is None:
            return x1

        if x2_set is not None:
            mask = child_mask.bool() if child_mask is not None else None
            summary = self.summary_encoder(x2_set, mask=mask)
        else:
            summary = x1.new_zeros(x1.shape[0], self.summary_dim)
        return torch.cat([x1, summary], dim=-1)


def build_full_posterior_encoder_input(
    *,
    x1_dim: int,
    x2_dim: int,
    hidden_dim: int,
    set_num_heads: int,
    set_num_inducing: int,
    set_encoder_type: Literal["attention", "mean_pool"] = "attention",
):
    """Create a parent encoder input composer backed by a set encoder.

    Parameters
    ----------
    x1_dim : int
        Dimensionality of the parent observation.
    x2_dim : int
        Dimensionality of each child observation.
    hidden_dim : int
        Internal hidden dimension for the set encoder.
    set_num_heads : int
        Number of attention heads (only used by attention encoder).
    set_num_inducing : int
        Number of inducing points (only used by attention encoder).
    set_encoder_type : {"attention", "mean_pool"}
        Type of set encoder to use:
        - "attention": Set Transformer with ISAB and PMA (Lee et al., 2019)
        - "mean_pool": MLP + mean pooling (Neural Statistician style)
    """
    encoder_cls = SetEncoder if set_encoder_type == "attention" else MeanPoolSetEncoder
    return ParentEncoderInput(
        x1_dim=x1_dim,
        summary_encoder=encoder_cls(
            input_dim=x2_dim,
            d_model=hidden_dim,
            output_dim=hidden_dim,
            num_heads=set_num_heads,
            num_inducing=set_num_inducing,
        ),
        summary_dim=hidden_dim,
    )
