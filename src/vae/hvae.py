"""
Hierarchical VAE: one shared global latent per date, shared child
encoder/decoder reused across all single names, no name-specific heads.

Training unit: one date t (not one (t, n) pair).

Data layout (batched by date):
    x1:         (B, x1_dim)           — Index surface change
    x2:         (B, N_max, x2_dim)    — padded single-name surfaces
    c2:         (B, N_max, c2_dim)    — padded single-name conditioning
    child_mask: (B, N_max)            — True where child n exists on date t

Generative model:
    z_t     ~ N(0, I)
    x1_t    ~ p(x1 | z_t)
    u_t_n   ~ p(u | z_t, c2_t_n)          (shared prior)
    x2_t_n  ~ p(x2 | u_t_n, z_t, c2_t_n)  (shared decoder)

Inference:
    q(z_t   | x1_t)
    q(u_t_n | x2_t_n, z_t, c2_t_n)
"""

from typing import Iterable

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from src.vae.utils.vae_plots import plot_llk

pyro.set_rng_seed(42)
torch.manual_seed(42)


class ParentEncoder(nn.Module):
    """q(z | x1) — infer global latent from Index only."""

    def __init__(self, x1_dim: int, z_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(x1_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, z_dim)
        self.fc_scale = nn.Linear(hidden_dim, z_dim)

    def forward(self, x1):
        x1 = x1.reshape(-1, self.fc1.in_features)
        hidden = F.softplus(self.fc1(x1))
        z_loc = self.fc_loc(hidden)
        z_scale = F.softplus(self.fc_scale(hidden)) + 1e-4
        return z_loc, z_scale


class ParentDecoder(nn.Module):
    """p(x1 | z) — reconstruct Index from global latent."""

    def __init__(self, x1_dim: int, z_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, x1_dim)
        self.log_x_scale = nn.Parameter(torch.zeros(x1_dim))

    def forward(self, z):
        hidden = F.softplus(self.fc1(z))
        x_loc = self.fc_loc(hidden)
        return x_loc, self.log_x_scale.expand_as(x_loc)


class ChildPrior(nn.Module):
    """p(u | z, c2) — learned conditional prior for child latent."""

    def __init__(self, z_dim: int, c2_dim: int, u_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(z_dim + c2_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, u_dim)
        self.fc_scale = nn.Linear(hidden_dim, u_dim)

    def forward(self, z, c2):
        inp = torch.cat([z, c2], dim=-1)
        hidden = F.softplus(self.fc1(inp))
        u_loc = self.fc_loc(hidden)
        u_scale = F.softplus(self.fc_scale(hidden)) + 1e-4
        return u_loc, u_scale


class ChildEncoder(nn.Module):
    """q(u | x2, z, c2) — shared child encoder across all names."""

    def __init__(
        self, x2_dim: int, z_dim: int, c2_dim: int, u_dim: int, hidden_dim: int
    ):
        super().__init__()
        self.fc1 = nn.Linear(x2_dim + z_dim + c2_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, u_dim)
        self.fc_scale = nn.Linear(hidden_dim, u_dim)

    def forward(self, x2, z, c2):
        inp = torch.cat([x2, z, c2], dim=-1)
        hidden = F.softplus(self.fc1(inp))
        u_loc = self.fc_loc(hidden)
        u_scale = F.softplus(self.fc_scale(hidden)) + 1e-4
        return u_loc, u_scale


class ChildDecoder(nn.Module):
    """p(x2 | u, z, c2) — shared child decoder across all names."""

    def __init__(
        self, x2_dim: int, u_dim: int, z_dim: int, c2_dim: int, hidden_dim: int
    ):
        super().__init__()
        self.fc1 = nn.Linear(u_dim + z_dim + c2_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, x2_dim)
        self.log_x_scale = nn.Parameter(torch.zeros(x2_dim))

    def forward(self, u, z, c2):
        inp = torch.cat([u, z, c2], dim=-1)
        hidden = F.softplus(self.fc1(inp))
        x_loc = self.fc_loc(hidden)
        return x_loc, self.log_x_scale.expand_as(x_loc)


class HierarchicalVAE(nn.Module):
    """
    One-to-many hierarchical VAE with a shared global latent per date
    and shared child networks reused across all single names.

    No name-specific heads — generalisation to unseen names comes from
    transferable features in c2.
    """

    def __init__(
        self,
        x1_dim: int,
        x2_dim: int,
        c2_dim: int,
        z_dim: int = 4,
        u_dim: int = 2,
        hidden_dim: int = 64,
        use_cuda: bool = False,
    ):
        super().__init__()
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        self.c2_dim = c2_dim
        self.z_dim = z_dim
        self.u_dim = u_dim

        # stage 1: Index
        self.index_encoder = ParentEncoder(x1_dim, z_dim, hidden_dim)
        self.index_decoder = ParentDecoder(x1_dim, z_dim, hidden_dim)

        # stage 2: shared child networks (one set for all names)
        self.child_prior = ChildPrior(z_dim, c2_dim, u_dim, hidden_dim)
        self.child_encoder = ChildEncoder(x2_dim, z_dim, c2_dim, u_dim, hidden_dim)
        self.child_decoder = ChildDecoder(x2_dim, u_dim, z_dim, c2_dim, hidden_dim)

        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def model(self, x1, x2, c2, child_mask, annealing_factor=1.0):
        """
        Generative model.

        Parameters
        ----------
        x1 : Tensor (B, x1_dim)
        x2 : Tensor (B, N_max, x2_dim)
        c2 : Tensor (B, N_max, c2_dim)
        child_mask : Tensor (B, N_max) — bool / float mask
        annealing_factor : float
        """
        pyro.module("index_decoder", self.index_decoder)
        pyro.module("child_prior", self.child_prior)
        pyro.module("child_decoder", self.child_decoder)

        B = x1.shape[0]
        N_max = x2.shape[1]

        # --- global latent z (dates plate) ---
        with pyro.plate("dates", B):
            z_loc = x1.new_zeros(B, self.z_dim)
            z_scale = x1.new_ones(B, self.z_dim)

            with pyro.poutine.scale(scale=annealing_factor):
                z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            x1_loc, log_x1_scale = self.index_decoder(z)
            x1_scale = torch.exp(log_x1_scale)
            pyro.sample(
                "obs_x1",
                dist.Normal(x1_loc, x1_scale, validate_args=False).to_event(1),
                obs=x1.reshape(B, self.x1_dim),
            )

        # --- children (flat plate with masking) ---
        z_exp = (
            z.unsqueeze(1).expand(B, N_max, self.z_dim).reshape(B * N_max, self.z_dim)
        )
        c2_flat = c2.reshape(B * N_max, self.c2_dim)
        x2_flat = x2.reshape(B * N_max, self.x2_dim)
        mask_flat = child_mask.reshape(B * N_max).bool()

        u_loc, u_scale = self.child_prior(z_exp, c2_flat)

        with pyro.plate("children", B * N_max):
            with pyro.poutine.mask(mask=mask_flat):
                with pyro.poutine.scale(scale=annealing_factor):
                    u = pyro.sample("u", dist.Normal(u_loc, u_scale).to_event(1))

                x2_loc, log_x2_scale = self.child_decoder(u, z_exp, c2_flat)
                x2_scale = torch.exp(log_x2_scale)
                pyro.sample(
                    "obs_x2",
                    dist.Normal(x2_loc, x2_scale, validate_args=False).to_event(1),
                    obs=x2_flat,
                )

    def guide(self, x1, x2, c2, child_mask, annealing_factor=1.0):
        """Variational posterior."""
        pyro.module("index_encoder", self.index_encoder)
        pyro.module("child_encoder", self.child_encoder)

        B = x1.shape[0]
        N_max = x2.shape[1]

        # --- global posterior q(z | x1) ---
        with pyro.plate("dates", B):
            z_loc, z_scale = self.index_encoder(x1)
            with pyro.poutine.scale(scale=annealing_factor):
                z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        # --- child posteriors q(u | x2, z, c2) ---
        z_exp = (
            z.unsqueeze(1).expand(B, N_max, self.z_dim).reshape(B * N_max, self.z_dim)
        )
        c2_flat = c2.reshape(B * N_max, self.c2_dim)
        x2_flat = x2.reshape(B * N_max, self.x2_dim)
        mask_flat = child_mask.reshape(B * N_max).bool()

        u_loc, u_scale = self.child_encoder(x2_flat, z_exp, c2_flat)

        with pyro.plate("children", B * N_max):
            with pyro.poutine.mask(mask=mask_flat):
                with pyro.poutine.scale(scale=annealing_factor):
                    pyro.sample("u", dist.Normal(u_loc, u_scale).to_event(1))

    def encode_z(self, x1):
        """Return posterior mean of global latent z from Index observation."""
        z_loc, _ = self.index_encoder(x1)
        return z_loc

    def predict_x2(self, x1, c2_target):
        """
        MAP prediction of a single-name surface from Index and conditioning.

        Parameters
        ----------
        x1 : Tensor (B, x1_dim)
        c2_target : Tensor (B, c2_dim)

        Returns
        -------
        x2_loc : Tensor (B, x2_dim)
        """
        z_loc = self.encode_z(x1)
        u_loc, _ = self.child_prior(z_loc, c2_target)
        x2_loc, _ = self.child_decoder(u_loc, z_loc, c2_target)
        return x2_loc

    def sample_x2(self, x1, c2_target, n_samples=1):
        """
        Stochastic samples of a single-name surface.

        Parameters
        ----------
        x1 : Tensor (B, x1_dim) or (x1_dim,)
        c2_target : Tensor (B, c2_dim) or (c2_dim,)
        n_samples : int

        Returns
        -------
        x2_samples : Tensor (B, n_samples, x2_dim) or (n_samples, x2_dim)
        """
        squeeze = x1.dim() == 1
        if squeeze:
            x1 = x1.unsqueeze(0)
            c2_target = c2_target.unsqueeze(0)

        z_loc = self.encode_z(x1)  # (B, z_dim)
        B = z_loc.shape[0]

        z_exp = z_loc.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
        c2_exp = (
            c2_target.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
        )

        u_loc, u_scale = self.child_prior(z_exp, c2_exp)
        u = dist.Normal(u_loc, u_scale).sample()
        x2_loc, _ = self.child_decoder(u, z_exp, c2_exp)

        x2_loc = x2_loc.reshape(B, n_samples, -1)
        if squeeze:
            x2_loc = x2_loc.squeeze(0)
        return x2_loc

    def reconstruct(self, x1, x2, c2):
        """
        Stochastic reconstruction of x1 and a single child x2.

        Parameters
        ----------
        x1 : Tensor (B, x1_dim)
        x2 : Tensor (B, x2_dim)  — single child (not padded)
        c2 : Tensor (B, c2_dim)
        """
        z_loc, z_scale = self.index_encoder(x1)
        z = dist.Normal(z_loc, z_scale).sample()
        x1_loc, _ = self.index_decoder(z)

        u_loc, u_scale = self.child_encoder(x2, z, c2)
        u = dist.Normal(u_loc, u_scale).sample()
        x2_loc, _ = self.child_decoder(u, z, c2)
        return x1_loc, x2_loc

    def reconstruct_map(self, x1, x2, c2):
        """MAP reconstruction — no sampling noise."""
        z_loc, _ = self.index_encoder(x1)
        x1_loc, _ = self.index_decoder(z_loc)

        u_loc, _ = self.child_encoder(x2, z_loc, c2)
        x2_loc, _ = self.child_decoder(u_loc, z_loc, c2)
        return x1_loc, x2_loc

    def counterfactual_prediction(self, x1, c2_new):
        """
        Predict x2 under a new conditioning without observing x2.

        Uses posterior mean for z and prior mean for u (deterministic).
        """
        z_loc = self.encode_z(x1)
        x1_loc, _ = self.index_decoder(z_loc)
        u_loc, _ = self.child_prior(z_loc, c2_new)
        x2_loc, _ = self.child_decoder(u_loc, z_loc, c2_new)
        return x1_loc, x2_loc

    def encode(self, x1, x2, c2):
        """Return posterior means for z (global) and u (child)."""
        z_loc, _ = self.index_encoder(x1)
        u_loc, _ = self.child_encoder(x2, z_loc, c2)
        return z_loc, u_loc


def train(
    data_loaders: Iterable,
    x1_dim: int,
    x2_dim: int,
    c2_dim: int,
    z_dim: int = 4,
    u_dim: int = 2,
    hidden_dim: int = 64,
    beta: float = 1.0,
    annealing_start: float = 1.0,
    num_epochs: int = 30,
    test_frequency: int = 5,
    learning_rate: float = 1e-3,
    cuda: bool = False,
):
    """
    Train a HierarchicalVAE.

    Each batch yields (x1, x2, c2, child_mask) with shapes
    (B, x1_dim), (B, N_max, x2_dim), (B, N_max, c2_dim), (B, N_max).
    """
    pyro.clear_param_store()

    train_loader, test_loader = data_loaders

    vae = HierarchicalVAE(
        x1_dim=x1_dim,
        x2_dim=x2_dim,
        c2_dim=c2_dim,
        z_dim=z_dim,
        u_dim=u_dim,
        hidden_dim=hidden_dim,
        use_cuda=cuda,
    )

    optimizer = Adam({"lr": learning_rate})
    elbo = Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    train_elbo = {}
    test_elbo = {}

    for epoch in range(num_epochs):
        progress = min((epoch + 1) / num_epochs, 1.0)
        annealing_factor = (1 - progress) * annealing_start + progress * beta

        epoch_loss = 0.0
        for batch in train_loader:
            x1, x2, c2, child_mask = batch
            if cuda:
                x1, x2, c2, child_mask = (
                    x1.cuda(),
                    x2.cuda(),
                    c2.cuda(),
                    child_mask.cuda(),
                )
            epoch_loss += svi.step(x1, x2, c2, child_mask, annealing_factor)

        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo[epoch] = total_epoch_loss_train
        print(
            "[epoch %03d]  average training loss: %.4f"
            % (epoch, total_epoch_loss_train)
        )

        if epoch % test_frequency == 0:
            test_loss = 0.0
            for batch in test_loader:
                x1, x2, c2, child_mask = batch
                if cuda:
                    x1, x2, c2, child_mask = (
                        x1.cuda(),
                        x2.cuda(),
                        c2.cuda(),
                        child_mask.cuda(),
                    )
                test_loss += svi.evaluate_loss(x1, x2, c2, child_mask, annealing_factor)

            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo[epoch] = total_epoch_loss_test
            print(
                "[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test)
            )
            plot_llk(train_elbo, test_elbo)

    return vae


if __name__ == "__main__":
    pass
