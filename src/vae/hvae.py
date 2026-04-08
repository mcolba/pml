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


# ---------------------------------------------------------------------------
# Stage-1 components  (standard VAE for x1)
# ---------------------------------------------------------------------------


class Encoder1(nn.Module):
    """q(z1 | x1)"""

    def __init__(self, x1_dim, z1_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(x1_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, z1_dim)
        self.fc_scale = nn.Linear(hidden_dim, z1_dim)
        self.softplus = nn.Softplus()

    def forward(self, x1):
        x1 = x1.reshape(-1, self.fc1.in_features)
        hidden = self.softplus(self.fc1(x1))
        z_loc = self.fc_loc(hidden)
        z_scale = F.softplus(self.fc_scale(hidden)) + 1e-4
        return z_loc, z_scale


class Decoder1(nn.Module):
    """p(x1 | z1)"""

    def __init__(self, x1_dim, z1_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z1_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, x1_dim)
        self.softplus = nn.Softplus()
        self.log_x_scale = nn.Parameter(torch.zeros(x1_dim))

    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        x_loc = self.fc_loc(hidden)
        return x_loc, self.log_x_scale.expand_as(x_loc)


# ---------------------------------------------------------------------------
# Conditional prior for z2  (learned p(z2 | z1, C))
# ---------------------------------------------------------------------------


class PriorZ2(nn.Module):
    """p(z2 | z1, C) — learned conditional prior for the second latent layer."""

    def __init__(self, z1_dim, c_dim, z2_dim, hidden_dim):
        super().__init__()
        in_dim = z1_dim + c_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, z2_dim)
        self.fc_scale = nn.Linear(hidden_dim, z2_dim)
        self.softplus = nn.Softplus()

    def forward(self, z1, c):
        z1 = z1.reshape(-1, z1.shape[-1])
        c = c.reshape(-1, c.shape[-1])
        inp = torch.cat([z1, c], dim=-1)
        hidden = self.softplus(self.fc1(inp))
        z_loc = self.fc_loc(hidden)
        z_scale = F.softplus(self.fc_scale(hidden)) + 1e-4
        return z_loc, z_scale


# ---------------------------------------------------------------------------
# Stage-2 components  (conditional VAE for x2 given z1 and C)
# ---------------------------------------------------------------------------


class Encoder2(nn.Module):
    """q(z2 | x2, z1, C)"""

    def __init__(self, x2_dim, z1_dim, c_dim, z2_dim, hidden_dim):
        super().__init__()
        in_dim = x2_dim + z1_dim + c_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, z2_dim)
        self.fc_scale = nn.Linear(hidden_dim, z2_dim)
        self.softplus = nn.Softplus()
        self.x2_dim = x2_dim

    def forward(self, x2, z1, c):
        x2 = x2.reshape(-1, self.x2_dim)
        z1 = z1.reshape(-1, z1.shape[-1])
        c = c.reshape(-1, c.shape[-1])
        xzc = torch.cat([x2, z1, c], dim=-1)
        hidden = self.softplus(self.fc1(xzc))
        z_loc = self.fc_loc(hidden)
        z_scale = F.softplus(self.fc_scale(hidden)) + 1e-4
        return z_loc, z_scale


class Decoder2(nn.Module):
    """p(x2 | z2, z1, C)"""

    def __init__(self, x2_dim, z1_dim, z2_dim, c_dim, hidden_dim):
        super().__init__()
        in_dim = z2_dim + z1_dim + c_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, x2_dim)
        self.softplus = nn.Softplus()
        self.log_x_scale = nn.Parameter(torch.zeros(x2_dim))

    def forward(self, z2, z1, c):
        inp = torch.cat([z2, z1, c], dim=-1)
        hidden = self.softplus(self.fc1(inp))
        x_loc = self.fc_loc(hidden)
        return x_loc, self.log_x_scale.expand_as(x_loc)


# ---------------------------------------------------------------------------
# Hierarchical VAE
# ---------------------------------------------------------------------------


class HierarchicalVAE(nn.Module):
    """
    Two-level hierarchical VAE with a learned conditional prior on z2.

    Generative model:
        z1  ~ p(z1)                    (standard normal)
        x1  ~ p(x1 | z1)
        z2  ~ p(z2 | z1, c)            (learned conditional prior)
        x2  ~ p(x2 | z2, z1, c)

    Inference (guide):
        q(z1 | x1)                     z1 is inferred from x1 alone
        q(z2 | x2, z1, c)              z2 posterior uses observed x2

    The two stages share a single Pyro model / guide so that the whole
    hierarchy is optimised end-to-end with a single ELBO.  Although q(z1|x1)
    only reads x1, the stage-2 likelihood p(x2|z2,z1,c) shapes the learned
    z1 representation through gradients that back-propagate from the ELBO
    into the shared z1 sample.

    Predicting x2 from x1 alone (without observing x2):
        1. Encode  z1 ~ q(z1 | x1)
        2. Sample  z2 ~ p(z2 | z1, c)   (conditional prior, no x2 needed)
        3. Decode  x2 ~ p(x2 | z2, z1, c)
    See :meth:`sample_x2` and :meth:`predict_x2`.
    """

    def __init__(
        self,
        x1_dim: int,
        x2_dim: int,
        c_dim: int,
        z1_dim: int = 2,
        z2_dim: int = 2,
        hidden_dim: int = 50,
        use_cuda: bool = False,
    ):
        super().__init__()
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        self.c_dim = c_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim

        # stage-1
        self.encoder1 = Encoder1(x1_dim, z1_dim, hidden_dim)
        self.decoder1 = Decoder1(x1_dim, z1_dim, hidden_dim)

        # stage-2
        self.encoder2 = Encoder2(x2_dim, z1_dim, c_dim, z2_dim, hidden_dim)
        self.decoder2 = Decoder2(x2_dim, z1_dim, z2_dim, c_dim, hidden_dim)
        self.prior_z2 = PriorZ2(z1_dim, c_dim, z2_dim, hidden_dim)

        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    # ---- Pyro model / guide ------------------------------------------------

    def model(self, x1, x2, c2, annealing_factor=1.0):
        pyro.module("decoder1", self.decoder1)
        pyro.module("decoder2", self.decoder2)
        pyro.module("prior_z2", self.prior_z2)

        batch_size = x1.shape[0]

        with pyro.plate("data", batch_size):
            # --- Stage-1 prior & likelihood ---
            z1_loc = torch.zeros(
                batch_size, self.z1_dim, dtype=x1.dtype, device=x1.device
            )
            z1_scale = torch.ones(
                batch_size, self.z1_dim, dtype=x1.dtype, device=x1.device
            )

            with pyro.poutine.scale(scale=annealing_factor):
                z1 = pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))

            x1_loc, log_x1_scale = self.decoder1(z1)
            x1_scale = torch.exp(log_x1_scale)

            pyro.sample(
                "obs_x1",
                dist.Normal(x1_loc, x1_scale, validate_args=False).to_event(1),
                obs=x1.reshape(-1, self.x1_dim),
            )

            # --- Stage-2 conditional prior & likelihood ---
            z2_loc, z2_scale = self.prior_z2(z1, c2)

            with pyro.poutine.scale(scale=annealing_factor):
                z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

            x2_loc, log_x2_scale = self.decoder2(z2, z1, c2)
            x2_scale = torch.exp(log_x2_scale)

            pyro.sample(
                "obs_x2",
                dist.Normal(x2_loc, x2_scale, validate_args=False).to_event(1),
                obs=x2.reshape(-1, self.x2_dim),
            )

        return x1_loc, x2_loc

    def guide(self, x1, x2, c2, annealing_factor=1.0):
        pyro.module("encoder1", self.encoder1)
        pyro.module("encoder2", self.encoder2)

        with pyro.plate("data", x1.shape[0]):
            # --- Stage-1 posterior ---
            z1_loc, z1_scale = self.encoder1(x1)
            with pyro.poutine.scale(scale=annealing_factor):
                z1 = pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))

            # --- Stage-2 posterior (conditioned on z1) ---
            z2_loc, z2_scale = self.encoder2(x2, z1, c2)
            with pyro.poutine.scale(scale=annealing_factor):
                pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

    # ---- Inference helpers --------------------------------------------------

    def reconstruct(self, x1, x2, c2):
        """Stochastic reconstruction of both x1 and x2."""
        z1_loc, z1_scale = self.encoder1(x1)
        z1 = dist.Normal(z1_loc, z1_scale).sample()
        x1_loc, _ = self.decoder1(z1)

        z2_loc, z2_scale = self.encoder2(x2, z1, c2)
        z2 = dist.Normal(z2_loc, z2_scale).sample()
        x2_loc, _ = self.decoder2(z2, z1, c2)
        return x1_loc, x2_loc

    def reconstruct_map(self, x1, x2, c2):
        """MAP (mean) reconstruction – no sampling noise."""
        z1_loc, _ = self.encoder1(x1)
        x1_loc, _ = self.decoder1(z1_loc)

        z2_loc, _ = self.encoder2(x2, z1_loc, c2)
        x2_loc, _ = self.decoder2(z2_loc, z1_loc, c2)
        return x1_loc, x2_loc

    def counterfactual_prediction(self, x1, c2_new):
        """
        Predict x2 under a new condition without observing x2.

        Infers z1 from x1, draws z2 from the conditional prior
        p(z2 | z1, c2_new), and decodes x2.  Uses the prior *mean*
        for z2 (MAP estimate), so the output is deterministic.
        """
        z1_loc, _ = self.encoder1(x1)
        z2_loc, _ = self.prior_z2(z1_loc, c2_new)
        x2_loc, _ = self.decoder2(z2_loc, z1_loc, c2_new)
        x1_loc, _ = self.decoder1(z1_loc)
        return x1_loc, x2_loc

    def predict_x2(self, x1, c2):
        """
        MAP prediction of x2 from x1 and condition c2 (no x2 observed).

        Uses the posterior mean for z1 and the conditional prior mean
        for z2, so the output is fully deterministic.
        """
        z1_loc, _ = self.encoder1(x1)
        z2_loc, _ = self.prior_z2(z1_loc, c2)
        x2_loc, _ = self.decoder2(z2_loc, z1_loc, c2)
        return x2_loc

    def encode(self, x1, x2, c2):
        """Return posterior means for both latent layers."""
        z1_loc, _ = self.encoder1(x1)
        z2_loc, _ = self.encoder2(x2, z1_loc, c2)
        return z1_loc, z2_loc

    def sample_x2(self, x1, c2, n_samples=1):
        """
        Generate x2 samples given observed x1 and condition c2.

        Supports batched inputs: x1 of shape (B, x1_dim) and c2 of
        shape (B, c2_dim).  Returns shape (B, n_samples, x2_dim).

        For a single sample (no batch dim), returns (n_samples, x2_dim).
        """
        squeeze = x1.dim() == 1
        if squeeze:
            x1 = x1.unsqueeze(0)
            c2 = c2.unsqueeze(0)

        z1_loc, _ = self.encoder1(x1)  # (B, z1_dim)
        B = z1_loc.shape[0]

        # Repeat each sample n_samples times: (B*n_samples, dim)
        z1_exp = z1_loc.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
        c2_exp = c2.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)

        z2_loc, z2_scale = self.prior_z2(z1_exp, c2_exp)
        z2 = dist.Normal(z2_loc, z2_scale).sample()
        x2_loc, _ = self.decoder2(z2, z1_exp, c2_exp)

        x2_loc = x2_loc.reshape(B, n_samples, -1)
        if squeeze:
            x2_loc = x2_loc.squeeze(0)  # (n_samples, x2_dim)
        return x2_loc


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    data_loaders: Iterable,
    x1_dim: int,
    x2_dim: int,
    c_dim: int,
    z1_dim: int = 2,
    z2_dim: int = 2,
    hidden_dim: int = 50,
    beta: float = 1.0,
    annealing_start: float = 1.0,
    num_epochs: int = 30,
    test_frequency: int = 5,
    learning_rate: float = 1e-3,
    cuda: bool = False,
):
    pyro.clear_param_store()

    train_loader, test_loader = data_loaders

    vae = HierarchicalVAE(
        x1_dim=x1_dim,
        x2_dim=x2_dim,
        c_dim=c_dim,
        z1_dim=z1_dim,
        z2_dim=z2_dim,
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
            x1, _c1, x2, c2 = batch
            if cuda:
                x1, x2, c2 = x1.cuda(), x2.cuda(), c2.cuda()
            epoch_loss += svi.step(x1, x2, c2, annealing_factor)

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
                x1, _c1, x2, c2 = batch
                if cuda:
                    x1, x2, c2 = x1.cuda(), x2.cuda(), c2.cuda()
                test_loss += svi.evaluate_loss(x1, x2, c2, annealing_factor)

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
