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
# Components
# ---------------------------------------------------------------------------


class ConditionalPrior(nn.Module):
    """p(z | x1, c) — learned conditional prior."""

    def __init__(self, x1_dim, c_dim, z_dim, hidden_dim) -> None:
        super().__init__()
        in_dim = x1_dim + c_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, z_dim)
        self.fc_scale = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(
        self, x1: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the conditional prior location and scale for a batch."""
        x1 = x1.reshape(-1, x1.shape[-1])
        c = c.reshape(-1, c.shape[-1])
        inp = torch.cat([x1, c], dim=-1)
        hidden = self.softplus(self.fc1(inp))
        z_loc = self.fc_loc(hidden)
        z_scale = F.softplus(self.fc_scale(hidden)) + 1e-4
        return z_loc, z_scale


class Encoder(nn.Module):
    """q(z | x2, x1, c) — encode child observations with observed context."""

    def __init__(self, x2_dim, x1_dim, c_dim, z_dim, hidden_dim) -> None:
        super().__init__()
        in_dim = x2_dim + x1_dim + c_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, z_dim)
        self.fc_scale = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()
        self.x2_dim = x2_dim

    def forward(
        self, x2: torch.Tensor, x1: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the posterior location and scale for a batch."""
        x2 = x2.reshape(-1, self.x2_dim)
        x1 = x1.reshape(-1, x1.shape[-1])
        c = c.reshape(-1, c.shape[-1])
        inp = torch.cat([x2, x1, c], dim=-1)
        hidden = self.softplus(self.fc1(inp))
        z_loc = self.fc_loc(hidden)
        z_scale = F.softplus(self.fc_scale(hidden)) + 1e-4
        return z_loc, z_scale


class Decoder(nn.Module):
    """p(x2 | z, x1, c) — decode child observations from latent and context."""

    def __init__(self, x2_dim, z_dim, x1_dim, c_dim, hidden_dim) -> None:
        super().__init__()
        in_dim = z_dim + x1_dim + c_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, x2_dim)
        self.softplus = nn.Softplus()
        self.log_x_scale = nn.Parameter(torch.zeros(x2_dim))

    def forward(
        self, z: torch.Tensor, x1: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the decoded observation mean and log scale."""
        inp = torch.cat([z, x1, c], dim=-1)
        hidden = self.softplus(self.fc1(inp))
        x_loc = self.fc_loc(hidden)
        return x_loc, self.log_x_scale.expand_as(x_loc)


# ---------------------------------------------------------------------------
# Hierarchical VAE  (d-separated: x1 is observed conditioning, not modeled)
# ---------------------------------------------------------------------------


class HierarchicalVAE(nn.Module):
    """
    Conditional VAE for x2 given observed x1 and conditioning c.

    By removing the shared latent z1, the model treats x1 as an observed
    conditioning variable rather than modeling it generatively.  All
    information from x1 flows through (x1, c) → z → x2, with x1 also
    feeding the decoder directly for expressiveness.

    Generative model:
        z   ~ p(z | x1, c)              (learned conditional prior)
        x2  ~ p(x2 | z, x1, c)

    Inference (guide):
        q(z | x2, x1, c)

    Predicting x2 from x1 alone (without observing x2):
        1. Compute  z_loc = E[z | x1, c]  from the conditional prior
        2. Decode   x2 ~ p(x2 | z_loc, x1, c)
    See :meth:`sample_x2` and :meth:`predict_x2`.
    """

    def __init__(
        self,
        x1_dim: int,
        x2_dim: int,
        c_dim: int,
        z_dim: int = 2,
        hidden_dim: int = 50,
        use_cuda: bool = False,
    ) -> None:
        super().__init__()
        self.x1_dim: int = x1_dim
        self.x2_dim: int = x2_dim
        self.c_dim: int = c_dim
        self.z_dim: int = z_dim

        self.encoder: Encoder = Encoder(x2_dim, x1_dim, c_dim, z_dim, hidden_dim)
        self.decoder: Decoder = Decoder(x2_dim, z_dim, x1_dim, c_dim, hidden_dim)
        self.prior_z: ConditionalPrior = ConditionalPrior(x1_dim, c_dim, z_dim, hidden_dim)

        self.use_cuda: bool = use_cuda
        if use_cuda:
            self.cuda()

    # ---- Pyro model / guide ------------------------------------------------

    def model(
        self, x1: torch.Tensor, x2: torch.Tensor, c2: torch.Tensor, annealing_factor: float = 1.0
    ) -> torch.Tensor:
        """Define the conditional hierarchical generative model for a batch."""
        pyro.module("decoder", self.decoder)
        pyro.module("prior_z", self.prior_z)

        batch_size = x1.shape[0]

        with pyro.plate("data", batch_size):
            # --- Conditional prior & likelihood ---
            z_loc, z_scale = self.prior_z(x1, c2)

            with pyro.poutine.scale(scale=annealing_factor):
                z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            x2_loc, log_x2_scale = self.decoder(z, x1, c2)
            x2_scale = torch.exp(log_x2_scale)

            pyro.sample(
                "obs_x2",
                dist.Normal(x2_loc, x2_scale, validate_args=False).to_event(1),
                obs=x2.reshape(-1, self.x2_dim),
            )

        return x2_loc

    def guide(
        self, x1: torch.Tensor, x2: torch.Tensor, c2: torch.Tensor, annealing_factor: float = 1.0
    ) -> None:
        """Define the conditional hierarchical variational posterior for a batch."""
        pyro.module("encoder", self.encoder)

        with pyro.plate("data", x1.shape[0]):
            z_loc, z_scale = self.encoder(x2, x1, c2)
            with pyro.poutine.scale(scale=annealing_factor):
                pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

    # ---- Inference helpers --------------------------------------------------

    def reconstruct(
        self, x1: torch.Tensor, x2: torch.Tensor, c2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a stochastic reconstruction of x1 and one child observation."""
        z_loc, z_scale = self.encoder(x2, x1, c2)
        z = dist.Normal(z_loc, z_scale).sample()
        x2_loc, _ = self.decoder(z, x1, c2)
        return x1, x2_loc

    def reconstruct_map(
        self, x1: torch.Tensor, x2: torch.Tensor, c2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the deterministic reconstruction of x1 and one child observation."""
        z_loc, _ = self.encoder(x2, x1, c2)
        x2_loc, _ = self.decoder(z_loc, x1, c2)
        return x1, x2_loc

    def counterfactual_prediction(
        self, x1: torch.Tensor, c2_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict a child observation under a new condition without observing that child."""
        z_loc, _ = self.prior_z(x1, c2_new)
        x2_loc, _ = self.decoder(z_loc, x1, c2_new)
        return x1, x2_loc

    def predict_x2(self, x1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        """Return the MAP child prediction from x1 and conditioning."""
        z_loc, _ = self.prior_z(x1, c2)
        x2_loc, _ = self.decoder(z_loc, x1, c2)
        return x2_loc

    def encode(self, x1: torch.Tensor, x2: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        """Return the posterior mean of the latent for a batch."""
        z_loc, _ = self.encoder(x2, x1, c2)
        return z_loc

    def sample_x2(
        self, x1: torch.Tensor, c2: torch.Tensor, n_samples: int = 1
    ) -> torch.Tensor:
        """Sample child observations conditioned on x1 and c2."""
        squeeze = x1.dim() == 1
        if squeeze:
            x1 = x1.unsqueeze(0)
            c2 = c2.unsqueeze(0)

        B = x1.shape[0]

        # Repeat each sample n_samples times: (B*n_samples, dim)
        x1_exp = x1.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
        c2_exp = c2.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)

        z_loc, z_scale = self.prior_z(x1_exp, c2_exp)
        z = dist.Normal(z_loc, z_scale).sample()
        x2_loc, _ = self.decoder(z, x1_exp, c2_exp)

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
    z_dim: int = 2,
    hidden_dim: int = 50,
    beta: float = 1.0,
    annealing_start: float = 1.0,
    num_epochs: int = 30,
    test_frequency: int = 5,
    learning_rate: float = 1e-3,
    cuda: bool = False,
) -> HierarchicalVAE:
    """Train the simplified hierarchical VAE on loaders that yield `(x1, _, x2, c2)` batches."""
    pyro.clear_param_store()

    train_loader, test_loader = data_loaders

    vae = HierarchicalVAE(
        x1_dim=x1_dim,
        x2_dim=x2_dim,
        c_dim=c_dim,
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        use_cuda=cuda,
    )

    optimizer = Adam({"lr": learning_rate})
    elbo = Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    train_elbo = {}
    test_elbo = {}

    for epoch in range(num_epochs):
        progress: float = min((epoch + 1) / num_epochs, 1.0)
        annealing_factor: float = (1 - progress) * annealing_start + progress * beta

        epoch_loss = 0.0
        for batch in train_loader:
            x1, _c1, x2, c2 = batch
            if cuda:
                x1, x2, c2 = x1.cuda(), x2.cuda(), c2.cuda()
            epoch_loss += svi.step(x1, x2, c2, annealing_factor)

        normalizer_train: int = len(train_loader.dataset)
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

            normalizer_test: int = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo[epoch] = total_epoch_loss_test
            print(
                "[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test)
            )
            plot_llk(train_elbo, test_elbo)

    return vae


if __name__ == "__main__":
    pass
