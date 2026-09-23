# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Iterable, Sequence

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam
from torch import nn

from src.vae.training import EpochMetrics, TrainingConfig, TrainingStatus, train_svi
from src.vae.utils.vae_plots import plot_llk

pyro.set_rng_seed(42)
torch.manual_seed(42)


class Encoder(nn.Module):
    """Encode observations into diagonal-Gaussian latent parameters."""

    def __init__(self, x_dim: int, z_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the latent location and scale for a batch of observations."""
        x = x.reshape(-1, self.x_dim)
        hidden = self.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = F.softplus(self.fc22(hidden)) + 1e-4
        return z_loc, z_scale


class Decoder(nn.Module):
    """Decode latent draws into Gaussian observation parameters."""

    def __init__(self, x_dim: int, z_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, x_dim)
        self.softplus = nn.Softplus()
        self.log_x_scale = nn.Parameter(torch.zeros(x_dim))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the decoded observation mean and log scale."""
        hidden = self.softplus(self.fc1(z))
        x_loc = self.fc21(hidden)
        return x_loc, self.log_x_scale.expand_as(x_loc)


class VAE(nn.Module):
    """Fit a Gaussian decoder with a Normal or multivariate Student-t prior."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int = 50,
        hidden_dim: int = 400,
        use_cuda: bool = False,
        prior_t_df: float = np.inf,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.encoder = Encoder(x_dim, z_dim, hidden_dim)
        self.decoder = Decoder(x_dim, z_dim, hidden_dim)

        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.prior_t_df = prior_t_df

    def fit(
        self,
        train_loader: Iterable[Sequence[torch.Tensor] | torch.Tensor],
        validation_loader: Iterable[Sequence[torch.Tensor] | torch.Tensor],
        *,
        config: TrainingConfig,
        device: str | torch.device | None = None,
    ) -> tuple[TrainingStatus, list[EpochMetrics]]:
        """Fit on ``(x,)`` batches; return the status and epoch metrics.

        Uses the shared ordinary ELBO training loop.
        """
        return train_svi(
            self,
            train_loader,
            validation_loader,
            config=config,
            device=device,
        )

    def _latent_prior(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> dist.TorchDistribution:
        """Return the configured latent prior with a vector-valued event."""
        if np.isinf(self.prior_t_df):
            return dist.Normal(loc, scale).to_event(1)

        scale_tril = torch.diag_embed(scale)
        return dist.MultivariateStudentT(
            self.prior_t_df,
            loc,
            scale_tril,
        )

    def model(
        self, x: torch.Tensor, annealing_factor: float | torch.Tensor = 1.0
    ) -> torch.Tensor:
        """Define the generative model for a batch of observations."""
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)

            with pyro.poutine.scale(scale=annealing_factor):
                z = pyro.sample("latent", self._latent_prior(z_loc, z_scale))

            x_loc, log_x_scale = self.decoder(z)
            x_scale = torch.exp(log_x_scale)

            pyro.sample(
                "obs",
                dist.Normal(x_loc, x_scale, validate_args=False).to_event(1),
                obs=x.reshape(-1, self.x_dim),
            )
            return x_loc

    def guide(
        self, x: torch.Tensor, annealing_factor: float | torch.Tensor = 1.0
    ) -> None:
        """Define a diagonal-Gaussian variational posterior for a batch."""
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder(x)
            with pyro.poutine.scale(scale=annealing_factor):
                pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def sample(self, n: int) -> torch.Tensor:
        """Draw `n` observations from the fitted generative model."""
        if n <= 0:
            raise ValueError("n must be positive.")

        parameter = next(self.parameters())
        with torch.no_grad():
            z_loc = torch.zeros(
                n,
                self.z_dim,
                dtype=parameter.dtype,
                device=parameter.device,
            )
            z_scale = torch.ones_like(z_loc)
            z = self._latent_prior(z_loc, z_scale).sample()
            x_loc, log_x_scale = self.decoder(z)
            return dist.Normal(x_loc, torch.exp(log_x_scale)).sample()

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Return a stochastic reconstruction for a batch of observations."""
        z_loc, z_scale = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        x_loc, _ = self.decoder(z)
        return x_loc

    def reconstruct_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return the deterministic reconstruction for a batch of observations."""
        z_loc, _ = self.encoder(x)
        x_loc, _ = self.decoder(z_loc)
        return x_loc


def train(
    data_loaders: Iterable,
    x_dim: int,
    hidden_dim: int = 50,
    z_dim: int = 2,
    beta: float | torch.Tensor = 1.0,
    annealing_start: float | torch.Tensor = 1.0,
    num_epochs: int = 30,
    test_frequency: int = 5,
    learning_rate: float = 1.0e-3,
    cuda: bool = False,
    jit: bool = False,
    prior_t_df: float = np.inf,
    verbose: bool = True,
) -> VAE:
    """Train the base VAE on loaders that yield batches shaped `(B, x_dim)`.

    Deprecated: instantiate :class:`VAE` and call :meth:`VAE.fit` instead.
    """
    warnings.warn(
        "vae.train() is deprecated; instantiate VAE and call VAE.fit() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    pyro.clear_param_store()

    train_loader, test_loader = data_loaders

    vae = VAE(
        x_dim=x_dim,
        use_cuda=cuda,
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        prior_t_df=prior_t_df,
    )

    optimizer = Adam({"lr": learning_rate})
    elbo = JitTrace_ELBO() if jit else Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    train_elbo = {}
    test_elbo = {}
    for epoch in range(num_epochs):
        progress = min((epoch + 1) / num_epochs, 1.0)
        annealing_factor = (1 - progress) * annealing_start + progress * beta

        epoch_loss = 0.0
        for (x,) in train_loader:
            if cuda:
                x = x.cuda()
            factor = (
                torch.as_tensor(annealing_factor, dtype=x.dtype, device=x.device)
                if jit
                else annealing_factor
            )
            epoch_loss += svi.step(x, factor)

        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo[epoch] = total_epoch_loss_train
        if verbose:
            print(
                "[epoch %03d]  average training loss: %.4f"
                % (epoch, total_epoch_loss_train)
            )

        if epoch % test_frequency == 0:
            test_loss = 0.0
            for (x,) in test_loader:
                if cuda:
                    x = x.cuda()
                factor = (
                    torch.as_tensor(annealing_factor, dtype=x.dtype, device=x.device)
                    if jit
                    else annealing_factor
                )
                test_loss += svi.evaluate_loss(x, factor)

            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo[epoch] = total_epoch_loss_test
            if verbose:
                print(
                    "[epoch %03d]  average test loss: %.4f"
                    % (epoch, total_epoch_loss_test)
                )
                plot_llk(train_elbo, test_elbo)

    return vae


if __name__ == "__main__":
    pass
