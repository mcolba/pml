"""
Hierarchical VAE: one shared global latent per date, shared child
encoder/decoder reused across all single names, no name-specific heads.

Training unit: one date t (not one (t, n) pair).

Plate structure (nested)::

    dates    (dim=-2, size B)        — outer plate, one slot per date
      children (dim=-1, size N_max)  — inner plate, one slot per name

Data layout (batched by date):
    x1:         (B, x1_dim)           — Index surface change
    x2:         (B, N_max, x2_dim)    — padded single-name surfaces
    c2:         (B, N_max, c2_dim)    — padded single-name conditioning
    child_mask: (B, N_max)            — True where child n exists on date t

Generative model:
    z1_t     ~ N(0, I)                        [dates plate]
    x1_t     ~ p(x1 | z1_t)                   [dates plate]
    z2_t_n   ~ p(z2 | z1_t, c2_t_n)           [dates × children plates]
    x2_t_n   ~ p(x2 | z2_t_n, z1_t, c2_t_n)  [dates × children plates]

Inference:
    q(z1_t   | x1_t [, {x2_t_n, c2_t_n}])      [dates plate]
    q(z2_t_n | x2_t_n, z1_t, c2_t_n)           [dates × children plates]
"""

import warnings
from collections.abc import Iterable, Sequence

import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch import nn

from src.vae.nn_blocks import (
    ChildEncoder,
    ChildPrior,
    ParentDecoder,
    ParentEncoder,
    ParentEncoderInput,
)
from src.vae.training import EpochMetrics, TrainingConfig, TrainingStatus, train_svi
from src.vae.utils.vae_plots import plot_llk

pyro.set_rng_seed(42)
torch.manual_seed(42)


class ChildDecoder(nn.Module):
    """p(x2 | z2, z1, c2) — shared child decoder across all names."""

    def __init__(
        self, x2_dim: int, z2_dim: int, z1_dim: int, c2_dim: int, hidden_dim: int
    ):
        super().__init__()
        self.fc1 = nn.Linear(z2_dim + z1_dim + c2_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, x2_dim)
        self.log_x_scale = nn.Parameter(torch.zeros(x2_dim))

    def forward(
        self, z2: torch.Tensor, z1: torch.Tensor, c2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([z2, z1, c2], dim=-1)
        hidden = F.softplus(self.fc1(inp))
        x_loc = self.fc_loc(hidden)
        return x_loc, self.log_x_scale.expand_as(x_loc)


class HierarchicalVAE(nn.Module):
    """
    One-to-many hierarchical VAE with a shared global latent per date
    and shared child networks reused across all single names.

    No name-specific heads — generalisation to unseen names comes from
    transferable features in c2.

    Parent inference uses either x1 alone or a paired observed-child context.
    Context consists of x2_set (B, N, x2_dim), c2_set (B, N, c2_dim),
    and child_mask (B, N), supplied together. Models without a set encoder
    reject context in prediction and reconstruction helpers. For models with
    a set encoder, omitted context and an all-false mask both use a zero summary.
    """

    def __init__(
        self,
        x1_dim: int,
        x2_dim: int,
        c2_dim: int,
        z1_dim: int = 3,
        z2_dim: int = 3,
        hidden_dim: int = 64,
        set_encoder: nn.Module | None = None,
        z1_input: ParentEncoderInput | None = None,
        use_cuda: bool = False,
    ):
        """
        Parameters
        ----------
        x1_dim : int
            Dimensionality of parent observation (Index surface).
        x2_dim : int
            Dimensionality of child observations (single-name surfaces).
        c2_dim : int
            Dimensionality of child conditioning vector.
        z1_dim : int
            Dimensionality of global latent z1.
        z2_dim : int
            Dimensionality of child latent z2.
        hidden_dim : int
            Hidden dimension for encoder/decoder networks.
        set_encoder : nn.Module, optional
            Set encoder for aggregating child pairs ``(x2, c2)`` into a
            summary. If provided, the global encoder receives
            ``[x1, set_summary]``. Must have an `output_dim` attribute
            specifying output size. If None, only x1 is used for the global
            encoder.
        z1_input : ParentEncoderInput, optional
            Full control over parent encoder input composition.
            If provided, overrides `set_encoder`.
        use_cuda : bool
            Move model to GPU.
        """
        super().__init__()
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        self.c2_dim = c2_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim

        # Build z1_input: priority is z1_input > set_encoder > x1 only
        if z1_input is not None:
            pass  # use provided z1_input
        elif set_encoder is not None:
            if not hasattr(set_encoder, "output_dim"):
                raise ValueError("set_encoder must have an 'output_dim' attribute")
            z1_input = ParentEncoderInput(
                x1_dim=x1_dim,
                summary_encoder=set_encoder,
                summary_dim=set_encoder.output_dim,
                x2_dim=x2_dim,
                c2_dim=c2_dim,
            )
        else:
            z1_input = ParentEncoderInput(x1_dim=x1_dim)

        if z1_input.x1_dim != x1_dim:
            raise ValueError("z1_input x1_dim must match the model x1_dim")
        z1_input.bind_child_dims(x2_dim, c2_dim)
        self.z1_input = z1_input
        self.use_full_posterior = self.z1_input.uses_summary
        self.set_encoder = self.z1_input.summary_encoder

        # stage 1: Index
        encoder_input_dim = self.z1_input.encoder_input_dim

        self.index_encoder = ParentEncoder(encoder_input_dim, z1_dim, hidden_dim)
        self.index_decoder = ParentDecoder(x1_dim, z1_dim, hidden_dim)

        # stage 2: shared child networks (one set for all names)
        self.child_prior = ChildPrior(z1_dim, c2_dim, z2_dim, hidden_dim)
        self.child_encoder = ChildEncoder(x2_dim, z1_dim, c2_dim, z2_dim, hidden_dim)
        self.child_decoder = ChildDecoder(x2_dim, z2_dim, z1_dim, c2_dim, hidden_dim)

        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def fit(
        self,
        train_loader: Iterable[Sequence[torch.Tensor] | torch.Tensor],
        validation_loader: Iterable[Sequence[torch.Tensor] | torch.Tensor],
        *,
        config: TrainingConfig,
        device: str | torch.device | None = None,
    ) -> tuple[TrainingStatus, list[EpochMetrics]]:
        """Fit on hierarchical batches; return the status and epoch metrics.

        Shapes are ``(B, x1_dim)``, ``(B, N, x2_dim)``,
        ``(B, N, c2_dim)``, and ``(B, N)``. Uses ordinary ELBO.
        """
        return train_svi(
            self,
            train_loader,
            validation_loader,
            config=config,
            device=device,
        )

    def model(self, x1, x2, c2, child_mask, annealing_factor=1.0):
        """
        Generative model with nested plates.

        Plate structure
        ---------------
        dates    (dim=-2, size B)       — outer plate, one slot per date
          children (dim=-1, size N_max) — inner plate, one slot per name

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

        with pyro.plate("dates", B, dim=-2):
            # ── global latent z1 ──
            # shape (B, 1, z1_dim): dim-2 = B (dates), dim-1 = 1 (no children)
            z1_loc = x1.new_zeros(B, 1, self.z1_dim)
            z1_scale = x1.new_ones(B, 1, self.z1_dim)

            with pyro.poutine.scale(scale=annealing_factor):
                z1 = pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))
            # z1: (B, 1, z1_dim)

            # ── reconstruct x1 (date-level observation) ──
            z1_2d = z1.squeeze(-2)  # (B, z1_dim)
            x1_loc, log_x1_scale = self.index_decoder(z1_2d)  # each (B, x1_dim)
            x1_scale = torch.exp(log_x1_scale)
            pyro.sample(
                "obs_x1",
                dist.Normal(
                    x1_loc.unsqueeze(-2),  # (B, 1, x1_dim)
                    x1_scale.unsqueeze(-2),
                    validate_args=False,
                ).to_event(1),
                obs=x1.unsqueeze(-2),  # (B, 1, x1_dim)
            )

            # ── children (nested plate) ──
            z1_exp = z1.expand(B, N_max, self.z1_dim)  # broadcast
            # flatten for NN forward passes
            z1_flat = z1_exp.reshape(B * N_max, self.z1_dim)
            c2_flat = c2.reshape(B * N_max, self.c2_dim)

            z2_prior_loc, z2_prior_scale = self.child_prior(z1_flat, c2_flat)
            z2_prior_loc = z2_prior_loc.reshape(B, N_max, self.z2_dim)
            z2_prior_scale = z2_prior_scale.reshape(B, N_max, self.z2_dim)

            with pyro.plate("children", N_max, dim=-1):
                with pyro.poutine.mask(mask=child_mask.bool()):
                    with pyro.poutine.scale(scale=annealing_factor):
                        z2 = pyro.sample(
                            "z2",
                            dist.Normal(z2_prior_loc, z2_prior_scale).to_event(1),
                        )
                    # z2: (B, N_max, z2_dim)

                    z2_flat = z2.reshape(B * N_max, self.z2_dim)
                    x2_loc_flat, log_x2_scale_flat = self.child_decoder(
                        z2_flat, z1_flat, c2_flat
                    )
                    x2_loc = x2_loc_flat.reshape(B, N_max, self.x2_dim)
                    x2_scale = torch.exp(log_x2_scale_flat).reshape(
                        B, N_max, self.x2_dim
                    )
                    pyro.sample(
                        "obs_x2",
                        dist.Normal(x2_loc, x2_scale, validate_args=False).to_event(1),
                        obs=x2,
                    )

    def guide(self, x1, x2, c2, child_mask, annealing_factor=1.0):
        """Variational posterior (nested-plate layout matching the model)."""
        pyro.module("index_encoder", self.index_encoder)
        pyro.module("child_encoder", self.child_encoder)
        if self.set_encoder is not None:
            pyro.module("set_encoder", self.set_encoder)

        B = x1.shape[0]
        N_max = x2.shape[1]

        # Compute z1 encoder input (deterministic, outside plates)
        if self.use_full_posterior:
            z1_encoder_input = self.z1_input(x1, x2, child_mask, c2_set=c2)
        else:
            z1_encoder_input = self.z1_input(x1)

        with pyro.plate("dates", B, dim=-2):
            # ── global posterior q(z1 | x1 [, {x2}]) ──
            z1_loc, z1_scale = self.index_encoder(z1_encoder_input)
            with pyro.poutine.scale(scale=annealing_factor):
                z1 = pyro.sample(
                    "z1",
                    dist.Normal(
                        z1_loc.unsqueeze(-2),  # (B, 1, z1_dim)
                        z1_scale.unsqueeze(-2),
                    ).to_event(1),
                )
            # z1: (B, 1, z1_dim)

            # ── child posteriors q(z2 | x2, z1, c2) ──
            z1_exp = z1.expand(B, N_max, self.z1_dim)
            z1_flat = z1_exp.reshape(B * N_max, self.z1_dim)
            c2_flat = c2.reshape(B * N_max, self.c2_dim)
            x2_flat = x2.reshape(B * N_max, self.x2_dim)

            z2_loc, z2_scale = self.child_encoder(x2_flat, z1_flat, c2_flat)
            z2_loc = z2_loc.reshape(B, N_max, self.z2_dim)
            z2_scale = z2_scale.reshape(B, N_max, self.z2_dim)

            with pyro.plate("children", N_max, dim=-1):
                with pyro.poutine.mask(mask=child_mask.bool()):
                    with pyro.poutine.scale(scale=annealing_factor):
                        pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

    def encode_z(
        self,
        x1: torch.Tensor,
        x2_set: torch.Tensor | None = None,
        child_mask: torch.Tensor | None = None,
        c2_set: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the posterior mean of the global latent from x1 and optional child context."""
        encoder_input = self.z1_input(x1, x2_set, child_mask, c2_set=c2_set)
        z1_loc, _ = self.index_encoder(encoder_input)
        return z1_loc

    def predict_x2(
        self,
        x1: torch.Tensor,
        c2_target: torch.Tensor,
        x2_set: torch.Tensor | None = None,
        child_mask: torch.Tensor | None = None,
        c2_set: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the MAP child prediction from x1, target c2, and optional child context."""
        z1_loc = self.encode_z(x1, x2_set, child_mask, c2_set)
        z2_loc, _ = self.child_prior(z1_loc, c2_target)
        x2_loc, _ = self.child_decoder(z2_loc, z1_loc, c2_target)
        return x2_loc

    def sample_x2(
        self,
        x1: torch.Tensor,
        c2_target: torch.Tensor,
        n_samples: int = 1,
        x2_set: torch.Tensor | None = None,
        child_mask: torch.Tensor | None = None,
        c2_set: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample child reconstructions conditioned on x1, target c2, and optional child context."""
        squeeze = x1.dim() == 1
        if squeeze:
            x1 = x1.unsqueeze(0)
            c2_target = c2_target.unsqueeze(0)
            if x2_set is not None:
                x2_set = x2_set.unsqueeze(0)
            if child_mask is not None:
                child_mask = child_mask.unsqueeze(0)
            if c2_set is not None:
                c2_set = c2_set.unsqueeze(0)

        z1_loc = self.encode_z(x1, x2_set, child_mask, c2_set)  # (B, z1_dim)
        B = z1_loc.shape[0]

        z1_exp = z1_loc.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
        c2_exp = (
            c2_target.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
        )

        z2_loc, z2_scale = self.child_prior(z1_exp, c2_exp)
        z2 = dist.Normal(z2_loc, z2_scale).sample()
        x2_loc, _ = self.child_decoder(z2, z1_exp, c2_exp)

        x2_loc = x2_loc.reshape(B, n_samples, -1)
        if squeeze:
            x2_loc = x2_loc.squeeze(0)
        return x2_loc

    def reconstruct(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        c2: torch.Tensor,
        x2_set: torch.Tensor | None = None,
        child_mask: torch.Tensor | None = None,
        c2_set: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a stochastic reconstruction of x1 and one child observation."""
        encoder_input = self.z1_input(x1, x2_set, child_mask, c2_set=c2_set)
        z1_loc, z1_scale = self.index_encoder(encoder_input)
        z1 = dist.Normal(z1_loc, z1_scale).sample()
        x1_loc, _ = self.index_decoder(z1)

        z2_loc, z2_scale = self.child_encoder(x2, z1, c2)
        z2 = dist.Normal(z2_loc, z2_scale).sample()
        x2_loc, _ = self.child_decoder(z2, z1, c2)
        return x1_loc, x2_loc

    def reconstruct_map(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        c2: torch.Tensor,
        x2_set: torch.Tensor | None = None,
        child_mask: torch.Tensor | None = None,
        c2_set: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the deterministic reconstruction of x1 and one child observation."""
        encoder_input = self.z1_input(x1, x2_set, child_mask, c2_set=c2_set)
        z1_loc, _ = self.index_encoder(encoder_input)
        x1_loc, _ = self.index_decoder(z1_loc)

        z2_loc, _ = self.child_encoder(x2, z1_loc, c2)
        x2_loc, _ = self.child_decoder(z2_loc, z1_loc, c2)
        return x1_loc, x2_loc

    def counterfactual_prediction(
        self,
        x1: torch.Tensor,
        c2_new: torch.Tensor,
        x2_set: torch.Tensor | None = None,
        child_mask: torch.Tensor | None = None,
        c2_set: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict a child observation under a new condition without observing that child."""
        z1_loc = self.encode_z(x1, x2_set, child_mask, c2_set)
        x1_loc, _ = self.index_decoder(z1_loc)
        z2_loc, _ = self.child_prior(z1_loc, c2_new)
        x2_loc, _ = self.child_decoder(z2_loc, z1_loc, c2_new)
        return x1_loc, x2_loc

    def encode(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        c2: torch.Tensor,
        x2_set: torch.Tensor | None = None,
        child_mask: torch.Tensor | None = None,
        c2_set: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior means for the global and child latents."""
        z1_loc = self.encode_z(x1, x2_set, child_mask, c2_set)
        z2_loc, _ = self.child_encoder(x2, z1_loc, c2)
        return z1_loc, z2_loc


def train(
    data_loaders: Iterable,
    x1_dim: int,
    x2_dim: int,
    c2_dim: int,
    z1_dim: int = 4,
    z2_dim: int = 2,
    hidden_dim: int = 64,
    set_encoder: nn.Module | None = None,
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

    Deprecated: instantiate :class:`HierarchicalVAE` and call ``fit()`` instead.
    """
    warnings.warn(
        "hvae.train() is deprecated; instantiate HierarchicalVAE and call fit() "
        "instead",
        DeprecationWarning,
        stacklevel=2,
    )
    pyro.clear_param_store()

    train_loader, test_loader = data_loaders

    vae = HierarchicalVAE(
        x1_dim=x1_dim,
        x2_dim=x2_dim,
        c2_dim=c2_dim,
        z1_dim=z1_dim,
        z2_dim=z2_dim,
        hidden_dim=hidden_dim,
        set_encoder=set_encoder,
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
