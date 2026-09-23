"""Shared SVI training loop for VAEs with observed and paired latent sites.

This module covers the ordinary ELBO used by the VAE, CVAE, and HVAE. It does
not implement the optional HSIC objective.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
import warnings
import weakref
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TypeVar

import pyro
import torch
from pyro.distributions.util import is_identically_zero
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.trace_elbo import _compute_log_r
from pyro.optim import Adam
from pyro.util import warn_if_nan
from torch import nn
from typing_extensions import Protocol

log = logging.getLogger(__name__)


class TrainingStatus(IntEnum):
    IN_PROGRESS = 0
    COMPLETE = 1
    MAX_EPOCHS_REACHED = 2


class Beta(Protocol):
    def __call__(self, epoch: int) -> float: ...


StateT = TypeVar("StateT")


class StoppingRule(Protocol[StateT]):
    @property
    def max_epochs(self) -> int: ...

    def init(self) -> StateT: ...

    def update(
        self, metrics: EpochMetrics, state: StateT
    ) -> tuple[TrainingStatus, StateT]: ...


@dataclass(frozen=True)
class ConstantBeta(Beta):
    """Always return the same beta value."""

    value: float = 1.0

    def __call__(self, epoch: int) -> float:
        return self.value


@dataclass(frozen=True)
class LinearBetaAnnealing(Beta):
    """Move beta linearly from its epoch-zero start to its target, then hold it."""

    start: float = 0.0
    target: float = 1.0
    epochs: int = 20

    def __post_init__(self) -> None:
        if self.epochs < 0:
            raise ValueError("epochs must be nonnegative")

    @property
    def annealing_epochs(self) -> int:
        return self.epochs

    def __call__(self, epoch: int) -> float:
        """Return beta for a one-based epoch number."""
        if epoch < 1:
            raise ValueError("epoch must be at least 1")
        if self.epochs == 0:
            return self.target
        progress = min(epoch / self.epochs, 1.0)
        return self.start + progress * (self.target - self.start)


@dataclass(frozen=True)
class EpochMetrics:
    """Per-example losses and run metadata for one completed epoch.

    ``kl`` is the Monte Carlo estimate of log q(z) - log p(z), without beta.
    Training terms use the same pre-update draws as the optimizer steps;
    validation terms use the post-epoch model and a fixed beta.
    """

    epoch: int
    beta: float
    train_objective: float
    train_reconstruction: float
    train_kl: float
    validation_beta: float
    validation_objective: float
    validation_reconstruction: float
    validation_kl: float
    learning_rate: float
    epoch_seconds: float
    elapsed_seconds: float
    seed: int | None
    train_examples: int
    validation_examples: int


@dataclass(frozen=True)
class FixedStopping(StoppingRule[None]):
    """Stop after a fixed number of epochs."""

    max_epochs: int = 500

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")

    def init(self) -> None:
        return None

    def update(self, metrics: EpochMetrics, _: None) -> tuple[TrainingStatus, None]:
        status = (
            TrainingStatus.COMPLETE
            if metrics.epoch >= self.max_epochs
            else TrainingStatus.IN_PROGRESS
        )
        return status, None


@dataclass(frozen=True)
class _PatienceState:
    best: float = math.inf
    failed: int = 0
    checks: int = 0


@dataclass(frozen=True)
class PatienceStopping(StoppingRule[_PatienceState]):
    """Stop after ``patience`` eligible checks without sufficient improvement."""

    patience: int = 10
    min_delta: float = 0.0
    warmup_epochs: int = 0
    max_epochs: int = 500

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be nonnegative")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if self.max_epochs <= self.warmup_epochs:
            raise ValueError(
                "max_epochs must be greater than warmup_epochs"
            )

    def init(self) -> _PatienceState:
        return _PatienceState()

    def update(
        self, metrics: EpochMetrics, state: _PatienceState
    ) -> tuple[TrainingStatus, _PatienceState]:
        checks = state.checks + 1
        if checks <= self.warmup_epochs:
            return TrainingStatus.IN_PROGRESS, _PatienceState(
                best=state.best, failed=state.failed, checks=checks
            )

        value = metrics.validation_objective
        if value < state.best - self.min_delta:
            state = _PatienceState(best=value, failed=0, checks=checks)
        else:
            state = _PatienceState(
                best=state.best, failed=state.failed + 1, checks=checks
            )

        if state.failed >= self.patience:
            status = TrainingStatus.COMPLETE
        elif metrics.epoch >= self.max_epochs:
            status = TrainingStatus.MAX_EPOCHS_REACHED
        else:
            status = TrainingStatus.IN_PROGRESS

        return status, state


@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    """Reusable settings for a VAE training run.

    Training uses patience stopping by default. ``stop_rule.max_epochs`` counts
    epochs after beta annealing; use ``FixedStopping`` for a fixed post-annealing
    run. Custom beta schedules and stopping rules may prevent JSON serialization
    of this config.
    """

    stop_rule: StoppingRule = field(default_factory=PatienceStopping)
    beta_schedule: Beta = field(default_factory=ConstantBeta)
    learning_rate: float = 1e-3
    validation_beta: float = 1.0
    num_particles: int = 1
    seed: int | None = None
    jit: bool = False

    def __post_init__(self) -> None:
        annealing_epochs = getattr(self.beta_schedule, "annealing_epochs", 0)
        warmup_epochs = getattr(self.stop_rule, "warmup_epochs", 0)
        max_epochs = self.stop_rule.max_epochs
        if max_epochs < warmup_epochs + annealing_epochs:
            raise ValueError(
                "max_epochs must be greater than or equal to the sum of warmup_epochs and annealing_epochs"
            )


@dataclass
class _LossTerms:
    reconstruction: torch.Tensor
    kl: torch.Tensor


def _trace_terms(model_trace, guide_trace, beta) -> _LossTerms:
    model_sites = {
        name: site
        for name, site in model_trace.nodes.items()
        if site["type"] == "sample"
    }
    guide_sites = {
        name: site
        for name, site in guide_trace.nodes.items()
        if site["type"] == "sample"
    }
    observed = [site for site in model_sites.values() if site["is_observed"]]
    latent_names = {
        name for name, site in model_sites.items() if not site["is_observed"]
    }
    if not observed or not latent_names or latent_names != guide_sites.keys():
        raise ValueError(
            "ELBO metrics require observed model sites and matching "
            "model/guide latent sites"
        )

    # Pyro has already applied plate weights, masks, and beta here.
    # Only latent sites carry beta in the supported model signatures.
    reconstruction = -sum(site["log_prob_sum"].detach() for site in observed)
    kl = (
        sum(
            guide_sites[name]["log_prob_sum"].detach()
            - model_sites[name]["log_prob_sum"].detach()
            for name in latent_names
        )
        / beta
    )
    return _LossTerms(reconstruction, kl)


class _RecordingELBO(Trace_ELBO):
    """Read loss components from Pyro's existing traces without rerunning them."""

    def __init__(self, *, num_particles: int = 1) -> None:
        super().__init__(num_particles=num_particles)
        self._terms: _LossTerms | None = None

    def _unrecorded_traces(self, model, guide, args, kwargs):
        return super()._get_traces(model, guide, args, kwargs)

    def _get_traces(self, model, guide, args, kwargs):
        # Both Trace_ELBO.loss_and_grads() and .loss() use this generator.
        for model_trace, guide_trace in self._unrecorded_traces(
            model, guide, args, kwargs
        ):
            terms = _trace_terms(model_trace, guide_trace, float(args[-1]))
            terms.reconstruction /= self.num_particles
            terms.kl /= self.num_particles
            if self._terms is None:
                self._terms = terms
            else:
                self._terms.reconstruction += terms.reconstruction
                self._terms.kl += terms.kl
            yield model_trace, guide_trace

    def take_terms(self) -> _LossTerms:
        """Return and clear the components from the most recent SVI call."""
        if self._terms is None:
            raise RuntimeError("SVI did not produce a trace")
        terms, self._terms = self._terms, None
        return terms


class _RecordingJitELBO(_RecordingELBO):
    """Compile training loss and metrics while evaluating validation eagerly."""

    def loss_and_grads(self, model, guide, *args, **kwargs):
        kwargs["_pyro_model_id"] = id(model)
        kwargs["_pyro_guide_id"] = id(guide)
        if getattr(self, "_compiled_loss", None) is None:
            weakself = weakref.ref(self)

            @pyro.ops.jit.trace(
                ignore_warnings=self.ignore_jit_warnings,
                jit_options=self.jit_options,
            )
            def compiled_loss(*args, **kwargs):
                kwargs.pop("_pyro_model_id")
                kwargs.pop("_pyro_guide_id")
                current = weakself()
                loss = 0.0
                surrogate_loss = 0.0
                reconstruction = 0.0
                kl = 0.0

                for model_trace, guide_trace in current._unrecorded_traces(
                    model, guide, args, kwargs
                ):
                    elbo_particle = 0.0
                    surrogate_particle = 0.0
                    log_r = None
                    for site in model_trace.nodes.values():
                        if site["type"] == "sample":
                            elbo_particle += site["log_prob_sum"]
                            surrogate_particle += site["log_prob_sum"]
                    for site in guide_trace.nodes.values():
                        if site["type"] != "sample":
                            continue
                        _, score_function_term, entropy_term = site["score_parts"]
                        elbo_particle -= site["log_prob_sum"]
                        if not is_identically_zero(entropy_term):
                            surrogate_particle -= entropy_term.sum()
                        if not is_identically_zero(score_function_term):
                            if log_r is None:
                                log_r = _compute_log_r(model_trace, guide_trace)
                            scaled_log_r = log_r.sum_to(site["cond_indep_stack"])
                            surrogate_particle += (
                                scaled_log_r * score_function_term
                            ).sum()

                    loss -= elbo_particle / current.num_particles
                    surrogate_loss -= surrogate_particle / current.num_particles
                    terms = _trace_terms(model_trace, guide_trace, args[-1])
                    reconstruction += terms.reconstruction / current.num_particles
                    kl += terms.kl / current.num_particles

                return loss, surrogate_loss, reconstruction, kl

            self._compiled_loss = compiled_loss

        loss, surrogate_loss, reconstruction, kl = self._compiled_loss(*args, **kwargs)
        surrogate_loss.backward()
        self._terms = _LossTerms(reconstruction, kl)
        loss_value = loss.item()
        warn_if_nan(loss_value, "loss")
        return loss_value


def _prepare_batch(
    batch: Sequence[torch.Tensor] | torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, ...]:
    if isinstance(batch, torch.Tensor):
        batch = (batch,)
    if not batch or any(not isinstance(value, torch.Tensor) for value in batch):
        raise TypeError("each batch must contain one or more tensors")
    args = tuple(value.to(device) for value in batch)
    if any(value.ndim == 0 for value in args):
        raise ValueError("batch tensors must have a batch dimension")
    size = args[0].shape[0]
    if size == 0 or any(value.shape[0] != size for value in args):
        raise ValueError("batch tensors must have the same nonempty first dimension")
    return args


def _run_epoch(
    loader: Iterable[Sequence[torch.Tensor] | torch.Tensor],
    svi: SVI,
    elbo: _RecordingELBO,
    *,
    beta: float,
    device: torch.device,
    training: bool,
) -> tuple[float, float, float, int]:
    objective = 0.0
    reconstruction: torch.Tensor | None = None
    kl: torch.Tensor | None = None
    examples = 0

    phase = "training" if training else "validation"
    for batch_index, batch in enumerate(loader):
        args = _prepare_batch(batch, device)
        factor = (
            torch.as_tensor(beta, dtype=args[0].dtype, device=device)
            if training and isinstance(elbo, _RecordingJitELBO)
            else beta
        )
        loss = svi.step(*args, factor) if training else svi.evaluate_loss(*args, factor)
        terms = elbo.take_terms()
        if not math.isfinite(loss):
            raise FloatingPointError(
                f"non-finite SVI objective in {phase} batch {batch_index}"
            )
        objective += loss
        reconstruction = (
            terms.reconstruction
            if reconstruction is None
            else reconstruction + terms.reconstruction
        )
        kl = terms.kl if kl is None else kl + terms.kl
        examples += args[0].shape[0]

    if examples == 0 or reconstruction is None or kl is None:
        raise ValueError(f"{phase} loader must yield examples")
    reconstruction_value = reconstruction.item()
    kl_value = kl.item()
    if not math.isfinite(reconstruction_value) or not math.isfinite(kl_value):
        raise FloatingPointError(f"non-finite {phase} reconstruction or KL estimate")
    return (
        objective / examples,
        reconstruction_value / examples,
        kl_value / examples,
        examples,
    )


def _untracked_pyro_params(module: nn.Module) -> list[str]:
    module_params = {id(param) for param in module.parameters()}

    return [
        name
        for name, param in pyro.get_param_store().named_parameters()
        if id(param) not in module_params
    ]


def train_svi(
    module: nn.Module,
    train_loader: Iterable[Sequence[torch.Tensor] | torch.Tensor],
    validation_loader: Iterable[Sequence[torch.Tensor] | torch.Tensor],
    *,
    config: TrainingConfig,
    device: str | torch.device | None = None,
) -> tuple[TrainingStatus, list[EpochMetrics]]:
    """Fit a Pyro VAE and return the training status and epoch metrics.

    ``module`` must expose ``model`` and ``guide`` methods whose final
    positional argument is beta.

    ``config`` contains all training settings, including the stopping rule. The
    built-in rule defaults to ten validation checks without sufficient
    improvement. Stopping starts after beta annealing. Adaptive stopping restores the
    best eligible model state. Reaching the post-annealing epoch cap emits a warning.
    """
    stop_rule = config.stop_rule
    state = stop_rule.init()

    annealing_epochs = getattr(config.beta_schedule, "annealing_epochs", 0)
    total_epochs = stop_rule.max_epochs

    if config.seed is not None:
        pyro.set_rng_seed(config.seed)
    pyro.clear_param_store()

    if device is not None:
        module.to(device)
    parameter = next(module.parameters(), None)
    if parameter is None:
        raise ValueError("module must have parameters")
    training_device = parameter.device
    elbo_type = _RecordingJitELBO if config.jit else _RecordingELBO
    elbo = elbo_type(num_particles=config.num_particles)
    svi = SVI(
        module.model,
        module.guide,
        Adam({"lr": config.learning_rate}),
        loss=elbo,
    )

    history: list[EpochMetrics] = []
    best_state: dict | None = None
    best_loss = math.inf
    best_epoch: int | None = None
    restore_best = not isinstance(stop_rule, FixedStopping)
    started = time.perf_counter()
    previous_mode = module.training

    try:
        for epoch in range(1, total_epochs + 1):
            epoch_started = time.perf_counter()
            beta = float(config.beta_schedule(epoch))
            if not math.isfinite(beta) or beta <= 0:
                raise ValueError("beta_schedule must return a finite positive beta")

            module.train()
            train_loss, train_recon, train_kl, train_count = _run_epoch(
                train_loader,
                svi,
                elbo,
                beta=beta,
                device=training_device,
                training=True,
            )

            if epoch == 1:
                untracked_params = _untracked_pyro_params(module)
                if untracked_params:
                    raise RuntimeError(
                        "Pyro parameters exist outside the module and would not be "
                        "restored by module.state_dict(): "
                        + ", ".join(untracked_params)
                    )

            module.eval()
            val_loss, val_recon, val_kl, val_count = _run_epoch(
                validation_loader,
                svi,
                elbo,
                beta=config.validation_beta,
                device=training_device,
                training=False,
            )
            metrics = EpochMetrics(
                epoch=epoch,
                train_objective=train_loss,
                train_reconstruction=train_recon,
                train_kl=train_kl,
                validation_objective=val_loss,
                validation_reconstruction=val_recon,
                validation_kl=val_kl,
                beta=beta,
                validation_beta=config.validation_beta,
                learning_rate=config.learning_rate,
                epoch_seconds=time.perf_counter() - epoch_started,
                elapsed_seconds=time.perf_counter() - started,
                seed=config.seed,
                train_examples=train_count,
                validation_examples=val_count,
            )
            history.append(metrics)

            if epoch >= annealing_epochs:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    if restore_best:
                        best_state = copy.deepcopy(module.state_dict())

                status, state = stop_rule.update(metrics, state)
                if status != TrainingStatus.IN_PROGRESS:
                    break

        if best_state is not None:
            module.load_state_dict(best_state)

        if status == TrainingStatus.MAX_EPOCHS_REACHED:
            warnings.warn(
                f"max_epochs={stop_rule.max_epochs} reached",
                RuntimeWarning,
                stacklevel=2,
            )

        log.info(
            "vae_training_stop %s",
            json.dumps(
                {
                    "reason": status,
                    "stop_epoch": len(history),
                    "best_epoch": best_epoch,
                    "best_validation_objective": (
                        best_loss if best_epoch is not None else None
                    ),
                },
                allow_nan=False,
            ),
        )

    finally:
        module.train(previous_mode)

    return status, history
