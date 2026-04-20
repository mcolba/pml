"""Shared MCMC imputation routines for VAE, CVAE, and Hierarchical VAE.

All samplers implement the Metropolis-within-Gibbs scheme of Mattei &
Frellsen (2019, ICML — "MIWAE").  The HVAE variant extends the flat
algorithm to a two-level hierarchy with a **joint** MH accept/reject
step for the parent–child latent pair ``(z, u)``.

References
----------
- Mattei & Frellsen (2019), "Missing Data Imputation …", ICML.
- Roberts, Gelman & Gilks (1997), "Weak Convergence …", Ann. Appl. Probab.
- Gilks, Richardson & Spiegelhalter (1996), "MCMC in Practice", Chapman & Hall.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Normal

# ============================================================================
# Shared log-density helpers
# ============================================================================


def _log_normal(x, loc, scale):
    """Diagonal Normal log-density, summed over the last dimension."""
    return Normal(loc, scale).log_prob(x).sum(-1)


def _log_prior_z(z):
    """log p(z) — standard Normal prior."""
    return _log_normal(z, torch.zeros_like(z), torch.ones_like(z))


# ============================================================================
# Flat-VAE imputation  (single latent z, unconditional decoder)
# ============================================================================


def vae_mcmc_impute(vae, x_with_nan, *, num_steps=1000, burn_in=200, use_mh=True):
    """MH-within-Gibbs imputation for a **flat VAE** with partially observed x.

    Parameters
    ----------
    vae : VAE with ``.encoder(x) -> (loc, scale)`` and ``.decoder(z) -> (loc, log_scale)``.
    x_with_nan : Tensor (x_dim,)  — observed entries; NaN at missing positions.
    num_steps  : int               — total MCMC iterations (including burn-in).
    burn_in    : int               — iterations to discard before collecting.
    use_mh     : bool              — if False, always accept encoder proposals.

    Returns
    -------
    x_imputed    : ndarray (x_dim,)          — posterior-mean point estimate.
    miss_samples : ndarray (n_post, n_miss)  — posterior draws for missing dims.
    accept_rate  : float                     — MH acceptance rate.
    """
    vae.eval()
    miss_mask = torch.isnan(x_with_nan)
    x_curr = x_with_nan.clone()

    # ---- Initialisation -----------------------------------------------------
    if miss_mask.all():
        with torch.no_grad():
            z_init = torch.randn(1, vae.z_dim)
            x_loc_init, _ = vae.decoder(z_init)
            x_curr = x_loc_init.squeeze(0)
    else:
        x_curr[miss_mask] = x_curr[~miss_mask].mean()

    with torch.no_grad():
        z_loc, z_scale = vae.encoder(x_curr.unsqueeze(0))
        z_curr = Normal(z_loc, z_scale).sample()

    # ---- Helpers (flat VAE) -------------------------------------------------
    def _log_q(z, x_b):
        loc, scale = vae.encoder(x_b)
        return _log_normal(z, loc, scale)

    def _log_lik(x_b, z):
        loc, log_scale = vae.decoder(z)
        return _log_normal(x_b, loc, torch.exp(log_scale))

    # ---- MCMC loop ----------------------------------------------------------
    miss_samples: list[np.ndarray] = []
    n_accept = 0

    with torch.no_grad():
        for t in range(num_steps):
            x_b = x_curr.unsqueeze(0)

            # -- MH step for z ------------------------------------------------
            z_loc_p, z_scale_p = vae.encoder(x_b)
            z_prop = Normal(z_loc_p, z_scale_p).sample()

            if use_mh:
                log_alpha = (
                    _log_prior_z(z_prop)
                    + _log_lik(x_b, z_prop)
                    + _log_q(z_curr, x_b)
                    - _log_prior_z(z_curr)
                    - _log_lik(x_b, z_curr)
                    - _log_q(z_prop, x_b)
                )
                accept = torch.rand(()).log().item() < log_alpha.item()
            else:
                accept = True

            if accept:
                z_curr = z_prop
                n_accept += 1

            # -- Gibbs step for x_miss ----------------------------------------
            x_loc, log_x_scale = vae.decoder(z_curr)
            x_draw = Normal(x_loc, torch.exp(log_x_scale)).sample().squeeze(0)
            x_curr[miss_mask] = x_draw[miss_mask]

            if t >= burn_in:
                miss_samples.append(x_curr[miss_mask].clone().cpu().numpy())

    miss_samples_arr = np.stack(miss_samples, axis=0)
    x_imputed = x_with_nan.cpu().numpy().copy()
    x_imputed[miss_mask.cpu().numpy()] = miss_samples_arr.mean(axis=0)
    return x_imputed, miss_samples_arr, n_accept / num_steps


# ============================================================================
# CVAE imputation  (single latent z, conditional decoder)
# ============================================================================


def cvae_mcmc_impute(vae, x_with_nan, c, *, num_steps=1000, burn_in=200, use_mh=True):
    """MH-within-Gibbs imputation for a **CVAE** with partially observed x.

    Parameters
    ----------
    vae : CVAE with ``.encoder(x, c)``, ``.decoder(z, c)``, and ``.z_dim``.
    x_with_nan : Tensor (x_dim,)
    c           : Tensor (c_dim,)  — conditioning vector.
    num_steps, burn_in : int
    use_mh : bool

    Returns
    -------
    x_imputed, miss_samples, accept_rate  — same layout as :func:`vae_mcmc_impute`.
    """
    vae.eval()
    miss_mask = torch.isnan(x_with_nan)
    x_curr = x_with_nan.clone()
    c_b = c.unsqueeze(0)

    # ---- Initialisation -----------------------------------------------------
    if miss_mask.all():
        with torch.no_grad():
            z_init = torch.randn(1, vae.z_dim)
            x_loc_init, _ = vae.decoder(z_init, c_b)
            x_curr = x_loc_init.squeeze(0)
    else:
        x_curr[miss_mask] = x_curr[~miss_mask].mean()

    with torch.no_grad():
        z_loc, z_scale = vae.encoder(x_curr.unsqueeze(0), c_b)
        z_curr = Normal(z_loc, z_scale).sample()

    # ---- Helpers (CVAE) -----------------------------------------------------
    def _log_q(z, x_b):
        loc, scale = vae.encoder(x_b, c_b)
        return _log_normal(z, loc, scale)

    def _log_lik(x_b, z):
        loc, log_scale = vae.decoder(z, c_b)
        return _log_normal(x_b, loc, torch.exp(log_scale))

    # ---- MCMC loop ----------------------------------------------------------
    miss_samples: list[np.ndarray] = []
    n_accept = 0

    with torch.no_grad():
        for t in range(num_steps):
            x_b = x_curr.unsqueeze(0)

            z_loc_p, z_scale_p = vae.encoder(x_b, c_b)
            z_prop = Normal(z_loc_p, z_scale_p).sample()

            if use_mh:
                log_alpha = (
                    _log_prior_z(z_prop)
                    + _log_lik(x_b, z_prop)
                    + _log_q(z_curr, x_b)
                    - _log_prior_z(z_curr)
                    - _log_lik(x_b, z_curr)
                    - _log_q(z_prop, x_b)
                )
                accept = torch.rand(()).log().item() < log_alpha.item()
            else:
                accept = True

            if accept:
                z_curr = z_prop
                n_accept += 1

            x_loc, log_x_scale = vae.decoder(z_curr, c_b)
            x_draw = Normal(x_loc, torch.exp(log_x_scale)).sample().squeeze(0)
            x_curr[miss_mask] = x_draw[miss_mask]

            if t >= burn_in:
                miss_samples.append(x_curr[miss_mask].clone().cpu().numpy())

    miss_samples_arr = np.stack(miss_samples, axis=0)
    x_imputed = x_with_nan.cpu().numpy().copy()
    x_imputed[miss_mask.cpu().numpy()] = miss_samples_arr.mean(axis=0)
    return x_imputed, miss_samples_arr, n_accept / num_steps


# ============================================================================
# Hierarchical VAE imputation  — joint MH for (z, u)
# ============================================================================


def _get_hvae_methods(vae):
    """Resolve encoder/decoder/prior methods for both HVAE implementations.

    ``hvae_simp.HierarchicalVAE``  uses  encoder1 / decoder1 / encoder2 / decoder2 / prior_z2
    ``hvae.HierarchicalVAE``       uses  index_encoder / index_decoder / child_encoder / child_decoder / child_prior

    Returns
    -------
    enc_z, dec_z, enc_u, dec_u, prior_u, z_dim_attr, u_dim_attr, z1_input
    """
    if hasattr(vae, "encoder1"):
        return (
            vae.encoder1,
            vae.decoder1,
            vae.encoder2,
            vae.decoder2,
            vae.prior_z2,
            "z1_dim",
            "z2_dim",
            None,
        )
    return (
        vae.index_encoder,
        vae.index_decoder,
        vae.child_encoder,
        vae.child_decoder,
        vae.child_prior,
        "z_dim",
        "u_dim",
        vae.z1_input,
    )


def _z1_enc_input(z1_input, x1_b, x2_b=None, x2_siblings=None):
    """Build encoder input for z1, optionally enriched with an x2 summary.

    When ``z1_input`` is ``None`` (legacy branch) or when
    ``z1_input.uses_summary`` is ``False``, returns *x1_b* unchanged.

    Parameters
    ----------
    z1_input : ParentEncoderInput | None
    x1_b     : Tensor (1, x1_dim)
    x2_b     : Tensor (1, x2_dim) | None — current x2 (may contain imputed values).
    x2_siblings : Tensor (1, K, x2_dim) | None — fully-observed sibling x2s.
        When x2_b is provided, it is prepended to the sibling set so the
        SetEncoder sees the full child set at the same cardinality as training.
        When only x2_siblings is provided (x2_b is None), siblings alone are used.
    """
    if z1_input is None:
        return x1_b
    if x2_b is not None:
        target = x2_b.unsqueeze(1)  # (1, 1, x2_dim)
        if x2_siblings is not None:
            x2_set = torch.cat([target, x2_siblings], dim=1)  # (1, 1+K, x2_dim)
        else:
            x2_set = target  # (1, 1, x2_dim)
        K = x2_set.shape[1]
        child_mask = torch.ones(1, K, dtype=torch.bool, device=x1_b.device)
        return z1_input(x1_b, x2_set, child_mask)
    elif x2_siblings is not None:
        # Warm-start case: use siblings only (no target x2 yet)
        K = x2_siblings.shape[1]
        child_mask = torch.ones(1, K, dtype=torch.bool, device=x1_b.device)
        return z1_input(x1_b, x2_siblings, child_mask)
    return z1_input(x1_b)


def hvae_mcmc_impute(
    vae,
    x1,
    x2_with_nan,
    c,
    *,
    num_steps=5000,
    burn_in=500,
    x2_siblings=None,
    use_mh=True,
):
    """Impute missing entries in x2 via a hierarchical MH-within-Gibbs sampler.

    Extends the flat-VAE MCMC imputation scheme of Mattei & Frellsen (2019,
    ICML) to a two-level hierarchical VAE.  The algorithm uses a **joint**
    Metropolis–Hastings proposal for the parent–child latent pair ``(z, u)``
    followed by an exact Gibbs draw for the missing data entries.

    When the model was trained with ``use_full_posterior=True`` the global
    encoder ``q(z | x1, {x2})`` receives an attention-based summary of the
    child observations via ``vae.z1_input``.  The summary is recomputed at
    every MCMC step using the current (partially imputed) x2.

    Algorithm
    =========

    **Joint MH step for (z, u)**

        The generative model factorises as

            p(z) p(x1 | z) p(u | z, c) p(x2 | u, z, c)

        We propose ``(z*, u*)`` jointly via an ancestral draw through the
        inference network:

            z*  ~ q(z  | x1, {x2_curr})    — parent encoder (set-enriched)
            u*  ~ q(u  | x2_curr, z*, c)    — child encoder, conditioned on z*

        The joint acceptance ratio is

            α = [p(z*) p(x1|z*) p(u*|z*,c) p(x2|u*,z*,c)]
              × [q(z_old|x1,{x2}) q(u_old|x2,z_old,c)]
              ÷ [p(z)  p(x1|z)  p(u|z,c)  p(x2|u,z,c)]
              ÷ [q(z*|x1,{x2}) q(u*|x2,z*,c)]

    **Gibbs step for x2_miss**

        Conditional on (z, u, c) the diagonal Gaussian decoder factorises
        over dimensions, so drawing each missing entry from
        ``p(x2_j | u, z, c)`` is an exact Gibbs step.  Observed entries
        remain fixed.

    Parameters
    ----------
    vae          : HierarchicalVAE   — either ``hvae_simp`` or ``hvae`` variant.
    x1           : Tensor (x1_dim,)  — fully-observed index surface.
    x2_with_nan  : Tensor (x2_dim,)  — single-name surface with NaN at missing.
    c            : Tensor (c_dim,)   — conditioning vector.
    num_steps    : int               — total MCMC iterations (including burn-in).
    burn_in      : int               — iterations to discard before collecting.
    use_mh       : bool              — if False, always accept latent proposals.

    Returns
    -------
    x2_imputed   : ndarray (x2_dim,)           — posterior-mean point estimate.
    miss_samples : ndarray (n_post, n_miss)     — posterior draws for missing dims.
    full_samples : ndarray (n_post, x2_dim)     — full x2 at each post-burn-in step.
    accept_rate  : float                        — joint MH acceptance rate for (z, u).
    """
    enc_z, dec_z, enc_u, dec_u, prior_u, _, _, z1_input = _get_hvae_methods(vae)
    vae.eval()

    if getattr(vae, "use_full_posterior", False) and x2_siblings is None:
        raise ValueError(
            "Model was trained with use_full_posterior=True but x2_siblings "
            "was not provided. Pass the observed sibling x2s so the "
            "SetEncoder receives the same input structure as during training."
        )

    miss_mask = torch.isnan(x2_with_nan)
    x1_b = x1.unsqueeze(0)
    c_b = c.unsqueeze(0)

    # ------------------------------------------------------------------
    # Warm-start: fill missing entries via the prior pathway
    #   x1 → enc_z → z → prior_u → u_loc → dec_u → x2_loc
    # Use siblings for summary if available to match proposal distribution.
    # ------------------------------------------------------------------
    with torch.no_grad():
        enc_input_init = _z1_enc_input(z1_input, x1_b, x2_siblings=x2_siblings)
        z_loc_init, z_scale_init = enc_z(enc_input_init)
        z_curr = Normal(z_loc_init, z_scale_init).sample()

        u_loc_init, _ = prior_u(z_curr, c_b)
        x2_loc_init, _ = dec_u(u_loc_init, z_curr, c_b)

    x2_curr = x2_with_nan.clone()
    if miss_mask.all():
        x2_curr = x2_loc_init.squeeze(0)
    else:
        x2_curr[miss_mask] = x2_loc_init.squeeze(0)[miss_mask]

    # Initialise u from the child encoder using the warm-started x2
    with torch.no_grad():
        u_loc, u_scale = enc_u(x2_curr.unsqueeze(0), z_curr, c_b)
        u_curr = Normal(u_loc, u_scale).sample()

    # ------------------------------------------------------------------
    # Log-density helpers (closed over enc_z, dec_z, etc.)
    # ------------------------------------------------------------------
    def _log_lik_x1(z):
        loc, log_scale = dec_z(z)
        return _log_normal(x1_b, loc, torch.exp(log_scale))

    def _log_q_z(z, x2_b):
        enc_input = _z1_enc_input(z1_input, x1_b, x2_b, x2_siblings=x2_siblings)
        loc, scale = enc_z(enc_input)
        return _log_normal(z, loc, scale)

    def _log_prior_u(u, z):
        loc, scale = prior_u(z, c_b)
        return _log_normal(u, loc, scale)

    def _log_lik_x2(x2_b, u, z):
        loc, log_scale = dec_u(u, z, c_b)
        return _log_normal(x2_b, loc, torch.exp(log_scale))

    def _log_q_u(u, x2_b, z):
        loc, scale = enc_u(x2_b, z, c_b)
        return _log_normal(u, loc, scale)

    # ------------------------------------------------------------------
    # MCMC loop
    # ------------------------------------------------------------------
    miss_samples: list[np.ndarray] = []
    full_samples: list[np.ndarray] = []
    n_accept = 0

    for t in range(num_steps):
        with torch.no_grad():
            x2_b = x2_curr.unsqueeze(0)

            # ==============================================================
            # Joint MH step for (z, u)
            # ==============================================================
            # Ancestral proposal:  z* ~ q(z|x1,{x2}),  u* ~ q(u|x2, z*, c)
            enc_input = _z1_enc_input(z1_input, x1_b, x2_b, x2_siblings=x2_siblings)
            z_loc_p, z_scale_p = enc_z(enc_input)
            z_prop = Normal(z_loc_p, z_scale_p).sample()

            u_loc_p, u_scale_p = enc_u(x2_b, z_prop, c_b)
            u_prop = Normal(u_loc_p, u_scale_p).sample()

            # Forward proposal: log q(z*, u* | x1, {x2}, c)
            log_q_fwd = _log_q_z(z_prop, x2_b) + _log_q_u(u_prop, x2_b, z_prop)

            # Reverse proposal: log q(z_old, u_old | x1, {x2}, c)
            log_q_rev = _log_q_z(z_curr, x2_b) + _log_q_u(u_curr, x2_b, z_curr)

            # Target (unnormalised) at proposed (z*, u*)
            log_p_prop = (
                _log_prior_z(z_prop)
                + _log_lik_x1(z_prop)
                + _log_prior_u(u_prop, z_prop)
                + _log_lik_x2(x2_b, u_prop, z_prop)
            )

            # Target (unnormalised) at current (z, u)
            log_p_curr = (
                _log_prior_z(z_curr)
                + _log_lik_x1(z_curr)
                + _log_prior_u(u_curr, z_curr)
                + _log_lik_x2(x2_b, u_curr, z_curr)
            )

            # Accept / reject
            if use_mh:
                log_alpha = log_p_prop + log_q_rev - log_p_curr - log_q_fwd
                accept = torch.log(torch.rand(1)) < log_alpha.item()
            else:
                accept = True

            if accept:
                z_curr = z_prop
                u_curr = u_prop
                n_accept += 1

            # ==============================================================
            # Gibbs step for x2_miss
            # ==============================================================
            x2_loc, log_x2_scale = dec_u(u_curr, z_curr, c_b)
            x2_draw = Normal(x2_loc, torch.exp(log_x2_scale)).sample().squeeze(0)
            x2_curr[miss_mask] = x2_draw[miss_mask]

            # Collect posterior samples after burn-in
            if t >= burn_in:
                miss_samples.append(x2_curr[miss_mask].clone().cpu().numpy())
                full_samples.append(x2_curr.clone().cpu().numpy())

    accept_rate = n_accept / num_steps
    miss_samples_arr = np.stack(miss_samples, axis=0)
    full_samples_arr = np.stack(full_samples, axis=0)

    x2_imputed = x2_with_nan.cpu().numpy().copy()
    x2_imputed[miss_mask.cpu().numpy()] = miss_samples_arr.mean(axis=0)

    return x2_imputed, miss_samples_arr, full_samples_arr, accept_rate


# ============================================================================
# Sequential MH — block-wise updates for z then u
# ============================================================================


def hvae_mcmc_impute_sequential(
    vae,
    x1,
    x2_with_nan,
    c,
    *,
    num_steps=5000,
    burn_in=500,
    x2_siblings=None,
    use_mh=True,
):
    """Impute missing entries in x2 with **sequential** block-MH for z then u.

    Instead of a single joint ``(z, u)`` proposal, this sampler performs two
    separate Metropolis–Hastings steps per iteration:

    1. **MH for z** (u held fixed):

       Propose ``z* ~ q(z | x1, {x2})``.  Accept with ratio

           α_z = p(z*) p(x1|z*) p(u|z*,c) p(x2|u,z*,c)  ·  q(z|x1,{x2})
               ÷ p(z)  p(x1|z)  p(u|z,c)  p(x2|u,z,c)   ÷  q(z*|x1,{x2})

    2. **MH for u** (z held fixed):

       Propose ``u* ~ q(u | x2, z, c)``.  Accept with ratio

           α_u = p(u*|z,c) p(x2|u*,z,c)  ·  q(u|x2,z,c)
               ÷ p(u|z,c)  p(x2|u,z,c)   ÷  q(u*|x2,z,c)

    3. **Gibbs for x2_miss** — exact draw from ``p(x2_j | u, z, c)``.

    Parameters
    ----------
    vae          : HierarchicalVAE
    x1           : Tensor (x1_dim,)
    x2_with_nan  : Tensor (x2_dim,)
    c            : Tensor (c_dim,)
    num_steps    : int
    burn_in      : int
    use_mh       : bool

    Returns
    -------
    x2_imputed   : ndarray (x2_dim,)
    miss_samples : ndarray (n_post, n_miss)
    full_samples : ndarray (n_post, x2_dim)
    accept_rate_z : float — MH acceptance rate for z.
    accept_rate_u : float — MH acceptance rate for u.
    """
    enc_z, dec_z, enc_u, dec_u, prior_u, _, _, z1_input = _get_hvae_methods(vae)
    vae.eval()

    if getattr(vae, "use_full_posterior", False) and x2_siblings is None:
        raise ValueError(
            "Model was trained with use_full_posterior=True but x2_siblings "
            "was not provided. Pass the observed sibling x2s so the "
            "SetEncoder receives the same input structure as during training."
        )

    miss_mask = torch.isnan(x2_with_nan)
    x1_b = x1.unsqueeze(0)
    c_b = c.unsqueeze(0)

    # ------------------------------------------------------------------
    # Warm-start — use siblings for summary if available
    # ------------------------------------------------------------------
    with torch.no_grad():
        enc_input_init = _z1_enc_input(z1_input, x1_b, x2_siblings=x2_siblings)
        z_loc_init, z_scale_init = enc_z(enc_input_init)
        z_curr = Normal(z_loc_init, z_scale_init).sample()

        u_loc_init, _ = prior_u(z_curr, c_b)
        x2_loc_init, _ = dec_u(u_loc_init, z_curr, c_b)

    x2_curr = x2_with_nan.clone()
    if miss_mask.all():
        x2_curr = x2_loc_init.squeeze(0)
    else:
        x2_curr[miss_mask] = x2_loc_init.squeeze(0)[miss_mask]

    with torch.no_grad():
        u_loc, u_scale = enc_u(x2_curr.unsqueeze(0), z_curr, c_b)
        u_curr = Normal(u_loc, u_scale).sample()

    # ------------------------------------------------------------------
    # Log-density helpers
    # ------------------------------------------------------------------
    def _log_lik_x1(z):
        loc, log_scale = dec_z(z)
        return _log_normal(x1_b, loc, torch.exp(log_scale))

    def _log_q_z(z, x2_b):
        enc_input = _z1_enc_input(z1_input, x1_b, x2_b, x2_siblings=x2_siblings)
        loc, scale = enc_z(enc_input)
        return _log_normal(z, loc, scale)

    def _log_prior_u(u, z):
        loc, scale = prior_u(z, c_b)
        return _log_normal(u, loc, scale)

    def _log_lik_x2(x2_b, u, z):
        loc, log_scale = dec_u(u, z, c_b)
        return _log_normal(x2_b, loc, torch.exp(log_scale))

    def _log_q_u(u, x2_b, z):
        loc, scale = enc_u(x2_b, z, c_b)
        return _log_normal(u, loc, scale)

    # ------------------------------------------------------------------
    # MCMC loop
    # ------------------------------------------------------------------
    miss_samples: list[np.ndarray] = []
    full_samples: list[np.ndarray] = []
    n_accept_z = 0
    n_accept_u = 0

    for t in range(num_steps):
        with torch.no_grad():
            x2_b = x2_curr.unsqueeze(0)

            # ==============================================================
            # Block 1 — MH for z  (u held fixed)
            # ==============================================================
            enc_input = _z1_enc_input(z1_input, x1_b, x2_b, x2_siblings=x2_siblings)
            z_loc_p, z_scale_p = enc_z(enc_input)
            z_prop = Normal(z_loc_p, z_scale_p).sample()

            log_p_z_prop = (
                _log_prior_z(z_prop)
                + _log_lik_x1(z_prop)
                + _log_prior_u(u_curr, z_prop)
                + _log_lik_x2(x2_b, u_curr, z_prop)
            )
            log_p_z_curr = (
                _log_prior_z(z_curr)
                + _log_lik_x1(z_curr)
                + _log_prior_u(u_curr, z_curr)
                + _log_lik_x2(x2_b, u_curr, z_curr)
            )
            log_q_z_fwd = _log_q_z(z_prop, x2_b)
            log_q_z_rev = _log_q_z(z_curr, x2_b)

            if use_mh:
                log_alpha_z = log_p_z_prop + log_q_z_rev - log_p_z_curr - log_q_z_fwd
                accept_z = torch.log(torch.rand(1)) < log_alpha_z.item()
            else:
                accept_z = True

            if accept_z:
                z_curr = z_prop
                n_accept_z += 1

            # ==============================================================
            # Block 2 — MH for u  (z held fixed)
            # ==============================================================
            u_loc_p, u_scale_p = enc_u(x2_b, z_curr, c_b)
            u_prop = Normal(u_loc_p, u_scale_p).sample()

            log_p_u_prop = _log_prior_u(u_prop, z_curr) + _log_lik_x2(
                x2_b, u_prop, z_curr
            )
            log_p_u_curr = _log_prior_u(u_curr, z_curr) + _log_lik_x2(
                x2_b, u_curr, z_curr
            )
            log_q_u_fwd = _log_q_u(u_prop, x2_b, z_curr)
            log_q_u_rev = _log_q_u(u_curr, x2_b, z_curr)

            if use_mh:
                log_alpha_u = log_p_u_prop + log_q_u_rev - log_p_u_curr - log_q_u_fwd
                accept_u = torch.log(torch.rand(1)) < log_alpha_u.item()
            else:
                accept_u = True

            if accept_u:
                u_curr = u_prop
                n_accept_u += 1

            # ==============================================================
            # Gibbs step for x2_miss
            # ==============================================================
            x2_loc, log_x2_scale = dec_u(u_curr, z_curr, c_b)
            x2_draw = Normal(x2_loc, torch.exp(log_x2_scale)).sample().squeeze(0)
            x2_curr[miss_mask] = x2_draw[miss_mask]

            if t >= burn_in:
                miss_samples.append(x2_curr[miss_mask].clone().cpu().numpy())
                full_samples.append(x2_curr.clone().cpu().numpy())

    accept_rate_z = n_accept_z / num_steps
    accept_rate_u = n_accept_u / num_steps
    miss_samples_arr = np.stack(miss_samples, axis=0)
    full_samples_arr = np.stack(full_samples, axis=0)

    x2_imputed = x2_with_nan.cpu().numpy().copy()
    x2_imputed[miss_mask.cpu().numpy()] = miss_samples_arr.mean(axis=0)

    return x2_imputed, miss_samples_arr, full_samples_arr, accept_rate_z, accept_rate_u


# ============================================================================
# Approximate sequential — exact z, MH for u only
# ============================================================================


def hvae_mcmc_impute_approx(
    vae,
    x1,
    x2_with_nan,
    c,
    *,
    num_steps=5000,
    burn_in=500,
    x2_siblings=None,
    use_mh=True,
):
    """Simplified imputation assuming the global encoder is exact.

    The global encoder ``q(z | x1, {x2})`` is treated as a perfect
    approximation to the full conditional ``p(z | x1, x2, u, c)``, so
    every z proposal is **accepted without an MH correction**.  Only the
    child latent u goes through an MH accept/reject step.

    This is cheaper per iteration (one fewer set of log-density
    evaluations) and can be a reasonable approximation when the parent
    encoder is well trained and x1 is highly informative about z.

    Algorithm (per iteration)
    =========================

    1. **Direct draw for z** — always accepted:

           z  ←  z* ~ q(z | x1, {x2_curr})

    2. **MH for u** (z held fixed):

           u* ~ q(u | x2, z, c)
           α_u = p(u*|z,c) p(x2|u*,z,c) q(u|x2,z,c)
               ÷ p(u|z,c)  p(x2|u,z,c)  q(u*|x2,z,c)

    3. **Gibbs for x2_miss** — exact draw from ``p(x2_j | u, z, c)``.

    Parameters
    ----------
    vae          : HierarchicalVAE
    x1           : Tensor (x1_dim,)
    x2_with_nan  : Tensor (x2_dim,)
    c            : Tensor (c_dim,)
    num_steps    : int
    burn_in      : int
    use_mh       : bool

    Returns
    -------
    x2_imputed   : ndarray (x2_dim,)
    miss_samples : ndarray (n_post, n_miss)
    full_samples : ndarray (n_post, x2_dim)
    accept_rate_u : float — MH acceptance rate for u.
    """
    enc_z, dec_z, enc_u, dec_u, prior_u, _, _, z1_input = _get_hvae_methods(vae)
    vae.eval()

    if getattr(vae, "use_full_posterior", False) and x2_siblings is None:
        raise ValueError(
            "Model was trained with use_full_posterior=True but x2_siblings "
            "was not provided. Pass the observed sibling x2s so the "
            "SetEncoder receives the same input structure as during training."
        )

    miss_mask = torch.isnan(x2_with_nan)
    x1_b = x1.unsqueeze(0)
    c_b = c.unsqueeze(0)

    # ------------------------------------------------------------------
    # Warm-start — use siblings for summary if available
    # ------------------------------------------------------------------
    with torch.no_grad():
        enc_input_init = _z1_enc_input(z1_input, x1_b, x2_siblings=x2_siblings)
        z_loc_init, z_scale_init = enc_z(enc_input_init)
        z_curr = Normal(z_loc_init, z_scale_init).sample()

        u_loc_init, _ = prior_u(z_curr, c_b)
        x2_loc_init, _ = dec_u(u_loc_init, z_curr, c_b)

    x2_curr = x2_with_nan.clone()
    if miss_mask.all():
        x2_curr = x2_loc_init.squeeze(0)
    else:
        x2_curr[miss_mask] = x2_loc_init.squeeze(0)[miss_mask]

    with torch.no_grad():
        u_loc, u_scale = enc_u(x2_curr.unsqueeze(0), z_curr, c_b)
        u_curr = Normal(u_loc, u_scale).sample()

    # ------------------------------------------------------------------
    # Log-density helpers (only those needed for the u MH step)
    # ------------------------------------------------------------------
    def _log_prior_u(u, z):
        loc, scale = prior_u(z, c_b)
        return _log_normal(u, loc, scale)

    def _log_lik_x2(x2_b, u, z):
        loc, log_scale = dec_u(u, z, c_b)
        return _log_normal(x2_b, loc, torch.exp(log_scale))

    def _log_q_u(u, x2_b, z):
        loc, scale = enc_u(x2_b, z, c_b)
        return _log_normal(u, loc, scale)

    # ------------------------------------------------------------------
    # MCMC loop
    # ------------------------------------------------------------------
    miss_samples: list[np.ndarray] = []
    full_samples: list[np.ndarray] = []
    n_accept_u = 0

    for t in range(num_steps):
        with torch.no_grad():
            x2_b = x2_curr.unsqueeze(0)

            # ==============================================================
            # Direct draw for z  (encoder assumed exact — always accepted)
            # ==============================================================
            enc_input = _z1_enc_input(z1_input, x1_b, x2_b, x2_siblings=x2_siblings)
            z_loc_p, z_scale_p = enc_z(enc_input)
            z_curr = Normal(z_loc_p, z_scale_p).sample()

            # ==============================================================
            # MH for u  (z held fixed)
            # ==============================================================
            u_loc_p, u_scale_p = enc_u(x2_b, z_curr, c_b)
            u_prop = Normal(u_loc_p, u_scale_p).sample()

            log_p_u_prop = _log_prior_u(u_prop, z_curr) + _log_lik_x2(
                x2_b, u_prop, z_curr
            )
            log_p_u_curr = _log_prior_u(u_curr, z_curr) + _log_lik_x2(
                x2_b, u_curr, z_curr
            )
            log_q_u_fwd = _log_q_u(u_prop, x2_b, z_curr)
            log_q_u_rev = _log_q_u(u_curr, x2_b, z_curr)

            if use_mh:
                log_alpha_u = log_p_u_prop + log_q_u_rev - log_p_u_curr - log_q_u_fwd
                accept_u = torch.log(torch.rand(1)) < log_alpha_u.item()
            else:
                accept_u = True

            if accept_u:
                u_curr = u_prop
                n_accept_u += 1

            # ==============================================================
            # Gibbs step for x2_miss
            # ==============================================================
            x2_loc, log_x2_scale = dec_u(u_curr, z_curr, c_b)
            x2_draw = Normal(x2_loc, torch.exp(log_x2_scale)).sample().squeeze(0)
            x2_curr[miss_mask] = x2_draw[miss_mask]

            if t >= burn_in:
                miss_samples.append(x2_curr[miss_mask].clone().cpu().numpy())
                full_samples.append(x2_curr.clone().cpu().numpy())

    accept_rate_u = n_accept_u / num_steps
    miss_samples_arr = np.stack(miss_samples, axis=0)
    full_samples_arr = np.stack(full_samples, axis=0)

    x2_imputed = x2_with_nan.cpu().numpy().copy()
    x2_imputed[miss_mask.cpu().numpy()] = miss_samples_arr.mean(axis=0)

    return x2_imputed, miss_samples_arr, full_samples_arr, accept_rate_u


# ============================================================================
# Private-Shared HVAE imputation — joint MH for (z, u, z_private)
# ============================================================================


def _get_ps_hvae_methods(vae):
    """Resolve encoder/decoder/prior methods for HierarchicalPSVAE.

    Returns
    -------
    enc_z, dec_z, enc_u, dec_u, prior_u, private_enc, z1_input, z_private_dim
    """
    return (
        vae.index_encoder,
        vae.index_decoder,
        vae.child_encoder,
        vae.child_decoder,
        vae.child_prior,
        vae.private_encoder,
        vae.z1_input,
        vae.z_private_dim,
    )


def hvae_ps_mcmc_impute(
    vae,
    x1,
    x2_with_nan,
    c,
    *,
    num_steps=5000,
    burn_in=500,
    x2_siblings=None,
    use_mh=True,
):
    """Impute missing entries in x2 via PS-HVAE joint MH-within-Gibbs.

    Extends the hierarchical imputation to a three-latent model with
    ``(z, u, z_private)``.  A single **joint** MH proposal draws all
    three latents ancestrally through the inference network, then
    accepts/rejects as a block.

    Generative model
    ================
        p(z) p(x1|z) p(u|z,c) p(z_p) p(x2|u,z,c,z_p)

    Proposal (ancestral)
    ====================
        z*  ~ q(z | x1, {x2})
        u*  ~ q(u | x2, z*, c)
        zp* ~ q(z_p | x2, c)

    Parameters
    ----------
    vae          : HierarchicalPSVAE
    x1           : Tensor (x1_dim,)
    x2_with_nan  : Tensor (x2_dim,)
    c            : Tensor (c_dim,)  — conditioning vector (c2 for the child).
    num_steps    : int
    burn_in      : int
    use_mh       : bool

    Returns
    -------
    x2_imputed   : ndarray (x2_dim,)
    miss_samples : ndarray (n_post, n_miss)
    full_samples : ndarray (n_post, x2_dim)
    accept_rate  : float — joint MH acceptance rate.
    """
    enc_z, dec_z, enc_u, dec_u, prior_u, priv_enc, z1_input, zp_dim = (
        _get_ps_hvae_methods(vae)
    )
    vae.eval()

    if getattr(vae, "use_full_posterior", False) and x2_siblings is None:
        raise ValueError(
            "Model was trained with use_full_posterior=True but x2_siblings "
            "was not provided. Pass the observed sibling x2s so the "
            "SetEncoder receives the same input structure as during training."
        )

    miss_mask = torch.isnan(x2_with_nan)
    x1_b = x1.unsqueeze(0)
    c_b = c.unsqueeze(0)

    # ------------------------------------------------------------------
    # Warm-start — use siblings for summary if available
    # ------------------------------------------------------------------
    with torch.no_grad():
        enc_input_init = _z1_enc_input(z1_input, x1_b, x2_siblings=x2_siblings)
        z_loc_init, z_scale_init = enc_z(enc_input_init)
        z_curr = Normal(z_loc_init, z_scale_init).sample()

        u_loc_init, _ = prior_u(z_curr, c_b)
        zp_curr = torch.randn(1, zp_dim)
        x2_loc_init, _ = dec_u(u_loc_init, z_curr, c_b, zp_curr)

    x2_curr = x2_with_nan.clone()
    if miss_mask.all():
        x2_curr = x2_loc_init.squeeze(0)
    else:
        x2_curr[miss_mask] = x2_loc_init.squeeze(0)[miss_mask]

    with torch.no_grad():
        u_loc, u_scale = enc_u(x2_curr.unsqueeze(0), z_curr, c_b)
        u_curr = Normal(u_loc, u_scale).sample()
        zp_loc, zp_scale = priv_enc(x2_curr.unsqueeze(0), c_b)
        zp_curr = Normal(zp_loc, zp_scale).sample()

    # ------------------------------------------------------------------
    # Log-density helpers
    # ------------------------------------------------------------------
    def _log_lik_x1(z):
        loc, log_scale = dec_z(z)
        return _log_normal(x1_b, loc, torch.exp(log_scale))

    def _log_q_z(z, x2_b):
        enc_input = _z1_enc_input(z1_input, x1_b, x2_b, x2_siblings=x2_siblings)
        loc, scale = enc_z(enc_input)
        return _log_normal(z, loc, scale)

    def _log_prior_u(u, z):
        loc, scale = prior_u(z, c_b)
        return _log_normal(u, loc, scale)

    def _log_lik_x2(x2_b, u, z, zp):
        loc, log_scale = dec_u(u, z, c_b, zp)
        return _log_normal(x2_b, loc, torch.exp(log_scale))

    def _log_q_u(u, x2_b, z):
        loc, scale = enc_u(x2_b, z, c_b)
        return _log_normal(u, loc, scale)

    def _log_q_zp(zp, x2_b):
        loc, scale = priv_enc(x2_b, c_b)
        return _log_normal(zp, loc, scale)

    # ------------------------------------------------------------------
    # MCMC loop
    # ------------------------------------------------------------------
    miss_samples: list[np.ndarray] = []
    full_samples: list[np.ndarray] = []
    n_accept = 0

    for t in range(num_steps):
        with torch.no_grad():
            x2_b = x2_curr.unsqueeze(0)

            # ==============================================================
            # Joint MH step for (z, u, z_private)
            # ==============================================================
            # Ancestral proposal
            enc_input = _z1_enc_input(z1_input, x1_b, x2_b, x2_siblings=x2_siblings)
            z_loc_p, z_scale_p = enc_z(enc_input)
            z_prop = Normal(z_loc_p, z_scale_p).sample()

            u_loc_p, u_scale_p = enc_u(x2_b, z_prop, c_b)
            u_prop = Normal(u_loc_p, u_scale_p).sample()

            zp_loc_p, zp_scale_p = priv_enc(x2_b, c_b)
            zp_prop = Normal(zp_loc_p, zp_scale_p).sample()

            # Forward proposal log-density
            log_q_fwd = (
                _log_q_z(z_prop, x2_b)
                + _log_q_u(u_prop, x2_b, z_prop)
                + _log_q_zp(zp_prop, x2_b)
            )

            # Reverse proposal log-density
            log_q_rev = (
                _log_q_z(z_curr, x2_b)
                + _log_q_u(u_curr, x2_b, z_curr)
                + _log_q_zp(zp_curr, x2_b)
            )

            # Target at proposed
            log_p_prop = (
                _log_prior_z(z_prop)
                + _log_lik_x1(z_prop)
                + _log_prior_u(u_prop, z_prop)
                + _log_prior_z(zp_prop)  # p(z_private) = N(0,I)
                + _log_lik_x2(x2_b, u_prop, z_prop, zp_prop)
            )

            # Target at current
            log_p_curr = (
                _log_prior_z(z_curr)
                + _log_lik_x1(z_curr)
                + _log_prior_u(u_curr, z_curr)
                + _log_prior_z(zp_curr)
                + _log_lik_x2(x2_b, u_curr, z_curr, zp_curr)
            )

            if use_mh:
                log_alpha = log_p_prop + log_q_rev - log_p_curr - log_q_fwd
                accept = torch.log(torch.rand(1)) < log_alpha.item()
            else:
                accept = True

            if accept:
                z_curr = z_prop
                u_curr = u_prop
                zp_curr = zp_prop
                n_accept += 1

            # ==============================================================
            # Gibbs step for x2_miss
            # ==============================================================
            x2_loc, log_x2_scale = dec_u(u_curr, z_curr, c_b, zp_curr)
            x2_draw = Normal(x2_loc, torch.exp(log_x2_scale)).sample().squeeze(0)
            x2_curr[miss_mask] = x2_draw[miss_mask]

            if t >= burn_in:
                miss_samples.append(x2_curr[miss_mask].clone().cpu().numpy())
                full_samples.append(x2_curr.clone().cpu().numpy())

    accept_rate = n_accept / num_steps
    miss_samples_arr = np.stack(miss_samples, axis=0)
    full_samples_arr = np.stack(full_samples, axis=0)

    x2_imputed = x2_with_nan.cpu().numpy().copy()
    x2_imputed[miss_mask.cpu().numpy()] = miss_samples_arr.mean(axis=0)

    return x2_imputed, miss_samples_arr, full_samples_arr, accept_rate


# ============================================================================
# PS-HVAE sequential MH — block-wise updates for z, u, z_private
# ============================================================================


def hvae_ps_mcmc_impute_sequential(
    vae,
    x1,
    x2_with_nan,
    c,
    *,
    num_steps=5000,
    burn_in=500,
    x2_siblings=None,
    use_mh=True,
):
    """Impute missing x2 via PS-HVAE with sequential block MH.

    Three separate MH blocks per iteration:

    1. **MH for z** (u, z_private held fixed)
    2. **MH for u** (z, z_private held fixed)
    3. **MH for z_private** (z, u held fixed)
    4. **Gibbs for x2_miss**

    Parameters
    ----------
    vae          : HierarchicalPSVAE
    x1           : Tensor (x1_dim,)
    x2_with_nan  : Tensor (x2_dim,)
    c            : Tensor (c_dim,)
    num_steps    : int
    burn_in      : int
    use_mh       : bool

    Returns
    -------
    x2_imputed     : ndarray (x2_dim,)
    miss_samples   : ndarray (n_post, n_miss)
    full_samples   : ndarray (n_post, x2_dim)
    accept_rate_z  : float
    accept_rate_u  : float
    accept_rate_zp : float
    """
    enc_z, dec_z, enc_u, dec_u, prior_u, priv_enc, z1_input, zp_dim = (
        _get_ps_hvae_methods(vae)
    )
    vae.eval()

    if getattr(vae, "use_full_posterior", False) and x2_siblings is None:
        raise ValueError(
            "Model was trained with use_full_posterior=True but x2_siblings "
            "was not provided. Pass the observed sibling x2s so the "
            "SetEncoder receives the same input structure as during training."
        )

    miss_mask = torch.isnan(x2_with_nan)
    x1_b = x1.unsqueeze(0)
    c_b = c.unsqueeze(0)

    # ------------------------------------------------------------------
    # Warm-start — use siblings for summary if available
    # ------------------------------------------------------------------
    with torch.no_grad():
        enc_input_init = _z1_enc_input(z1_input, x1_b, x2_siblings=x2_siblings)
        z_loc_init, z_scale_init = enc_z(enc_input_init)
        z_curr = Normal(z_loc_init, z_scale_init).sample()

        u_loc_init, _ = prior_u(z_curr, c_b)
        zp_curr = torch.randn(1, zp_dim)
        x2_loc_init, _ = dec_u(u_loc_init, z_curr, c_b, zp_curr)

    x2_curr = x2_with_nan.clone()
    if miss_mask.all():
        x2_curr = x2_loc_init.squeeze(0)
    else:
        x2_curr[miss_mask] = x2_loc_init.squeeze(0)[miss_mask]

    with torch.no_grad():
        u_loc, u_scale = enc_u(x2_curr.unsqueeze(0), z_curr, c_b)
        u_curr = Normal(u_loc, u_scale).sample()
        zp_loc, zp_scale = priv_enc(x2_curr.unsqueeze(0), c_b)
        zp_curr = Normal(zp_loc, zp_scale).sample()

    # ------------------------------------------------------------------
    # Log-density helpers
    # ------------------------------------------------------------------
    def _log_lik_x1(z):
        loc, log_scale = dec_z(z)
        return _log_normal(x1_b, loc, torch.exp(log_scale))

    def _log_q_z(z, x2_b):
        enc_input = _z1_enc_input(z1_input, x1_b, x2_b, x2_siblings=x2_siblings)
        loc, scale = enc_z(enc_input)
        return _log_normal(z, loc, scale)

    def _log_prior_u(u, z):
        loc, scale = prior_u(z, c_b)
        return _log_normal(u, loc, scale)

    def _log_lik_x2(x2_b, u, z, zp):
        loc, log_scale = dec_u(u, z, c_b, zp)
        return _log_normal(x2_b, loc, torch.exp(log_scale))

    def _log_q_u(u, x2_b, z):
        loc, scale = enc_u(x2_b, z, c_b)
        return _log_normal(u, loc, scale)

    def _log_q_zp(zp, x2_b):
        loc, scale = priv_enc(x2_b, c_b)
        return _log_normal(zp, loc, scale)

    # ------------------------------------------------------------------
    # MCMC loop
    # ------------------------------------------------------------------
    miss_samples: list[np.ndarray] = []
    full_samples: list[np.ndarray] = []
    n_accept_z = 0
    n_accept_u = 0
    n_accept_zp = 0

    for t in range(num_steps):
        with torch.no_grad():
            x2_b = x2_curr.unsqueeze(0)

            # ==============================================================
            # Block 1 — MH for z  (u, z_private held fixed)
            # ==============================================================
            enc_input = _z1_enc_input(z1_input, x1_b, x2_b, x2_siblings=x2_siblings)
            z_loc_p, z_scale_p = enc_z(enc_input)
            z_prop = Normal(z_loc_p, z_scale_p).sample()

            log_p_z_prop = (
                _log_prior_z(z_prop)
                + _log_lik_x1(z_prop)
                + _log_prior_u(u_curr, z_prop)
                + _log_lik_x2(x2_b, u_curr, z_prop, zp_curr)
            )
            log_p_z_curr = (
                _log_prior_z(z_curr)
                + _log_lik_x1(z_curr)
                + _log_prior_u(u_curr, z_curr)
                + _log_lik_x2(x2_b, u_curr, z_curr, zp_curr)
            )
            log_q_z_fwd = _log_q_z(z_prop, x2_b)
            log_q_z_rev = _log_q_z(z_curr, x2_b)

            if use_mh:
                log_alpha_z = log_p_z_prop + log_q_z_rev - log_p_z_curr - log_q_z_fwd
                accept_z = torch.log(torch.rand(1)) < log_alpha_z.item()
            else:
                accept_z = True

            if accept_z:
                z_curr = z_prop
                n_accept_z += 1

            # ==============================================================
            # Block 2 — MH for u  (z, z_private held fixed)
            # ==============================================================
            u_loc_p, u_scale_p = enc_u(x2_b, z_curr, c_b)
            u_prop = Normal(u_loc_p, u_scale_p).sample()

            log_p_u_prop = _log_prior_u(u_prop, z_curr) + _log_lik_x2(
                x2_b, u_prop, z_curr, zp_curr
            )
            log_p_u_curr = _log_prior_u(u_curr, z_curr) + _log_lik_x2(
                x2_b, u_curr, z_curr, zp_curr
            )
            log_q_u_fwd = _log_q_u(u_prop, x2_b, z_curr)
            log_q_u_rev = _log_q_u(u_curr, x2_b, z_curr)

            if use_mh:
                log_alpha_u = log_p_u_prop + log_q_u_rev - log_p_u_curr - log_q_u_fwd
                accept_u = torch.log(torch.rand(1)) < log_alpha_u.item()
            else:
                accept_u = True

            if accept_u:
                u_curr = u_prop
                n_accept_u += 1

            # ==============================================================
            # Block 3 — MH for z_private  (z, u held fixed)
            # ==============================================================
            zp_loc_p, zp_scale_p = priv_enc(x2_b, c_b)
            zp_prop = Normal(zp_loc_p, zp_scale_p).sample()

            log_p_zp_prop = _log_prior_z(zp_prop) + _log_lik_x2(
                x2_b, u_curr, z_curr, zp_prop
            )
            log_p_zp_curr = _log_prior_z(zp_curr) + _log_lik_x2(
                x2_b, u_curr, z_curr, zp_curr
            )
            log_q_zp_fwd = _log_q_zp(zp_prop, x2_b)
            log_q_zp_rev = _log_q_zp(zp_curr, x2_b)

            if use_mh:
                log_alpha_zp = (
                    log_p_zp_prop + log_q_zp_rev - log_p_zp_curr - log_q_zp_fwd
                )
                accept_zp = torch.log(torch.rand(1)) < log_alpha_zp.item()
            else:
                accept_zp = True

            if accept_zp:
                zp_curr = zp_prop
                n_accept_zp += 1

            # ==============================================================
            # Gibbs step for x2_miss
            # ==============================================================
            x2_loc, log_x2_scale = dec_u(u_curr, z_curr, c_b, zp_curr)
            x2_draw = Normal(x2_loc, torch.exp(log_x2_scale)).sample().squeeze(0)
            x2_curr[miss_mask] = x2_draw[miss_mask]

            if t >= burn_in:
                miss_samples.append(x2_curr[miss_mask].clone().cpu().numpy())
                full_samples.append(x2_curr.clone().cpu().numpy())

    accept_rate_z = n_accept_z / num_steps
    accept_rate_u = n_accept_u / num_steps
    accept_rate_zp = n_accept_zp / num_steps
    miss_samples_arr = np.stack(miss_samples, axis=0)
    full_samples_arr = np.stack(full_samples, axis=0)

    x2_imputed = x2_with_nan.cpu().numpy().copy()
    x2_imputed[miss_mask.cpu().numpy()] = miss_samples_arr.mean(axis=0)

    return (
        x2_imputed,
        miss_samples_arr,
        full_samples_arr,
        accept_rate_z,
        accept_rate_u,
        accept_rate_zp,
    )
