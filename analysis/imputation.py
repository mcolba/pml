"""Compare MCMC imputation methods across all VAE variants.

Trains each model on a synthetic DGP, then imputes a held-out parent group.
The column-mean baseline uses training groups only. Prints MSE and acceptance
rates for exploratory comparison; pass/fail methodology checks live in tests.

Usage:
    python -m analysis.imputation
"""

from __future__ import annotations

import time

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from src.imputation import (
    cvae_mcmc_impute,
    hvae_mcmc_impute,
    hvae_mcmc_impute_approx,
    hvae_mcmc_impute_sequential,
    hvae_mcmc_impute_sequential_full,
    hvae_ps_mcmc_impute,
    hvae_ps_mcmc_impute_sequential,
    vae_mcmc_impute,
)
from src.vae.cvae import CVAE
from src.vae.hvae import HierarchicalVAE
from src.vae._deprecated.hvae_ps import HierarchicalPSVAE
from src.vae.utils.set_transformer import MeanPoolSetEncoder
from src.vae.vae import VAE
from tests.integration.data.dgp import (
    generate_conditional_hierarchical_sets,
    sample_bimodal_latents,
    sample_gaussian_latents,
)

# ============================================================================
# Config
# ============================================================================

N_OBS = 2000
N_CHILDREN = 50
MCMC_STEPS = 2000
BURN_IN = 100
SVI_STEPS = 4000
HIDDEN = 64
X1_DIM = 10
X2_DIM = 10
C_DIM = 1
MISS_FRAC = 0.9
TRAIN_FRACTION = 0.8
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================================
# Helpers
# ============================================================================


def train_svi(model_fn, guide_fn, data_batches, lr=1e-3, steps=SVI_STEPS, patience=50):
    optimizer = Adam({"lr": lr})
    svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
    best_loss = float("inf")
    wait = 0
    for epoch in range(steps):
        epoch_loss = 0.0
        for batch in data_batches:
            epoch_loss += svi.step(*batch)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {epoch + 1} (patience={patience})")
                return
    print(f"Completed {steps} epochs")


def _to_torch_dataset(x1, x2, c):
    return (
        torch.tensor(x1, dtype=torch.float32),
        torch.tensor(x2, dtype=torch.float32),
        torch.tensor(c, dtype=torch.float32),
    )


def _make_shared_hierarchical_dgp(n, latent_sampler, *, n_children=50, seed=SEED):
    x1, x2, c = generate_conditional_hierarchical_sets(
        n_datasets=n,
        latent_sampler=latent_sampler,
        n_conditions=n_children,
        x_dim=X1_DIM,
        z_dim=2,
        seed=seed,
    )
    return _to_torch_dataset(x1, x2, c)


def _make_identity_prior_hierarchical_dgp(n, *, n_children=50, seed=SEED):
    x1, x2, c = generate_conditional_hierarchical_sets(
        n_datasets=n,
        latent_sampler=sample_gaussian_latents,
        n_conditions=n_children,
        x_dim=X1_DIM,
        z_dim=2,
        cov=np.eye(2),
        seed=seed,
    )
    return _to_torch_dataset(x1, x2, c)


def make_gaussian_hierarchical_dgp(n, n_children=50, seed=SEED):
    """Hierarchical DGP with a 2-D Gaussian parent latent."""
    return _make_shared_hierarchical_dgp(
        n,
        sample_gaussian_latents,
        n_children=n_children,
        seed=seed,
    )


def make_exact_prior_hierarchical_dgp(n, n_children=50, seed=SEED):
    """Hierarchical DGP with z1 ~ N(0, I), matching the exact prior."""
    return _make_identity_prior_hierarchical_dgp(
        n,
        n_children=n_children,
        seed=seed,
    )


def make_multimodal_hierarchical_dgp(n, n_children=50, seed=SEED):
    """Hierarchical DGP with a 2-D bimodal parent latent."""
    return _make_shared_hierarchical_dgp(
        n,
        sample_bimodal_latents,
        n_children=n_children,
        seed=seed,
    )


def inject_nan(x, frac, seed=123):
    rng = np.random.RandomState(seed)
    x_obs = x[0].clone()
    n_miss = max(1, int(x_obs.shape[0] * frac))
    miss_idx = torch.tensor(rng.choice(x_obs.shape[0], n_miss, replace=False))
    true_vals = x_obs[miss_idx].clone()
    x_obs[miss_idx] = float("nan")
    return x_obs, true_vals, miss_idx


def mse(pred, true):
    return float(np.mean((pred - true) ** 2))


def naive_baseline_mse(x_train, miss_idx, true_vals):
    col_means = x_train.mean(dim=0)
    return mse(col_means[miss_idx].numpy(), true_vals.numpy())


def split_parent_groups(data):
    """Keep every child of a parent in the same train or held-out partition."""
    n_groups = data[0].shape[0]
    if any(tensor.shape[0] != n_groups for tensor in data):
        raise ValueError("All parent, child, and condition tensors must align")
    n_train = int(n_groups * TRAIN_FRACTION)
    if not 0 < n_train < n_groups:
        raise ValueError("At least one train and one held-out group are required")
    return tuple(tensor[:n_train] for tensor in data), tuple(
        tensor[n_train:] for tensor in data
    )


def fmt_rates(rates: dict[str, float]) -> str:
    parts = []
    for k, v in rates.items():
        parts.append(f"acc({k})={v:.1%}  rej({k})={1 - v:.1%}")
    return "  ".join(parts)


def print_header(title: str):
    w = 70
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def print_result(name: str, mse_val: float, mse_base: float, acc: dict, elapsed: float):
    ratio = mse_val / mse_base if mse_base > 0 else float("inf")
    marker = "OK" if ratio < 1.0 else "WORSE"
    print(
        f"  {name:<30s}  MSE={mse_val:.6f}  ratio={ratio:.3f}  [{marker}]  "
        f"({elapsed:.1f}s)"
    )
    print(f"    {fmt_rates(acc)}")


def mh_label(use_mh: bool) -> str:
    return "mh" if use_mh else "no_mh"


# ============================================================================
# 1. Flat VAE
# ============================================================================


def run_vae(data, dgp_label=""):
    (_, x2_train, _), (_, x2_test, _) = split_parent_groups(data)
    x2_flat = x2_train.reshape(-1, X2_DIM)
    x2_test_flat = x2_test.reshape(-1, X2_DIM)
    print_header(f"VAE  (flat, unconditional)  [{dgp_label}]")
    pyro.clear_param_store()

    vae = VAE(x_dim=X2_DIM, z_dim=2, hidden_dim=HIDDEN)
    print("  Training VAE ...")
    train_svi(vae.model, vae.guide, [(x2_flat,)])

    x_nan, true_vals, miss_idx = inject_nan(x2_test_flat, MISS_FRAC)
    mse_base = naive_baseline_mse(x2_flat, miss_idx, true_vals)
    print(f"  Naive baseline MSE = {mse_base:.6f}")

    pfx = f"{dgp_label} " if dgp_label else ""
    results = {f"Naive ({pfx}VAE)": mse_base}

    t0 = time.time()
    x_imp, _, acc = vae_mcmc_impute(
        vae,
        x_nan,
        num_steps=MCMC_STEPS,
        burn_in=BURN_IN,
        use_mh=True,
    )
    elapsed = time.time() - t0
    mse_val = mse(x_imp[miss_idx.numpy()], true_vals.numpy())
    rates = {"z": acc}
    print_result("vae_mcmc_impute [mh]", mse_val, mse_base, rates, elapsed)
    results[f"{pfx}VAE mh"] = (mse_val, rates)

    t0 = time.time()
    x_imp, _, acc = vae_mcmc_impute(
        vae,
        x_nan,
        num_steps=MCMC_STEPS,
        burn_in=BURN_IN,
        use_mh=False,
    )
    elapsed = time.time() - t0
    mse_val = mse(x_imp[miss_idx.numpy()], true_vals.numpy())
    rates = {"z": acc}
    print_result("vae_mcmc_impute [no_mh]", mse_val, mse_base, rates, elapsed)
    results[f"{pfx}VAE no_mh"] = (mse_val, rates)

    return results


# ============================================================================
# 2. CVAE
# ============================================================================


def run_cvae(data, dgp_label=""):
    (_, x2_train, c_train), (_, x2_test, c_test) = split_parent_groups(data)
    x2_flat = x2_train.reshape(-1, X2_DIM)
    c_flat = c_train.reshape(-1, C_DIM)
    x2_test_flat = x2_test.reshape(-1, X2_DIM)
    c_test_flat = c_test.reshape(-1, C_DIM)
    print_header(f"CVAE  (conditional)  [{dgp_label}]")
    pyro.clear_param_store()

    vae = CVAE(x_dim=X2_DIM, c_dim=C_DIM, z_dim=2, hidden_dim=HIDDEN)
    print("  Training CVAE ...")
    train_svi(vae.model, vae.guide, [(x2_flat, c_flat)])

    x_nan, true_vals, miss_idx = inject_nan(x2_test_flat, MISS_FRAC)
    c_obs = c_test_flat[0]
    mse_base = naive_baseline_mse(x2_flat, miss_idx, true_vals)
    print(f"  Naive baseline MSE = {mse_base:.6f}")

    pfx = f"{dgp_label} " if dgp_label else ""
    results = {f"Naive ({pfx}CVAE)": mse_base}

    t0 = time.time()
    x_imp, _, acc = cvae_mcmc_impute(
        vae,
        x_nan,
        c_obs,
        num_steps=MCMC_STEPS,
        burn_in=BURN_IN,
        use_mh=True,
    )
    elapsed = time.time() - t0
    mse_val = mse(x_imp[miss_idx.numpy()], true_vals.numpy())
    rates = {"z": acc}
    print_result("cvae_mcmc_impute [mh]", mse_val, mse_base, rates, elapsed)
    results[f"{pfx}CVAE mh"] = (mse_val, rates)

    t0 = time.time()
    x_imp, _, acc = cvae_mcmc_impute(
        vae,
        x_nan,
        c_obs,
        num_steps=MCMC_STEPS,
        burn_in=BURN_IN,
        use_mh=False,
    )
    elapsed = time.time() - t0
    mse_val = mse(x_imp[miss_idx.numpy()], true_vals.numpy())
    rates = {"z": acc}
    print_result("cvae_mcmc_impute [no_mh]", mse_val, mse_base, rates, elapsed)
    results[f"{pfx}CVAE no_mh"] = (mse_val, rates)

    return results


# ============================================================================
# 3. Hierarchical VAE  (joint / sequential / approx)
# ============================================================================


def run_hvae(data, use_set_encoder=False, dgp_label=""):
    (x1, x2, c), (x1_test, x2_test, c_test) = split_parent_groups(data)
    tag = " + set(x2,c2)" if use_set_encoder else ""
    print_header(f"HVAE  (hierarchical, 2-latent{tag})  [{dgp_label}]")
    pyro.clear_param_store()

    n_children = x2.shape[1]
    mask = torch.ones(x1.shape[0], n_children)

    set_encoder = (
        MeanPoolSetEncoder(input_dim=X2_DIM + C_DIM, d_model=HIDDEN, output_dim=5)
        if use_set_encoder
        else None
    )
    vae = HierarchicalVAE(
        x1_dim=X1_DIM,
        x2_dim=X2_DIM,
        c2_dim=C_DIM,
        z1_dim=2,
        z2_dim=2,
        hidden_dim=HIDDEN,
        set_encoder=set_encoder,
    )
    print("  Training HVAE ...")
    train_svi(vae.model, vae.guide, [(x1, x2, c, mask)])

    x2_flat = x2.reshape(-1, X2_DIM)
    x2_nan, true_vals, miss_idx = inject_nan(x2_test[:, 0, :], MISS_FRAC)
    x1_obs, c_obs = x1_test[0], c_test[0, 0]
    # Only pass sibling observations/conditions when the encoder uses the set summary.
    x2_siblings = (
        x2_test[0, 1:, :].unsqueeze(0)
        if use_set_encoder and n_children > 1
        else None
    )
    c2_siblings = (
        c_test[0, 1:, :].unsqueeze(0)
        if use_set_encoder and n_children > 1
        else None
    )
    mse_base = naive_baseline_mse(x2_flat, miss_idx, true_vals)
    print(f"  Naive baseline MSE = {mse_base:.6f}")

    pfx = (
        (f"{dgp_label} " if dgp_label else "")
        + "HVAE"
        + (" attn" if use_set_encoder else "")
    )
    results = {}

    t0 = time.time()
    res = hvae_mcmc_impute(
        vae,
        x1_obs,
        x2_nan,
        c_obs,
        num_steps=MCMC_STEPS,
        burn_in=BURN_IN,
        x2_siblings=x2_siblings,
        c2_siblings=c2_siblings,
        use_mh=True,
    )
    elapsed = time.time() - t0
    mse_val = mse(res[0][miss_idx.numpy()], true_vals.numpy())
    rates = {"(z,u)": res[3]}
    print_result("hvae_joint [mh]", mse_val, mse_base, rates, elapsed)
    results[f"{pfx} joint mh"] = (mse_val, rates)

    t0 = time.time()
    res = hvae_mcmc_impute_sequential(
        vae,
        x1_obs,
        x2_nan,
        c_obs,
        num_steps=MCMC_STEPS,
        burn_in=BURN_IN,
        x2_siblings=x2_siblings,
        c2_siblings=c2_siblings,
        use_mh=True,
    )
    elapsed = time.time() - t0
    mse_val = mse(res[0][miss_idx.numpy()], true_vals.numpy())
    rates = {"z": res[3], "u": res[4]}
    print_result("hvae_sequential [mh]", mse_val, mse_base, rates, elapsed)
    results[f"{pfx} seq mh"] = (mse_val, rates)

    t0 = time.time()
    res = hvae_mcmc_impute_approx(
        vae,
        x1_obs,
        x2_nan,
        c_obs,
        num_steps=MCMC_STEPS,
        burn_in=BURN_IN,
        x2_siblings=x2_siblings,
        c2_siblings=c2_siblings,
        use_mh=True,
    )
    elapsed = time.time() - t0
    mse_val = mse(res[0][miss_idx.numpy()], true_vals.numpy())
    rates = {"u": res[3]}
    print_result("hvae_approx [mh]", mse_val, mse_base, rates, elapsed)
    results[f"{pfx} approx mh"] = (mse_val, rates)

    t0 = time.time()
    res = hvae_mcmc_impute(
        vae,
        x1_obs,
        x2_nan,
        c_obs,
        num_steps=MCMC_STEPS,
        burn_in=BURN_IN,
        x2_siblings=x2_siblings,
        c2_siblings=c2_siblings,
        use_mh=False,
    )
    elapsed = time.time() - t0
    mse_val = mse(res[0][miss_idx.numpy()], true_vals.numpy())
    rates = {"(z,u)": res[3]}
    print_result("hvae [no_mh]", mse_val, mse_base, rates, elapsed)
    results[f"{pfx} no_mh"] = (mse_val, rates)

    if use_set_encoder:
        # --- hvae_mcmc_impute_sequential_full: centered (use_noncentered=False) ---
        t0 = time.time()
        res = hvae_mcmc_impute_sequential_full(
            vae,
            x1_obs,
            x2_nan,
            c_obs,
            num_steps=MCMC_STEPS,
            burn_in=BURN_IN,
            x2_siblings=x2_siblings,
            c_siblings=c2_siblings,
            use_mh=True,
            use_noncentered=False,
        )
        elapsed = time.time() - t0
        mse_val = mse(res[0][miss_idx.numpy()], true_vals.numpy())
        rates = {"z": res[3], "u": res[4], "sib": res[5]}
        print_result("hvae_seq_full [mh, centered]", mse_val, mse_base, rates, elapsed)
        results[f"{pfx} seq mh exact"] = (mse_val, rates)

        # --- hvae_mcmc_impute_sequential_full: non-centered (use_noncentered=True) ---
        t0 = time.time()
        res = hvae_mcmc_impute_sequential_full(
            vae,
            x1_obs,
            x2_nan,
            c_obs,
            num_steps=MCMC_STEPS,
            burn_in=BURN_IN,
            x2_siblings=x2_siblings,
            c_siblings=c2_siblings,
            use_mh=True,
            use_noncentered=True,
        )
        elapsed = time.time() - t0
        mse_val = mse(res[0][miss_idx.numpy()], true_vals.numpy())
        rates = {"z": res[3], "u": res[4], "sib": res[5]}
        print_result(
            "hvae_seq_full [mh, noncentered]", mse_val, mse_base, rates, elapsed
        )
        results[f"{pfx} seq mh exact non-cntr"] = (mse_val, rates)

    results[f"Naive ({pfx})"] = mse_base
    return results


# ============================================================================
# 4. PS-HVAE  (joint / sequential)
# ============================================================================


def run_ps_hvae(data, use_set_encoder=False, dgp_label=""):
    (x1, x2, c), (x1_test, x2_test, c_test) = split_parent_groups(data)
    tag = " + set(x2,c2)" if use_set_encoder else ""
    print_header(f"PS-HVAE  (private-shared, 2-latent{tag})  [{dgp_label}]")
    pyro.clear_param_store()

    n_children = x2.shape[1]
    mask = torch.ones(x1.shape[0], n_children)

    set_encoder = (
        MeanPoolSetEncoder(input_dim=X2_DIM + C_DIM, d_model=HIDDEN, output_dim=5)
        if use_set_encoder
        else None
    )
    vae = HierarchicalPSVAE(
        x1_dim=X1_DIM,
        x2_dim=X2_DIM,
        c2_dim=C_DIM,
        z1_dim=2,
        z2_dim=2,
        z_private_dim=1,
        hidden_dim=HIDDEN,
        set_encoder=set_encoder,
    )
    print("  Training PS-HVAE ...")
    train_svi(vae.model, vae.guide, [(x1, x2, c, mask)])

    x2_flat = x2.reshape(-1, X2_DIM)
    x2_nan, true_vals, miss_idx = inject_nan(x2_test[:, 0, :], MISS_FRAC)
    x1_obs, c_obs = x1_test[0], c_test[0, 0]
    # Only pass sibling observations/conditions when the encoder uses the set summary.
    x2_siblings = (
        x2_test[0, 1:, :].unsqueeze(0)
        if use_set_encoder and n_children > 1
        else None
    )
    c2_siblings = (
        c_test[0, 1:, :].unsqueeze(0)
        if use_set_encoder and n_children > 1
        else None
    )
    mse_base = naive_baseline_mse(x2_flat, miss_idx, true_vals)
    print(f"  Naive baseline MSE = {mse_base:.6f}")

    pfx = (
        (f"{dgp_label} " if dgp_label else "")
        + "PS-HVAE"
        + (" attn" if use_set_encoder else "")
    )
    results = {}

    t0 = time.time()
    res = hvae_ps_mcmc_impute(
        vae,
        x1_obs,
        x2_nan,
        c_obs,
        num_steps=MCMC_STEPS,
        burn_in=BURN_IN,
        x2_siblings=x2_siblings,
        c2_siblings=c2_siblings,
        use_mh=True,
    )
    elapsed = time.time() - t0
    mse_val = mse(res[0][miss_idx.numpy()], true_vals.numpy())
    rates = {"(z,u,zp)": res[3]}
    print_result("ps_hvae_joint [mh]", mse_val, mse_base, rates, elapsed)
    results[f"{pfx} joint mh"] = (mse_val, rates)

    t0 = time.time()
    res = hvae_ps_mcmc_impute_sequential(
        vae,
        x1_obs,
        x2_nan,
        c_obs,
        num_steps=MCMC_STEPS,
        burn_in=BURN_IN,
        x2_siblings=x2_siblings,
        c2_siblings=c2_siblings,
        use_mh=True,
    )
    elapsed = time.time() - t0
    mse_val = mse(res[0][miss_idx.numpy()], true_vals.numpy())
    rates = {"z": res[3], "u": res[4], "zp": res[5]}
    print_result("ps_hvae_sequential [mh]", mse_val, mse_base, rates, elapsed)
    results[f"{pfx} seq mh"] = (mse_val, rates)

    t0 = time.time()
    res = hvae_ps_mcmc_impute(
        vae,
        x1_obs,
        x2_nan,
        c_obs,
        num_steps=MCMC_STEPS,
        burn_in=BURN_IN,
        x2_siblings=x2_siblings,
        c2_siblings=c2_siblings,
        use_mh=False,
    )
    elapsed = time.time() - t0
    mse_val = mse(res[0][miss_idx.numpy()], true_vals.numpy())
    rates = {"(z,u,zp)": res[3]}
    print_result("ps_hvae_joint [no_mh]", mse_val, mse_base, rates, elapsed)
    results[f"{pfx} joint no_mh"] = (mse_val, rates)

    t0 = time.time()
    res = hvae_ps_mcmc_impute_sequential(
        vae,
        x1_obs,
        x2_nan,
        c_obs,
        num_steps=MCMC_STEPS,
        burn_in=BURN_IN,
        x2_siblings=x2_siblings,
        c2_siblings=c2_siblings,
        use_mh=False,
    )
    elapsed = time.time() - t0
    mse_val = mse(res[0][miss_idx.numpy()], true_vals.numpy())
    rates = {"z": res[3], "u": res[4], "zp": res[5]}
    print_result("ps_hvae_sequential [no_mh]", mse_val, mse_base, rates, elapsed)
    results[f"{pfx} seq no_mh"] = (mse_val, rates)

    results[f"Naive ({pfx})"] = mse_base
    return results


# ============================================================================
# Summary table
# ============================================================================


def _fmt_summary_rates(rates: dict[str, float]) -> str:
    return ", ".join(f"acc({k})={v:.1%}" for k, v in rates.items())


def print_summary(all_results: dict, dataset_labels: list[str]):
    for dataset_label in dataset_labels:
        print_header(f"SUMMARY — {dataset_label}")

        baselines = {
            k: v
            for k, v in all_results.items()
            if k.startswith("Naive") and dataset_label in k.lower()
        }
        models = {
            k: v
            for k, v in all_results.items()
            if not k.startswith("Naive") and k.lower().startswith(f"{dataset_label} ")
        }

        print(f"\n  {'Method':<35s}  {'MSE':>10s}  {'Rates'}")
        print(f"  {'-' * 35}  {'-' * 10}  {'-' * 50}")
        for name, (mse_val, rates) in sorted(models.items(), key=lambda kv: kv[1][0]):
            method_name = name[len(dataset_label) + 1 :]
            print(
                f"  {method_name:<35s}  {mse_val:>10.6f}  {_fmt_summary_rates(rates)}"
            )
        print()
        for name, val in sorted(baselines.items(), key=lambda kv: kv[1]):
            baseline_name = name.replace(f"Naive ({dataset_label} ", "").rstrip(")")
            print(f"  {'Naive ' + baseline_name:<35s}  {val:>10.6f}  (baseline)")
        print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    datasets = [
        (
            "exact_prior",
            make_exact_prior_hierarchical_dgp(N_OBS, N_CHILDREN, seed=SEED),
        ),
        ("gaussian", make_gaussian_hierarchical_dgp(N_OBS, N_CHILDREN, seed=SEED)),
        (
            "multimodal",
            make_multimodal_hierarchical_dgp(N_OBS, N_CHILDREN, seed=SEED),
        ),
    ]
    all_results: dict[str, float] = {}

    for label, data in datasets:
        all_results.update(run_vae(data, dgp_label=label))
        all_results.update(run_cvae(data, dgp_label=label))
        all_results.update(run_hvae(data, use_set_encoder=False, dgp_label=label))
        all_results.update(run_hvae(data, use_set_encoder=True, dgp_label=label))
        # all_results.update(run_ps_hvae(data, use_set_encoder=False, dgp_label=label))
        # all_results.update(run_ps_hvae(data, use_set_encoder=True, dgp_label=label))

    print_summary(all_results, [label for label, _ in datasets])
    print("Done.")
