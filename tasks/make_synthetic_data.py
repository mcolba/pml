# -*- coding: utf-8 -*-
"""
VAR(1)-Diagonal GARCH(1,1) Factor Engine.

Mean (VAR(1)):
    y_t = mu + B y_{t-1} + e_t

Innovations:
    e_t = D_t z_t,  z_t ~ N(0, R)
    D_t = diag(sqrt(h_{1,t}), ..., sqrt(h_{k,t}))

Diagonal MGARCH(1,1) (no volatility spillovers across factors):
    h_{i,t} = omega_i + alpha_i * e_{i,t-1}^2 + beta_i * h_{i,t-1}

Notes
- Spillovers are ruled out in the volatility recursion via diagonal A and G (alpha/beta).

References
- Diebold, F.X., Rudebusch, G.D., and Aruoba, S.B. (2006). "The Macroeconomy and the Yield Curve: A Dynamic Latent Factor Approach." https://www.frbsf.org/wp-content/uploads/wp03-18bk.pdf


"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.graphics.tsaplots import plot_acf

AR_PARAMS = np.array(
    [
        [0.995, 0.0, 0.0],
        [0.0, 0.99, 0.0],
        [0.0, 0.0, 0.98],
    ],
    dtype=float,
)

GARCH_PARAMS_BASE = {
    "omega": np.array([0.0025, 0.005, 0.01], dtype=float),
    "alpha": np.array([0.0, 0, 0], dtype=float),
    "beta": np.array([0.0, 0, 0], dtype=float),
}

GARCH_PARAMS_HVOL = {
    "omega": np.array([0.00002, 0.005, 0.01], dtype=float),
    "alpha": np.array([0.147, 0, 0], dtype=float),
    "beta": np.array([0.85, 0, 0], dtype=float),
}

# =========================== Configuration =========================== #
NS_TAU = 1

ar_params = AR_PARAMS
use_local_scaling = False
garch_params = GARCH_PARAMS_HVOL
output_file_name = "ns_curves_hvol.csv"

# ===================================================================== #


@dataclass
class NelsonSiegelParameters:
    tau: float  # Decay factor
    beta0: float  # Long-term level parameter
    beta1: float  # Short-term component parameter
    beta2: float  # Medium-term component parameter


def nelson_siegel(
    params: NelsonSiegelParameters, maturities: np.ndarray
) -> list[float]:
    """Generate a yield curve using the Nelson-Siegel model.

    Parameters
    ----------
    params : NelsonSiegelParameters
        Nelson-Siegel parameters (tau, beta0, beta1, beta2).
    maturities : array_like
        Maturities (in years) at which to compute the yields.

    Returns
    -------
    list[float]
        Yields corresponding to the input maturities.
    """

    maturities = np.asarray(maturities, dtype=float)
    tau = params.tau
    beta0 = params.beta0
    beta1 = params.beta1
    beta2 = params.beta2

    yields = []
    for m in maturities:
        term1 = beta0
        # Guard against division by (very) small m
        if m == 0:
            factor = 1.0
            exp_term = 1.0
        else:
            factor = (1.0 - np.exp(-m / tau)) / (m / tau)
            exp_term = np.exp(-m / tau)

        term2 = beta1 * factor
        term3 = beta2 * (factor - exp_term)
        y = term1 + term2 + term3
        yields.append(y)

    return yields


@dataclass
class MGARCHParams:
    """Diagonal MGARCH(1,1) parameters.

    For each factor i:
        h_{i,t} = omega_i + alpha_i * e_{i,t-1}^2 + beta_i * h_{i,t-1}

    The cross-factor dependence is allowed only through the correlation matrix R of z_t.
    """

    omega: np.ndarray
    alpha: np.ndarray = field(default_factory=lambda: np.broadcast_to(0, omega.shape))
    beta: np.ndarray = field(default_factory=lambda: np.broadcast_to(0, omega.shape))
    corr: np.ndarray | None = None

    def __post_init__(self):
        """Validate and normalize GARCH parameters."""

        self.omega = np.atleast_1d(self.omega)
        k = self.omega.shape[0]

        self.alpha = np.broadcast_to(self.alpha, self.omega.shape)
        self.beta = np.broadcast_to(self.beta, self.omega.shape)

        # Basic stationarity/positivity checks for diagonal GARCH(1,1)
        if np.any(self.omega <= 0):
            raise ValueError("omega must be positive")
        if np.any(self.alpha < 0) or np.any(self.beta < 0):
            raise ValueError("alpha and beta must be non-negative")
        if np.any(self.alpha + self.beta >= 1):
            raise ValueError("Require alpha_i + beta_i < 1 for all i")

        # Correlation matrix handling
        if self.corr is None:
            self.corr = np.eye(k)


@dataclass
class VARParams:
    """Parameters for VAR(1) mean equation.

    y_t = mu + B y_{t-1} + e_t
    """

    mu: np.ndarray
    B: np.ndarray

    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=float)
        self.B = np.asarray(self.B, dtype=float)

        if self.B.ndim != 2 or self.B.shape[0] != self.B.shape[1]:
            raise ValueError("B must be a square matrix")
        if self.mu.shape != (self.B.shape[0],):
            raise ValueError("mu must have dimension k")


class FactorEngine:
    """VAR(1)-GARCH(1,1) simulator.

    Inputs
    - mu: (k,) intercept vector
    - B:  (k,k) VAR(1) autoregressive coefficient matrix
    - garch_params: MGARCHParams (omega, alpha, beta, corr=R)

    The volatility recursion is diagonal (no volatility spillovers).
    Mean spillovers are allowed unless you pass B as diagonal.
    """

    def __init__(self, var_params: VARParams, garch_params: MGARCHParams):
        self.var = var_params
        self.mu = self.var.mu
        self.B = self.var.B
        self.k = self.B.shape[0]

        if self.mu.shape != (self.k,):
            raise ValueError("mu must have dimension k")

        if not isinstance(garch_params, MGARCHParams):
            raise TypeError("garch_params must be MGARCHParams")
        if garch_params.omega.shape[0] != self.k:
            raise ValueError("GARCH parameter dimension must match k")

        self.garch = garch_params

    def simulate(self, n_sim: int, n_burn: int = 0, local_scaling: bool = False):
        """Simulate T observations (after burn-in). Returns (y, h)."""
        if n_sim <= 0:
            raise ValueError("T must be positive")
        if n_burn < 0:
            raise ValueError("burnin must be non-negative")

        k = self.k
        omega = self.garch.omega
        alpha = self.garch.alpha
        beta = self.garch.beta

        y = np.zeros((n_sim + n_burn, k))
        h = np.zeros((n_sim + n_burn, k))
        e = np.zeros((n_sim + n_burn, k))

        chol_R = np.linalg.cholesky(self.garch.corr)

        # Initialise h_0 and y_0 at their unconditional
        h0 = omega / (1.0 - alpha - beta)
        h[0, :] = h0
        y0 = np.linalg.inv(np.eye(k) - self.B) @ self.mu
        y[0, :] = y0

        ls = np.array([1.0, 1.0, 1.0])

        for t in range(1, n_sim + n_burn):
            if local_scaling:
                ls = y[t - 1, 0].clip(2, 6) / y0[0]

            z = chol_R @ np.random.randn(k)
            h[t, :] = omega + alpha * (e[t - 1, :] ** 2) + beta * h[t - 1, :]
            e[t, :] = np.sqrt(h[t, :]) * z
            y[t, :] = self.mu + self.B @ y[t - 1, :] + ls * e[t, :]

        # plt.plot(h[5000:7500, 0])
        # plot_acf((e[:, 0]) ** 2, title="Squared innovations ACF for factor 1")
        # plot_acf((y[1:, 0] - y[:-1, 0]) ** 2, title="Squared returns ACF for factor 1")

        return y[n_burn:], h[n_burn:]


if __name__ == "__main__":
    np.random.seed(42)

    # --- Simulation settings (similar structure to the reference snippet) ---
    n_steps = 10000
    n_burn = 250

    # VAR(1) mean equation: y_t = mu + B y_{t-1} + e_t
    long_run_mean = np.array([5, -2, -0.5], dtype=float)
    B = np.array(
        [
            [0.995, 0.0, 0.0],
            [0.0, 0.99, 0.0],
            [0.0, 0.0, 0.98],
        ],
        dtype=float,
    )
    mu = ((np.eye(B.shape[0]) - B) @ long_run_mean[:, None]).ravel()
    R = np.diag([1.0, 1.0, 1.0])

    omega, alpha, beta = garch_params.values()

    var_params = VARParams(mu=mu, B=B)
    garch_params = MGARCHParams(omega=omega, alpha=alpha, beta=beta, corr=R)

    # Run simulation
    engine = FactorEngine(var_params=var_params, garch_params=garch_params)
    factors, variances = engine.simulate(
        n_sim=n_steps, n_burn=n_burn, local_scaling=use_local_scaling
    )

    # --- Plots (time-series with long-run means) ---
    time_steps = np.arange(n_steps)
    time_years = time_steps / 250.0

    plt.figure(figsize=(10, 6))
    plt.plot(time_years, factors[:, 0], color="C0", label="factor 1")
    plt.plot(time_years, factors[:, 1], color="C1", label="factor 2")
    plt.plot(time_years, factors[:, 2], color="C2", label="factor 3")

    # Long-run means
    plt.axhline(long_run_mean[0], color="C0", linestyle="--")
    plt.axhline(long_run_mean[1], color="C1", linestyle="--")
    plt.axhline(long_run_mean[2], color="C2", linestyle="--")

    plt.xlabel("Time (years)")
    plt.ylabel("Factor value")
    plt.legend()

    # Optional: conditional volatility time-series

    def basis1(t):
        t = np.atleast_1d(t)
        return np.repeat(1, t.shape)

    def basis2(t):
        return (1.0 - np.exp(-t / NS_TAU)) / (t / NS_TAU)

    def basis3(t):
        return basis2(t) - np.exp(-t / NS_TAU)

    plt.figure(figsize=(10, 6))
    plt.plot(time_years, np.sqrt(variances[:, 0] * basis1(1)), label="Beta1")
    plt.plot(time_years, np.sqrt(variances[:, 1] * basis2(1)), label="Beta2")
    plt.plot(time_years, np.sqrt(variances[:, 2] * basis3(1)), label="Beta3")
    plt.ylim(0, 0.3)
    plt.xlabel("Time (years)")
    plt.ylabel("sigma(beta_i) x base_func_i(t=1Y)")
    plt.title("Conditional volatility time series")
    plt.legend()

    # --- Nelson-Siegel yield curves from simulated factors ---
    maturities = np.linspace(0.01, 5.0, 100)

    # Plot basis functions
    fig, ax = plt.subplots()
    plt.plot(maturities, basis1(maturities))
    plt.plot(maturities, basis2(maturities))
    plt.plot(maturities, basis3(maturities))
    plt.title("Nelson-Siegel Basis Functions")
    plt.xlabel("Time (years)")
    plt.show()

    curves = [
        nelson_siegel(
            NelsonSiegelParameters(
                tau=NS_TAU,
                beta0=factors[t, 0],
                beta1=factors[t, 1],
                beta2=factors[t, 2],
            ),
            maturities,
        )
        for t in range(n_steps)
    ]

    # 3x2 grid of subplots, each with 5 consecutive curves
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    window = 5  # number of consecutive curves per subplot
    max_start = n_steps - window
    start_indices = [
        0,
        int(n_steps * 0.2),
        int(n_steps * 0.4),
        int(n_steps * 0.6),
        int(n_steps * 0.8),
        max_start,
    ]

    for i, start in enumerate(start_indices):
        ax = axes[i]
        for offset in range(window):
            t_idx = start + offset
            ax.plot(maturities, curves[t_idx], label=f"t={t_idx}")
        ax.set_title(f"Simulation {start}-{start + window - 1}")
        if i >= 4:
            ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Yield")
    axes[0].legend()

    # --- IR yield time series at selected maturities ---
    maturities_ts = [0.5, 2.0, 5.0]  # in years
    y_6m, y_2y, y_10y = [], [], []

    for t in range(n_steps):
        params_t = NelsonSiegelParameters(
            tau=NS_TAU,
            beta0=factors[t, 0],
            beta1=factors[t, 1],
            beta2=factors[t, 2],
        )
        y_t = nelson_siegel(params_t, maturities_ts)
        y_6m.append(y_t[0])
        y_2y.append(y_t[1])
        y_10y.append(y_t[2])

    y_6m = np.array(y_6m)
    y_2y = np.array(y_2y)
    y_10y = np.array(y_10y)

    plt.figure(figsize=(10, 6))
    plt.plot(time_years, y_6m, label="0.5Y yield")
    plt.plot(time_years, y_2y, label="2Y yield")
    plt.plot(time_years, y_10y, label="10Y yield")
    plt.xlabel("Time (years)")
    plt.ylabel("Yield")
    plt.legend()

    plt.show()
    import pandas as pd
    from sklearn.decomposition import PCA

    # PCA on the simulated yield curves (observations × maturities)
    curves_ret = np.asarray(curves[1:]) - np.asarray(curves[:-1])
    pca = PCA(n_components=4)
    _ = pca.fit(curves_ret)

    print(
        "PCA explained variance ratio:",
        ", ".join(f"{x:.1%}" for x in pca.explained_variance_ratio_),
    )

    evr = pca.explained_variance_ratio_
    pc_labels = [f"PC{i + 1} ({evr[i]:.1%})" for i in range(len(evr))]

    loadings = pd.DataFrame(
        pca.components_.T,  # features × PCs
        columns=pc_labels,
        index=maturities,
    )

    ax = loadings.plot(figsize=(6, 6))
    ax.set_title("PCA factor loadings by maturity")
    ax.set_xlabel("Maturity (year fraction)")
    ax.set_ylabel("Loading")
    plt.show()

    # Rolling std dev of 0.5Y yield
    dy = np.array(y_2y[1:]) - np.array(y_2y[:-1])
    plt.scatter(
        y=sliding_window_view(dy, 50).std(axis=1),
        x=sliding_window_view(y_2y[1:], 50)[:, -1],
    )
    plt.title("Rolling 50-day std dev of 2Y yield")
    plt.xlabel("2Y yield")
    plt.ylabel("Rolling std dev")
    plt.show()

    # Plot ACF
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))
    plot_acf((y_6m[1:] - y_6m[:-1]) ** 2, title="Returns ACF for 6M yield", ax=ax1)
    plot_acf((y_2y[1:] - y_2y[:-1]) ** 2, title="Returns ACF for 2Y yield", ax=ax2)
    plot_acf((y_10y[1:] - y_10y[:-1]) ** 2, title="Returns ACF for 10Y yield", ax=ax3)
    plt.tight_layout()
    plt.show()

    # Save output
    maturities_out = [0.01, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]
    labels = [f"{m:.2f}Y" for m in maturities_out]

    curves_out = [
        nelson_siegel(par, maturities_out)
        for par in [
            NelsonSiegelParameters(
                tau=NS_TAU,
                beta0=factors[t, 0],
                beta1=factors[t, 1],
                beta2=factors[t, 2],
            )
            for t in range(n_steps)
        ]
    ]

    df = pd.DataFrame(curves_out, columns=labels)
    df.to_csv(
        Path(__file__).parent.parent / "data" / output_file_name,
        index=False,
        mode="x",
    )
