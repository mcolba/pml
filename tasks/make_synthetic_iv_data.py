"""Generate synthetic IV surface data for end-to-end HVAE testing.

Uses the SVI (Stochastic Volatility Inspired) parameterisation to produce
implied-variance slices that are then converted to implied vol.  Random
walks on the SVI parameters create day-to-day variation.

Outputs a long-format CSV with columns:
    date, name, strike, tau, tau_label, type, value
"""

from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

# ── grid ─────────────────────────────────────────────────────────────────
NAMES = ["OMX", "NAME1", "NAME2", "NAME3"]
STRIKES = np.array([0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20])
TAUS = np.array([0.08, 0.25, 0.50, 1.00])
TAU_LABELS = ["1M", "3M", "6M", "1Y"]
N_DATES = 500

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "synthetic_iv_surfaces.csv"


# ── SVI total-variance slice ─────────────────────────────────────────────
def svi_total_var(
    k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float
) -> np.ndarray:
    """Raw SVI: w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    km = k - m
    return a + b * (rho * km + np.sqrt(km**2 + sigma**2))


def svi_iv(k: np.ndarray, tau: float, a, b, rho, m, sigma) -> np.ndarray:
    """Implied vol from SVI total-variance."""
    w = svi_total_var(k, a, b, rho, m, sigma)
    w = np.clip(w, 1e-6, None)
    return np.sqrt(w / tau)


# ── random SVI parameter paths ──────────────────────────────────────────
def _random_svi_params(n: int, base: dict | None = None) -> dict:
    """Generate a random-walk path of SVI parameters."""
    base = base or {}
    a0 = base.get("a", 0.04)
    b0 = base.get("b", 0.15)
    rho0 = base.get("rho", -0.3)
    m0 = base.get("m", 0.0)
    sigma0 = base.get("sigma", 0.15)

    def _rw(start, vol, lo, hi):
        path = np.empty(n)
        path[0] = start
        for t in range(1, n):
            path[t] = np.clip(path[t - 1] + np.random.normal(0, vol), lo, hi)
        return path

    return {
        "a": _rw(a0, 0.002, 0.005, 0.15),
        "b": _rw(b0, 0.003, 0.02, 0.40),
        "rho": _rw(rho0, 0.01, -0.90, 0.0),
        "m": _rw(m0, 0.005, -0.20, 0.20),
        "sigma": _rw(sigma0, 0.003, 0.05, 0.40),
    }


# ── main ─────────────────────────────────────────────────────────────────
def generate() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=N_DATES, freq="B")
    log_strikes = np.log(STRIKES)
    rows: list[dict] = []

    base_params_per_name = {
        "OMX": {"a": 0.04, "b": 0.12, "rho": -0.35, "m": 0.0, "sigma": 0.15},
        "NAME1": {"a": 0.05, "b": 0.18, "rho": -0.25, "m": 0.02, "sigma": 0.18},
        "NAME2": {"a": 0.06, "b": 0.20, "rho": -0.40, "m": -0.01, "sigma": 0.12},
        "NAME3": {"a": 0.03, "b": 0.14, "rho": -0.30, "m": 0.0, "sigma": 0.20},
    }

    for name in NAMES:
        # one SVI parameter path per (name, tau)
        for j, (tau, tau_lbl) in enumerate(zip(TAUS, TAU_LABELS)):
            params = _random_svi_params(N_DATES, base_params_per_name[name])
            for t_idx, dt in enumerate(dates):
                iv = svi_iv(
                    log_strikes,
                    tau,
                    params["a"][t_idx],
                    params["b"][t_idx],
                    params["rho"][t_idx],
                    params["m"][t_idx],
                    params["sigma"][t_idx],
                )
                for k_idx, (strike, iv_val) in enumerate(zip(STRIKES, iv)):
                    rows.append(
                        {
                            "date": dt.strftime("%Y-%m-%d"),
                            "name": name,
                            "strike": strike,
                            "tau": tau,
                            "tau_label": tau_lbl,
                            "type": "IVS",
                            "value": round(float(iv_val), 6),
                        }
                    )

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df)} rows  →  {OUTPUT}")
    print(df.groupby("name")["date"].nunique())
