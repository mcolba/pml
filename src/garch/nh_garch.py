# hngarch_pricer.py
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, pi, sqrt
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.integrate import quad
except Exception:  # pragma: no cover
    quad = None


@dataclass(frozen=True)
class HNGarchParams:
    # Notation consistent with the paper appendices:
    # a: GARCH MA parameter, b: GARCH AR parameter, om: intercept, gam: asymmetry, lam: risk premium
    a: float
    b: float
    om: float
    gam: float
    lam: float

    def gamstar_q(self) -> float:
        # Risk-neutral leverage parameter c* = c + λ + 1/2 (paper text). :contentReference[oaicite:6]{index=6}
        return self.gam + self.lam + 0.5


def bs_call_price(S: float, K: float, r: float, tau: float, sigma: float) -> float:
    """Black–Scholes European call price (continuous compounding)."""
    # Handle edge cases
    if tau <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        fwd = S * exp(r * tau)
        return exp(-r * tau) * max(fwd - K, 0.0)

    from math import erf

    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    return S * norm_cdf(d1) - K * exp(-r * tau) * norm_cdf(d2)


def hn_monte_carlo_call_appendix_a(
    S: float,
    K: float,
    r: float,  # per day, cont. comp. (paper uses r=0 in the snippet). :contentReference[oaicite:7]{index=7}
    T: int,  # days to maturity
    h1: float,  # initial conditional variance (daily)
    params: HNGarchParams,
    nsim: int = 10000,
    seed: Optional[int] = 1234,
) -> float:
    """
    Appendix A style:
    - simulate risk-neutral returns with exp(-0.5*h + sqrt(h)*z)
    - apply Duan & Simonato EMS (normalize by mean each step)
    - use BS control variate with homoskedastic path using same shocks
    :contentReference[oaicite:8]{index=8}
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(nsim, T))

    xt = np.ones(nsim, dtype=np.float64)
    xthomo = np.ones(nsim, dtype=np.float64)
    h = np.full(nsim, h1, dtype=np.float64)

    # The appendix computes sigma = sqrt(365*ht1) and uses BS with tau=T/365 and r*365.
    # We replicate that mapping. :contentReference[oaicite:9]{index=9}
    sigma = sqrt(365.0 * h1)
    bsprice = bs_call_price(S, K, r * 365.0, T / 365.0, sigma)

    gamstar = params.gamstar_q()

    for i in range(T):
        zi = z[:, i]

        # Risk-neutral one-step gross return multipliers (no risk premium in drift; uses -0.5*h). :contentReference[oaicite:10]{index=10}
        xt *= np.exp(-0.5 * h + np.sqrt(h) * zi)

        # Homoskedastic control variate path uses same shocks. :contentReference[oaicite:11]{index=11}
        xthomo *= np.exp(-(0.5 * sigma * sigma) / 365.0 + (sigma / sqrt(365.0)) * zi)

        mxt = float(np.mean(xt))
        mxthomo = float(np.mean(xthomo))

        # Appendix A has a guard if mean gets too small. :contentReference[oaicite:12]{index=12}
        if mxt < 0.5:
            raise RuntimeError("EMS failed: mean(xt) < 0.5 (matches Appendix A guard).")

        # Empirical martingale step (divide by mean). :contentReference[oaicite:13]{index=13}
        xt /= mxt
        xthomo /= mxthomo

        # Variance recursion (Appendix A): h = om + b*h + a*(z - sqrt(h)*gamstar)^2 :contentReference[oaicite:14]{index=14}
        h = params.om + params.b * h + params.a * (zi - np.sqrt(h) * gamstar) ** 2

    # Terminal prices (Appendix A uses xta=S*exp(r*T)*xt). :contentReference[oaicite:15]{index=15}
    ST = S * exp(r * T) * xt
    ST_homo = S * exp(r * T) * xthomo

    payoff = np.maximum(ST - K, 0.0)
    payoff_homo = np.maximum(ST_homo - K, 0.0)

    chgar = exp(-r * T) * float(np.mean(payoff))
    chgarhomo = exp(-r * T) * float(np.mean(payoff_homo))

    # Control variate combination: MC + BS - MC_homo :contentReference[oaicite:16]{index=16}
    return chgar + bsprice - chgarhomo


def _hn_integrand_appendix_b(
    phi: float,
    S: float,
    K: float,
    r: float,
    T: int,
    h1: float,
    a: float,
    b: float,
    om: float,
    lam_q: float,  # Appendix B passes lam = -0.5 into hnintres. :contentReference[oaicite:17]{index=17}
    gamstar: float,
) -> float:
    """
    Appendix B integrand hnintres.m / hnint:
    - compute f(i*phi+1) and f(i*phi)
    - backward recursion for A,B
    - combine the two integrands into fval
    :contentReference[oaicite:18]{index=18}
    """
    i = 1j
    ph = float(phi)

    # Öp = [i*Ö+1; i*Ö] :contentReference[oaicite:19]{index=19}
    op = np.array([i * ph + 1.0, i * ph], dtype=np.complex128)

    A = op * r
    B = lam_q * op + 0.5 * op * op

    # recurse backwards until time t=0 : for t=T-1:-1:1 :contentReference[oaicite:20]{index=20}
    for _t in range(T - 1, 0, -1):
        Ap = A
        Bp = B
        A = Ap + op * r + Bp * om - 0.5 * np.log(1.0 - 2.0 * a * Bp)
        B = (
            op * (lam_q + gamstar)
            - 0.5 * gamstar * gamstar
            + b * Bp
            + (0.5 * (op - gamstar) ** 2) / (1.0 - 2.0 * a * Bp)
        )

    f1 = (S ** op[0]) * np.exp(A[0] + B[0] * h1)
    f2 = (S ** op[1]) * np.exp(A[1] + B[1] * h1)

    # first integrand: real((K.^(-i*Ö)).*f1./(i*Ö)) :contentReference[oaicite:21]{index=21}
    f1val = np.real((K ** (-i * ph)) * f1 / (i * ph))
    # second integrand: real((K.^(-i*Ö)).*f2./(i*Ö)) :contentReference[oaicite:22]{index=22}
    f2val = np.real((K ** (-i * ph)) * f2 / (i * ph))

    # combined integrand: fval = (f1val/K - f2val)*exp(-r*T) :contentReference[oaicite:23]{index=23}
    return float((f1val / K - f2val) * exp(-r * T))


def hn_fourier_call_appendix_b(
    S: float,
    K: float,
    r: float,  # per day, cont. comp.
    T: int,
    h1: float,
    params: HNGarchParams,
    phi_lo: float = 1e-4,
    phi_hi: float = 1000.0,
    quad_epsabs: float = 1e-6,
    quad_epsrel: float = 1e-6,
) -> float:
    """
    Appendix B / Heston-Nandi closed-form (quasi closed-form) via Fourier inversion integral. :contentReference[oaicite:24]{index=24}
    Uses the same integrand and recursion; integrates from 0.0001 to 1000 as in the appendix. :contentReference[oaicite:25]{index=25}
    """
    if quad is None:  # pragma: no cover
        raise ImportError(
            "scipy is required for Fourier pricing (scipy.integrate.quad)."
        )

    gamstar = params.gamstar_q()

    # Appendix B explicitly passes lam = -0.5 into hnintres. :contentReference[oaicite:26]{index=26}
    lam_q = -0.5

    integr, _err = quad(
        lambda ph: _hn_integrand_appendix_b(
            ph,
            S=S,
            K=K,
            r=r,
            T=T,
            h1=h1,
            a=params.a,
            b=params.b,
            om=params.om,
            lam_q=lam_q,
            gamstar=gamstar,
        ),
        phi_lo,
        phi_hi,
        epsabs=quad_epsabs,
        epsrel=quad_epsrel,
        limit=200,
    )

    # op_price = .5*(S - K*exp(-r*T)) + K/pi*integr :contentReference[oaicite:27]{index=27}
    return 0.5 * (S - K * exp(-r * T)) + (K / pi) * float(integr)


# -----------------------
# test_hngarch_pricer.py
# -----------------------
def _default_params_from_appendix() -> Tuple[HNGarchParams, dict]:
    # Parameter block copied from both Appendix A and B snippets. :contentReference[oaicite:28]{index=28}
    params = HNGarchParams(
        a=1.32e-6,
        b=0.589,
        om=5.02e-6,
        gam=421.39,
        lam=0.205,
    )
    inputs = dict(
        r=0.0,
        T=90,
        h1=(0.15**2) / 252.0,
        K=100.0,
        S=100.0,
    )
    return params, inputs


def test_fourier_and_mc_are_close():
    """
    Unit test: Appendix A MC price should be close to Appendix B Fourier price
    for the same parameter set. This is the main sanity check that both engines
    and the risk-neutralization are implemented consistently.
    """
    params, inp = _default_params_from_appendix()

    fourier = hn_fourier_call_appendix_b(
        S=inp["S"], K=inp["K"], r=inp["r"], T=inp["T"], h1=inp["h1"], params=params
    )

    mc = hn_monte_carlo_call_appendix_a(
        S=inp["S"],
        K=inp["K"],
        r=inp["r"],
        T=inp["T"],
        h1=inp["h1"],
        params=params,
        nsim=80_000,
        seed=12345,
    )
    print(f"\nFourier price: {fourier:.6f}, MC price: {mc:.6f}")
    assert abs(mc - fourier) / max(1.0, abs(fourier)) < 0.02


if __name__ == "__main__":
    test_fourier_and_mc_are_close()
