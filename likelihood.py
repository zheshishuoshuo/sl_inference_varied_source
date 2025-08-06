"""Likelihood evaluation using pre-tabulated lensing grids.

This module builds upon :func:`make_tabulate.tabulate_likelihood_grids`
which pre-computes, for each observed lens, a grid of quantities required by
our likelihood evaluation.  Given these grids and the observed stellar masses
(from SPS modelling), the functions here compute the likelihood and posterior
for the population hyper-parameters.

The implementation is a streamlined version of the legacy code located in
``old_script/likelihood.py`` and avoids re-solving the lens equation at every
step of the MCMC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from scipy.stats import skewnorm, norm
import numpy as np
from scipy.stats import norm

from .cached_A import cached_A_interp
from .make_tabulate.make_tabulate import LensGrid, tabulate_likelihood_grids
from .mock_generator.mass_sampler import MODEL_PARAMS

# Parameters of the generative model (default: deVauc) used for sizes
MODEL_P = MODEL_PARAMS["deVauc"]


# -----------------------------------------------------------------------------
# Utility API
# -----------------------------------------------------------------------------

def precompute_grids(
    mock_observed_data,
    logMh_grid: Iterable[float],
    zl: float = 0.3,
    zs: float = 2.0,
    ms: float = 26.0,
    sigma_m: float = 0.1,
    m_lim: float = 26.5,
) -> list[LensGrid]:
    """Convenience wrapper around :func:`tabulate_likelihood_grids`.

    Parameters are passed directly to :func:`tabulate_likelihood_grids`.
    """

    return tabulate_likelihood_grids(
        mock_observed_data,
        logMh_grid,
        zl=zl,
        zs=zs,
        ms=ms,
        sigma_m=sigma_m,
        m_lim=m_lim,
    )


# -----------------------------------------------------------------------------
# Likelihood and posterior
# -----------------------------------------------------------------------------


def log_prior(theta: Sequence[float]) -> float:
    """Simple flat priors for the population hyper-parameters.

    ``theta`` is expected to contain ``(mu0, beta, sigmaDM, mu_alpha, sigma_alpha)``.
    """

    mu0, beta, sigmaDM, mu_alpha, sigma_alpha = theta
    if not (
        12.0 < mu0 < 13.0
        and 0.1 < sigmaDM < 0.5
        and 0.0 < sigma_alpha < 0.4
        and -0.1 < mu_alpha < 0.3
        and 1.0 < beta < 3.0
    ):
        return -np.inf
    return 0.0


def _single_lens_likelihood(
    grid: LensGrid,
    logM_sps_obs: float,
    theta: Sequence[float],
    logalpha_grid: np.ndarray,
) -> float:
    """Evaluate the integral for one lens on the supplied grids.

    The halo--mass prior is conditioned on the SPS-based stellar mass.
    """

    mu0, beta, sigmaDM, mu_alpha, sigma_alpha = theta

    # Extract arrays and mask invalid entries
    mask = (
        np.isfinite(grid.logM_star)
        & np.isfinite(grid.detJ)
        & (grid.detJ > 0)
        & np.isfinite(grid.p_magA)
        & np.isfinite(grid.p_magB)
    )
    if not np.any(mask):
        return 0.0

    logMh = grid.logMh_grid[mask]
    logM_star = grid.logM_star[mask]
    detJ = grid.detJ[mask]
    selA = grid.selA[mask]
    selB = grid.selB[mask]
    p_magA = grid.p_magA[mask]
    p_magB = grid.p_magB[mask]

    const = detJ * selA * selB * p_magA * p_magB

    # Halo–mass relation conditioned on the SPS-based stellar mass
    p_logMh = norm.pdf(
        logMh[None, :],
        loc=mu0 + beta * ((logM_star[None, :] - logalpha_grid[:, None]) - 11.4),
        scale=sigmaDM,
    )
    p_logalpha = norm.pdf(logalpha_grid, loc=mu_alpha, scale=sigma_alpha)

    # Stellar-mass likelihood (measurement scatter of 0.1 dex)
    p_Mstar = norm.pdf(
        logM_sps_obs,
        loc=logM_star[None, :] - logalpha_grid[:, None],
        scale=0.1,  # Measurement scatter of 0.1 dex
    )


    # ==== 模型参数 ====
    a = 10 ** MODEL_P["log_s_star"]
    loc = MODEL_P["mu_star"]
    scale = MODEL_P["sigma_star"]

    # ==== skew-normal prior on SPS mass ====
    p_Msps_prior = skewnorm.pdf(
        logM_star[None, :] - logalpha_grid[:, None],  # SPS mass
        a=a,
        loc=loc,
        scale=scale,
)

    # Size likelihood using the same relation as in the mock generator
    mu_Re = MODEL_P["mu_R0"] + MODEL_P["beta_R"] * (
        (logM_star[None, :] - logalpha_grid[:, None]) - 11.4
    )
    p_logRe = norm.pdf(grid.logRe, loc=mu_Re, scale=MODEL_P["sigma_R"])

    Z = p_Msps_prior * p_Mstar * p_logRe * p_logalpha[:, None] * p_logMh * const[None, :]

    # Integrate over logalpha and logMh
    integral_alpha = np.trapz(Z, logalpha_grid, axis=0)
    integral = np.trapz(integral_alpha, logMh)

    return float(max(integral, 1e-300))


def log_likelihood(
    theta: Sequence[float],
    grids: Sequence[LensGrid],
    logM_sps_obs: Sequence[float],
    logalpha_grid: np.ndarray | None = None,
) -> float:
    """Compute the joint log-likelihood for all lenses."""

    mu0, beta, sigmaDM, mu_alpha, sigma_alpha = theta
    if sigmaDM <= 0 or sigma_alpha <= 0 or sigmaDM > 2.0 or sigma_alpha > 2.0:
        return -np.inf
    
    Aeta = 1

    try:
        A_eta = cached_A_interp(mu0, sigmaDM, beta, 0.0)
        if not np.isfinite(A_eta) or A_eta <= 0:
            return -np.inf
    except Exception:
        return -np.inf

    if logalpha_grid is None:
        logalpha_grid = np.linspace(mu_alpha - 4 * sigma_alpha, mu_alpha + 4 * sigma_alpha, 35)

    logL = 0.0
    for grid, logM_obs in zip(grids, logM_sps_obs):
        L_i = _single_lens_likelihood(grid, float(logM_obs), theta, logalpha_grid)
        if not np.isfinite(L_i) or L_i <= 0:
            return -np.inf
        logL += np.log(L_i / A_eta)

    return float(logL)


def log_posterior(
    theta: Sequence[float],
    grids: Sequence[LensGrid],
    logM_sps_obs: Sequence[float],
    logalpha_grid: np.ndarray | None = None,
) -> float:
    """Posterior = prior + likelihood."""

    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, grids, logM_sps_obs, logalpha_grid)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


__all__ = [
    "precompute_grids",
    "log_prior",
    "log_likelihood",
    "log_posterior",
]
