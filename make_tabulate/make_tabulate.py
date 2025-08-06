"""Utilities for pre-computing grids used in the likelihood.

This module provides helper functions to tabulate, for each observed mock
lens, the quantities required by the likelihood evaluation on a grid of
dark-matter halo masses (``logMhalo``).

The main entry point is :func:`tabulate_likelihood_grids` which accepts a
``pandas.DataFrame`` with the observed lens properties and returns, for each
lens, arrays of the model quantities evaluated on the supplied halo-mass
grid.  These pre-computed tables can be used to accelerate the likelihood
computation by avoiding repeated calls to the expensive lens solver.

The implementation mirrors the calculations performed inside the legacy
``likelihood`` module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from ..mock_generator.lens_solver import (
    compute_detJ,
    solve_lens_parameters_from_obs,
)
from ..mock_generator.lens_model import LensModel


@dataclass
class LensGrid:
    """Container for pre-computed quantities of a single lens.

    Attributes
    ----------
    logMh_grid:
        The grid of halo masses on which all quantities are evaluated.
    logM_star:
        Stellar mass (log10) inferred from the lens equation on the grid.
    detJ:
        Determinant of the Jacobian of the transformation.
    muA, muB:
        Magnifications of the two images.
    logRe:
        Observed effective radius used when generating the grid.
    m1_obs, m2_obs:
        Observed magnitudes of the two lensed images.
    sigma_m, m_lim:
        Measurement scatter and limiting magnitude used in the selection
        function.
    """

    logMh_grid: np.ndarray
    logM_star: np.ndarray
    detJ: np.ndarray
    muA: np.ndarray
    muB: np.ndarray
    logRe: float
    m1_obs: float
    m2_obs: float
    sigma_m: float
    m_lim: float


def tabulate_likelihood_grids(
    mock_observed_data: pd.DataFrame,
    logMh_grid: Iterable[float],
    zl: float = 0.3,
    zs: float = 2.0,
    sigma_m: float = 0.1,
    m_lim: float = 26.5,
) -> List[LensGrid]:
    """Compute grids of lensing quantities required by the likelihood.

    Parameters
    ----------
    mock_observed_data:
        DataFrame containing the observed properties of the mock lenses.  It
        must contain the columns ``xA``, ``xB``, ``logRe``,
        ``magnitude_observedA`` and ``magnitude_observedB``.
    logMh_grid:
        Sequence of halo masses (``log10`` scale) on which the quantities are
        evaluated.
    zl, zs:
        Lens and source redshifts.
    sigma_m, m_lim:
        Parameters describing the measurement scatter and limiting magnitude
        used in the selection function.

    Returns
    -------
    list of :class:`LensGrid`
        One entry for each lens in ``mock_observed_data`` containing the
        pre-computed arrays on the requested halo-mass grid.
    """

    logMh_grid = np.asarray(list(logMh_grid))
    results: List[LensGrid] = []

    for _, row in mock_observed_data.iterrows():
        xA = float(row["xA"])
        xB = float(row["xB"])
        logRe = float(row["logRe"])
        m1_obs = float(row["magnitude_observedA"])
        m2_obs = float(row["magnitude_observedB"])

        logMstar_list = []
        detJ_list = []
        muA_list = []
        muB_list = []

        for logMh in logMh_grid:
            try:
                logM_star, _ = solve_lens_parameters_from_obs(
                    xA, xB, logRe, logMh, zl, zs
                )
                detJ = compute_detJ(xA, xB, logRe, logMh, zl, zs)

                model = LensModel(
                    logM_star=logM_star,
                    logM_halo=logMh,
                    logRe=logRe,
                    zl=zl,
                    zs=zs,
                )
                muA = model.mu_from_rt(xA)
                muB = model.mu_from_rt(xB)
            except Exception:
                logM_star = np.nan
                detJ = 0.0
                muA = np.nan
                muB = np.nan

            logMstar_list.append(logM_star)
            detJ_list.append(detJ)
            muA_list.append(muA)
            muB_list.append(muB)

        results.append(
            LensGrid(
                logMh_grid=logMh_grid,
                logM_star=np.array(logMstar_list),
                detJ=np.array(detJ_list),
                muA=np.array(muA_list),
                muB=np.array(muB_list),
                logRe=logRe,
                m1_obs=m1_obs,
                m2_obs=m2_obs,
                sigma_m=sigma_m,
                m_lim=m_lim,
            )
        )

    return results


__all__ = ["LensGrid", "tabulate_likelihood_grids"]

