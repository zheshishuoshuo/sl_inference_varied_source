import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import os
import sys
import types

if __package__ is None or __package__ == "":
    # Allow running as a standalone script without installing the package
    package_dir = os.path.dirname(os.path.abspath(__file__))
    pkg = types.ModuleType("compute_norm_acc")
    pkg.__path__ = [package_dir]
    sys.modules["compute_norm_acc"] = pkg
    sys.path.insert(0, os.path.dirname(package_dir))
    from compute_norm_acc.mock_generator.mass_sampler import generate_samples, MODEL_PARAMS
    from compute_norm_acc.mock_generator.lens_solver import solve_single_lens
    from compute_norm_acc.mock_generator.lens_model import LensModel
    from compute_norm_acc.utils import selection_function
    from compute_norm_acc.config import OBS_SCATTER_MAG
else:
    from .mock_generator.mass_sampler import generate_samples, MODEL_PARAMS
    from .mock_generator.lens_solver import solve_single_lens
    from .mock_generator.lens_model import LensModel
    from .utils import selection_function
    from .config import OBS_SCATTER_MAG


def sample_lens_population(n_samples, zl=0.3, zs=2.0):
    """Generate lens population samples consistent with existing simulation."""
    data = generate_samples(n_samples)
    logalpha_sps = np.random.normal(0.1, 0.05, n_samples)
    logM_star = data["logM_star_sps"] + logalpha_sps
    logRe = data["logRe"]
    beta = np.random.rand(n_samples) ** 0.5
    # sample halo mass uniformly to allow importance reweighting later
    logMh_min, logMh_max = 11.0, 15.0
    logMh = np.random.uniform(logMh_min, logMh_max, n_samples)
    return {
        "logM_star": logM_star,
        "logRe": logRe,
        "logMh": logMh,
        "beta": beta,
        "logMh_min": logMh_min,
        "logMh_max": logMh_max,
        "zl": zl,
        "zs": zs,
    }


def _solve_magnification(args):
    """Helper to solve a single lens and return magnifications."""
    logM_star, logRe, logMh, beta, zl, zs = args
    model = LensModel(
        logM_star=logM_star,
        logM_halo=logMh,
        logRe=logRe,
        zl=zl,
        zs=zs,
    )
    xA, xB = solve_single_lens(model, beta)
    return model.mu_from_rt(xA), model.mu_from_rt(xB)


def compute_magnifications(logM_star, logRe, logMh, beta, zl, zs, n_jobs=None):
    """Solve lens equation for each sample and return two magnifications."""
    n = len(logM_star)
    args = zip(logM_star, logRe, logMh, beta, repeat(zl), repeat(zs))
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        results = list(
            tqdm(
                pool.map(_solve_magnification, args),
                total=n,
                desc="solving lenses",
            )
        )
    mu1, mu2 = map(np.array, zip(*results))
    return mu1, mu2


def ms_distribution(ms_grid, alpha_s=-1.3, ms_star=24.5):
    """Normalized PDF of source magnitude on a grid."""
    L = 10 ** (-0.4 * (ms_grid - ms_star))
    pdf = L ** (alpha_s + 1) * np.exp(-L)
    pdf /= np.trapz(pdf, ms_grid)
    return pdf


def build_eta_grid():
    mu_DM_grid = np.linspace(12.5, 13.5, 30)
    sigma_DM_grid = np.linspace(0.2, 0.5, 30)
    beta_DM_grid = np.linspace(1.5, 2.5, 30)
    return mu_DM_grid, sigma_DM_grid, beta_DM_grid


def compute_A_eta(n_samples=10000, ms_points=15, m_lim=26.5):
    samples = sample_lens_population(n_samples)
    mu1, mu2 = compute_magnifications(
        samples["logM_star"],
        samples["logRe"],
        samples["logMh"],
        samples["beta"],
        samples["zl"],
        samples["zs"],
    )

    ms_grid = np.linspace(20.0, 30.0, ms_points)
    pdf_ms = ms_distribution(ms_grid)
    sel1 = selection_function(mu1[:, None], m_lim, ms_grid[None, :], OBS_SCATTER_MAG)
    sel2 = selection_function(mu2[:, None], m_lim, ms_grid[None, :], OBS_SCATTER_MAG)
    p_det = sel1 * sel2
    w_ms = np.trapz(p_det * pdf_ms[None, :], ms_grid, axis=1)
    # correct for beta sampling (beta ~ 2*beta_uniform)
    w_static = w_ms / (2.0 * samples["beta"])

    lens_df = pd.DataFrame(
        {
            "logM_star": samples["logM_star"],
            "logRe": samples["logRe"],
            "logMh": samples["logMh"],
            "beta": samples["beta"],
            "mu1": mu1,
            "mu2": mu2,
            "w_ms": w_ms,
            "w_static": w_static,
            "zl": np.full(n_samples, samples["zl"]),
            "zs": np.full(n_samples, samples["zs"]),
        }
    )
    lens_df.to_csv("lens_samples.csv", index=False)

    mu_DM_grid, sigma_DM_grid, beta_DM_grid = build_eta_grid()

    logM_star = samples["logM_star"][:, None, None, None]
    logMh = samples["logMh"][:, None, None, None]
    w = w_static[:, None, None, None]

    mu_grid = mu_DM_grid[None, :, None, None]
    sigma_grid = sigma_DM_grid[None, None, :, None]
    beta_grid = beta_DM_grid[None, None, None, :]

    mean = mu_grid + beta_grid * (logM_star - 11.4)
    p_Mh = norm.pdf(logMh, loc=mean, scale=sigma_grid)

    Mh_range = samples["logMh_max"] - samples["logMh_min"]
    A = Mh_range * np.sum(w * p_Mh, axis=0) / n_samples

    mu_flat, sigma_flat, beta_flat = np.meshgrid(
        mu_DM_grid, sigma_DM_grid, beta_DM_grid, indexing="ij"
    )
    df = pd.DataFrame(
        {
            "mu_DM": mu_flat.ravel(),
            "sigma_DM": sigma_flat.ravel(),
            "beta_DM": beta_flat.ravel(),
            "A": A.ravel(),
        }
    )
    df.to_csv("A_eta_table.csv", index=False)
    return df


if __name__ == "__main__":
    compute_A_eta()
