"""sl_inference package

Tools for modelling strong gravitational lenses and performing inference on their physical properties. The
module provides convenient imports for the most commonly used classes and functions."""

from .lens_model import LensModel
from .lens_solver import solve_single_lens, solve_lens_parameters_from_obs, compute_detJ
from .lens_properties import lens_properties, observed_data
from .mass_sampler import (
    mstar_gene,
    logRe_given_logM,
    logMh_given_logM_logRe,
    generate_samples,
)
from .run_mcmc import run_mcmc
from .likelihood import (
    set_context,
    initializer_for_pool,
    log_prior,
    likelihood_single_fast_optimized,
    log_likelihood,
    log_posterior,
)
from .utils import selection_function, mag_likelihood

__all__ = [
    "LensModel",
    "solve_single_lens",
    "solve_lens_parameters_from_obs",
    "compute_detJ",
    "lens_properties",
    "observed_data",
    "mstar_gene",
    "logRe_given_logM",
    "logMh_given_logM_logRe",
    "generate_samples",
    "theoretical_logRe_pdf",
    "theoretical_logMh_pdf",
    "run_mcmc",
    "set_context",
    "initializer_for_pool",
    "log_prior",
    "likelihood_single_fast_optimized",
    "log_likelihood",
    "log_posterior",
    "selection_function",
    "mag_likelihood",
]

__version__ = "0.1"
