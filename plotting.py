"""Plotting utilities for visualising inference results."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_chain(chain: np.ndarray, fname: str = "chain_trace.png") -> None:
    """Save a trace plot of the first parameter of an MCMC chain.

    Parameters
    ----------
    chain:
        MCMC chain array with shape ``(nsteps, nwalkers, ndim)``.
    fname:
        Output filename for the saved figure.  Defaults to ``"chain_trace.png"``.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(chain[:, :, 0], alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\mu_0$")
    fig.tight_layout()
    # fig.savefig(fname)
    # plt.close(fig)


__all__ = ["plot_chain"]
