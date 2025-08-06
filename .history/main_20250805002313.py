from __future__ import annotations

import numpy as np
import seaborn as sns
import pandas as pd
from .mock_generator.mock_generator import run_mock_simulation
from .likelihood import precompute_grids
from .run_mcmc import run_mcmc
from .plotting import plot_chain

def main() -> None:
    # Generate mock data for 10 samples
    _, mock_obs = run_mock_simulation(10)
    logM_sps_obs = mock_obs["logM_star_sps_observed"].values

    # Precompute grids on halo mass
    logMh_grid = np.linspace(11.5, 14.0, 50)
    grids = precompute_grids(mock_obs, logMh_grid)

    # Run MCMC sampling for 500 steps
    sampler = run_mcmc(grids, logM_sps_obs, nsteps=500, nwalkers=20)
    chain = sampler.get_chain()
    print("MCMC sampling completed.")
    # plot_chain(chain)
    g = sns.jointplot(
        x=chain[:, 0, 0], y=chain[:, 0, 1], kind="kde", fill=True, cmap="viridis"
    )
    g.ax_joint.set_xlabel("logM_halo")
    g.ax_joint.set_ylabel("logM_star")
    plt.show()

    print("Finished MCMC. Chain shape:", chain.shape)
   


if __name__ == "__main__":
    main()
