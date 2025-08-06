from __future__ import annotations

import numpy as np
import seaborn as sns
import pandas as pd
from .mock_generator.mock_generator import run_mock_simulation
from .likelihood import precompute_grids
from .run_mcmc import run_mcmc
from .plotting import plot_chain
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # 或者 Qt5Agg, MacOSX


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

    samples = chain.reshape(-1, chain.shape[-1])

    # 转为 DataFrame 并加上列名
    param_names = ["param1", "param2", "param3", "param4", "param5"]  # 你可以改成实际参数名
    df_samples = pd.DataFrame(samples, columns=param_names)

    # 画 pairplot
    sns.pairplot(
        df_samples,
        diag_kind="kde",
        markers=".",
        plot_kws={"alpha": 0.5, "s": 10},
        corner=True
    )

    plt.show()

    print("Finished MCMC. Chain shape:", chain.shape)
   


if __name__ == "__main__":
    main()
