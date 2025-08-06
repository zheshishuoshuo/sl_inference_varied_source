from __future__ import annotations

import numpy as np
import seaborn as sns
import pandas as pd
import multiprocessing as mp
from .mock_generator.mock_generator import run_mock_simulation
from .likelihood import precompute_grids
from .run_mcmc import run_mcmc
from .plotting import plot_chain
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # 或者 Qt5Agg, MacOSX


def main() -> None:
    # Generate mock data for  samples
    mock_lens_data, mock_observed_data = run_mock_simulation(1000)
    logM_sps_obs = mock_observed_data["logM_star_sps_observed"].values

    mock_lens_data.to_csv("mock_lens_data.csv", index=False)

    # Precompute grids on halo mass
    logMh_grid = np.linspace(11.5, 14.0, 100)
    grids = precompute_grids(mock_observed_data, logMh_grid)
    nsteps = 6000
    # Run MCMC sampling for 10000 steps
    sampler = run_mcmc(grids, logM_sps_obs, nsteps=nsteps, nwalkers=20, backend_file="chains_eta_new_table_no_eta_variedms10006.h5", parallel=True, nproc=mp.cpu_count()-3)
    chain = sampler.get_chain(discard=nsteps-2000, flat=True)
    print("MCMC sampling completed.")

    samples = chain.reshape(-1, chain.shape[-1])

    # 转为 DataFrame 并加上列名
    # param_names = ["param1", "param2", "param3", "param4", "param5"]  # 你可以改成实际参数名
    param_names = [ r"$\mu_{DM0}$", r"$\beta_{DM}$", r"$\sigma_{DM}$", r"$\mu_\alpha$", r"$\sigma_\alpha$" ]  # Example parameter names

    df_samples = pd.DataFrame(samples, columns=param_names)

    # # 画 pairplot
    # sns.pairplot(
    #     df_samples,
    #     diag_kind="kde",
    #     markers=".",
    #     plot_kws={"alpha": 0.5, "s": 10},
    #     corner=True
    # )

    # 真值
    true_values = [12.91, 2.04, 0.37, 0.1, 0.05]

    # 绘制 pairplot
    g = sns.pairplot(
        df_samples,
        diag_kind="kde",
        markers=".",
        plot_kws={"alpha": 0.1, "s": 10},
        corner=True
    )

    # 遍历对角线（直方图/KDE）添加真值竖线
    for i, ax in enumerate(np.diag(g.axes)):
        ax.axvline(true_values[i], color="red", linestyle="--", linewidth=1)

    # 遍历下三角（散点图）添加真值参考线
    for i in range(len(true_values)):
        for j in range(len(true_values)):
            if i > j:  # 下三角
                ax = g.axes[i, j]
                ax.axvline(true_values[j], color="red", linestyle="--", linewidth=1)
                ax.axhline(true_values[i], color="red", linestyle="--", linewidth=1)

    plt.show()


    print("Finished MCMC. Chain shape:", chain.shape)
   


if __name__ == "__main__":
    main()
