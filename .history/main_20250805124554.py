from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import corner

from .mock_generator.mock_generator import run_mock_simulation
from .likelihood import precompute_grids
from .run_mcmc import run_mcmc

matplotlib.use("TkAgg")  # 本地运行 GUI 绘图

def main() -> None:
    # === 模拟 mock 数据 ===
    _, mock_obs = run_mock_simulation(1000)
    logM_sps_obs = mock_obs["logM_star_sps_observed"].values

    # === 构建 halo mass 网格 ===
    logMh_grid = np.linspace(11.5, 14.0, 50)
    grids = precompute_grids(mock_obs, logMh_grid)

    # === 运行 MCMC ===
    nsteps = 5000
    sampler = run_mcmc(grids, logM_sps_obs, nsteps=nsteps, nwalkers=20)
    chain = sampler.get_chain(discard=nsteps-2000, flat=True)
    print("MCMC sampling completed.")

    # === MCMC 样本 reshape 并转为 DataFrame ===
    samples = chain.reshape(-1, chain.shape[-1])
    param_names = [r"$\mu_{\rm DM0}$", r"$\beta_{\rm DM}$", r"$\sigma_{\rm DM}$", r"$\mu_\alpha$", r"$\sigma_\alpha$"]
    df_samples = pd.DataFrame(samples, columns=param_names)

    # === 真值 ===
    true_values = [12.91, 2.04, 0.37, 0.1, 0.05]

    # === 创建左右子图布局 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # === 左侧子图：mock 数据分布（直方图）===
    ax = axes[0]
    ax.hist(logM_sps_obs, bins=30, color='skyblue', edgecolor='k')
    ax.set_xlabel(r"$\log M_*^{\mathrm{SPS}}$")
    ax.set_ylabel("Count")
    ax.set_title("Mock Observed Stellar Mass Distribution")

    # === 右侧子图：后验分布（corner 图）===
    # 使用 subplot2grid 嵌入右边图像
    from matplotlib.gridspec import GridSpec
    fig.clf()
    gs = GridSpec(1, 2, width_ratios=[1, 2.2])
    ax_mock = fig.add_subplot(gs[0])
    ax_corner = plt.subplot(gs[1])

    # 左图重新画 mock 数据
    ax_mock.hist(logM_sps_obs, bins=30, color='skyblue', edgecolor='k')
    ax_mock.set_xlabel(r"$\log M_*^{\mathrm{SPS}}$")
    ax_mock.set_ylabel("Count")
    ax_mock.set_title("Mock Observed")

    # 右图画 corner 图
    fig_corner = corner.corner(
        samples,
        labels=param_names,
        truths=true_values,
        truth_color="red",
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 10},
        fig=fig,
        hist_kwargs={"color": "steelblue", "edgecolor": "k"},
        plot_datapoints=True,
        max_n_ticks=4,
        label_kwargs={"fontsize": 11},
        color="steelblue",
        subplot_size=2.5,
        bins=30
    )

    plt.tight_layout()
    plt.show()

    print("Finished MCMC. Chain shape:", chain.shape)


if __name__ == "__main__":
    main()
