"""Not a Simple test module."""
from .run_mcmc import run_mcmc
from .likelihood import log_posterior
from .interpolator import build_interp_list_for_lenses
from .mock_generator import run_mock_simulation
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    mock_lens_data, mock_observed_data = run_mock_simulation(n_samples=10)
    logMh_grid = np.linspace(11.5, 14.0, 100)

    logMstar_list, detJ_list = build_interp_list_for_lenses(
        mock_observed_data, logMh_grid, zl=0.3, zs=2.0
    )
    test_filename = "chains_eta.h5"
    if os.path.exists(os.path.join(os.path.dirname(__file__),'chains', test_filename)):
        print(f"[INFO] 继续采样：读取已有文件 {test_filename}")

    sampler = run_mcmc(
        data_df=mock_observed_data,
        logMstar_interp_list=logMstar_list,
        detJ_interp_list=detJ_list,
        use_interp=True,
        log_posterior_func=log_posterior,
        backend_file=test_filename,
        nwalkers=12,
        nsteps=100,
        ndim=6,
        initial_guess=np.array([12.5, 2.0, 0.0, 0.3, 0.05, 0.05]),
        processes=4
        )
    samples = sampler.get_chain(flat=True, discard=0)
    df = pd.DataFrame(samples, columns=[
        "mu0", "beta", "xi", "sigma", "mu_alpha", "sigma_alpha"
    ])
    print(f"[INFO] 采样完成，样本数量: {len(samples)}")
    sns.pairplot(df, diag_kind="kde", markers=".")
    plt.show()


if __name__ == "__main__":
    run()
