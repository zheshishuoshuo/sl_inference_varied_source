"""Not a Simple test module."""
from .run_mcmc import run_mcmc
from .likelihood import log_posterior
from .interpolator import build_interp_list_for_lenses
from .mock_generator import run_mock_simulation
import numpy as np
import seaborn as sns

def run():
    mock_lens_data, mock_observed_data = run_mock_simulation(n_samples=1)
    logMh_grid = np.linspace(11.5, 14.0, 100)

    logMstar_list, detJ_list = build_interp_list_for_lenses(
        mock_observed_data, logMh_grid, zl=0.3, zs=2.0
    )

    if os.path.exists(os.path.join(os.path.dirname(__file__),'chains', "chains_eta.h5")):
        os.remove(os.path.join(os.getcwd(), "chains_eta.h5"))
    sampler = run_mcmc(
        data_df=mock_observed_data,
        logMstar_interp_list=logMstar_list,
        detJ_interp_list=detJ_list,
        use_interp=True,
        log_posterior_func=log_posterior,
        backend_file="mcmc_chain100.h5",
        nwalkers=12,
        nsteps=100,
        ndim=6,
        initial_guess=np.array([12.5, 2.0, 0.0, 0.3, 0.05, 0.05]),
        processes=4
        )
    samples = sampler.get_chain(flat=True, discard=100)
    print(f"[INFO] 采样完成，样本数量: {len(samples)}")
    sns.pairplot(samples, diag_kind="kde", markers=".")
    sns.plt.show()


if __name__ == "__main__":
    run()
