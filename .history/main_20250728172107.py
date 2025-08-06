from run_mcmc import run_mcmc
from likelihood import log_posterior
from interpolator import build_interp_list_for_lenses
import numpy as np
from mock_generator import run_mock_simulation





# 加载数据

logMh_grid = np.linspace(11.5, 14.0, 100)
interp_list = build_interp_list_for_lenses(mock_observed_data, logMh_grid)

# 运行 MCMC
sampler = run_mcmc(
    data_df=mock_observed_data,
    interp_list=interp_list,
    log_posterior_func=log_posterior,
    backend_file="mcmc_chain.h5",
    nwalkers=50,
    nsteps=3000,
    ndim=4,
    initial_guess=np.array([12.5, 0.3, 0.05, 0.05]),
)


# main.py
from run_mcmc import run_mcmc
from likelihood import log_posterior
from interpolators import build_interp_list_for_lenses
from data_loader import load_mock_data
import numpy as np

def main():
    mock_lens_data, mock_observed_data = run_mock_simulation(n_samples=100)
    logMh_grid = np.linspace(11.5, 14.0, 100)
    interp_list = build_interp_list_for_lenses(mock_observed_data)
    sampler = run_mcmc(
        data_df=data,
        interp_list=interp_list,
        log_posterior_func=log_posterior,
        backend_file="mcmc_chain.h5",
        nwalkers=50,
        nsteps=3000,
        ndim=4,
        initial_guess=np.array([12.5, 0.3, 0.05, 0.05]),
    )

if __name__ == "__main__":
    main()  # ✅ 所有并行逻辑必须包在这里面！
