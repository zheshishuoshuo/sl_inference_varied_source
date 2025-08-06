from run_mcmc import run_mcmc
from likelihood import log_posterior
from interpolators import build_interp_list_for_lenses
from data_loader import load_mock_data
import numpy as np
# 加载数据
data_df = load_mock_data()
logMh_grid = np.linspace(11.5, 14.0, 100)
interp_list = build_interp_list_for_lenses(data_df, logMh_grid)

# 运行 MCMC
sampler = run_mcmc(
    data_df=data_df,
    interp_list=interp_list,
    log_posterior_func=log_posterior,
    backend_file="mcmc_chain.h5",
    nwalkers=50,
    nsteps=3000,
    ndim=4,
    initial_guess=np.array([12.5, 0.3, 0.05, 0.05]),
)
