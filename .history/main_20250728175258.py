from run_mcmc import run_mcmc
from likelihood import log_posterior
from interpolator import build_interp_list_for_lenses
import numpy as np
from mock_generator import run_mock_simulation
# import resource

# limit = 6 * 1024**3  # 6 GB
# resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

def main():
    mock_lens_data, mock_observed_data = run_mock_simulation(n_samples=3)
    logMh_grid = np.linspace(12.5, 14.0, 100)
    print(f"[INFO] 生成模拟数据: {len(mock_observed_data)} lenses")
    print(f"Begin interpolating tabular data...")
    # interp_list = build_interp_list_for_lenses(mock_observed_data, logMh_grid)
    # print(f"[INFO] 插值器列表长度: {len(interp_list)}")
    sampler = run_mcmc(
        data_df=mock_observed_data,
        # interp_list=interp_list,
        log_posterior_func=log_posterior,
        backend_file="mcmc_chain1.h5",
        nwalkers=12,
        nsteps=300,
        ndim=4,
        initial_guess=np.array([12.5, 0.3, 0.05, 0.05]),
    )

if __name__ == "__main__":
    main()  # ✅ 所有并行逻辑必须包在这里面！
