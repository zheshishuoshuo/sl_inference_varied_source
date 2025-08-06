from run_mcmc import run_mcmc
from likelihood import log_posterior
from interpolator import build_interp_list_for_lenses
import numpy as np
from mock_generator import run_mock_simulation
# import resource
from likelihood import log_posterior, set_context

# limit = 6 * 1024**3  # 6 GB
# resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

# def main():
#     mock_lens_data, mock_observed_data = run_mock_simulation(n_samples=3)
#     logMh_grid = np.linspace(12.5, 14.0, 100)
#     print(f"[INFO] 生成模拟数据: {len(mock_observed_data)} lenses")
#     print(f"Begin interpolating tabular data...")
#     # interp_list = build_interp_list_for_lenses(mock_observed_data, logMh_grid)
#     # print(f"[INFO] 插值器列表长度: {len(interp_list)}")
#     sampler = run_mcmc(
#         data_df=mock_observed_data,
#         # interp_list=interp_list,
#         log_posterior_func=log_posterior,
#         backend_file="mcmc_chain.h5",
#         nwalkers=8,
#         nsteps=2000,
#         ndim=4,
#         initial_guess=np.array([12.5, 0.3, 0.05, 0.05]),
#     )

# if __name__ == "__main__":
#     main()  # ✅ 所有并行逻辑必须包在这里面！



def main():
    mock_lens_data, mock_observed_data = run_mock_simulation(n_samples=3)
    logMh_grid = np.linspace(12.5, 14.0, 100)
    print(f"[INFO] 生成模拟数据: {len(mock_observed_data)} lenses")

    # 构造插值器列表
    logMstar_list, detJ_list = build_interp_list_for_lenses(
        mock_observed_data, logMh_grid, zl=0.3, zs=2.0
    )

    # ✅ 设置 log_posterior 的全局上下文
    set_context(
        data_df=mock_observed_data,
        logMstar_interp_list=logMstar_list,
        detJ_interp_list=detJ_list,
        use_interp=True
    )

    # 运行 MCMC（data_df 和 interp_list 参数可以设为 None）
    sampler = run_mcmc(
        data_df=None,
        log_posterior_func=log_posterior,
        backend_file="mcmc_chain.h5",
        nwalkers=8,
        nsteps=2000,
        ndim=4,
        initial_guess=np.array([12.5, 0.3, 0.05, 0.05]),
    )

if __name__ == "__main__":
    main()

