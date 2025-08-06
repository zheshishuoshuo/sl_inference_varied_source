from run_mcmc import run_mcmc
from likelihood import log_posterior
from interpolator import build_interp_list_for_lenses
import numpy as np
from mock_generator import run_mock_simulation
# import resource


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


from run_mcmc import run_mcmc
from likelihood import log_posterior, set_context
from interpolator import build_interp_list_for_lenses
import numpy as np
from mock_generator import run_mock_simulation

def main():
    mock_lens_data, mock_observed_data = run_mock_simulation(n_samples=3)
    logMh_grid = np.linspace(12.5, 14.0, 100)
    print(f"[INFO] 生成模拟数据: {len(mock_observed_data)} lenses")

    print(f"[INFO] 构建插值器列表")
    interp_list = build_interp_list_for_lenses(mock_observed_data, logMh_grid)
    logMstar_interp_list = [interp["logMstar_interp"] for interp in interp_list]
    detJ_interp_list = [interp["detJ_interp"] for interp in interp_list]

    # ✅ 设置全局上下文，用于 log_posterior 多进程调用
    set_context(
        data_df=mock_observed_data,
        logMstar_interp_list=logMstar_interp_list,
        detJ_interp_list=detJ_interp_list,
        use_interp=True
    )

    # ✅ 注意：run_mcmc 不再需要传入 data_df 和 interp_list
    sampler = run_mcmc(
        data_df=None,
        interp_list=None,
        log_posterior_func=log_posterior,
        backend_file="mcmc_chain.h5",
        nwalkers=8,
        nsteps=2000,
        ndim=4,
        initial_guess=np.array([12.5, 0.3, 0.05, 0.05]),
    )

if __name__ == "__main__":
    main()  # ✅ multiprocessing 必须放在 main 保护下
