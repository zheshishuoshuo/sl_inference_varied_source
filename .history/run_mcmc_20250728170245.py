import os
import numpy as np
import emcee
from emcee.backends import HDFBackend
import multiprocessing
from tqdm import tqdm


def run_mcmc(data_df,
             interp_list,
             log_posterior_func,
             backend_file="chains_eta.h5",
             nwalkers=50,
             nsteps=3000,
             ndim=4,
             initial_guess=None,
             resume=True,
             processes=None):
    """
    运行 MCMC 主程序，使用 emcee 并支持 checkpoint。

    参数:
    - data_df: 模拟观测数据 DataFrame（会作为 log_posterior 参数传入）
    - interp_list: 插值器列表（提前构建）
    - log_posterior_func: 目标后验函数，应接收 (theta, data_df, interp_list) 参数
    - backend_file: HDF5 文件路径
    - nwalkers: MCMC walker 数量
    - nsteps: 总步数
    - ndim: 参数维数（默认4维）
    - initial_guess: 长度为 ndim 的初始值（如 [12.5, 0.3, 0.05, 0.05]）
    - resume: 是否尝试从已有 HDF5 中恢复
    - processes: 使用的 CPU 核心数（None 表示自动）

    返回:
    - sampler: emcee 的采样器对象
    """
    if processes is None:
        processes = max(1, int(multiprocessing.cpu_count() // 1.5))

    if resume and os.path.exists(backend_file):
        print(f"[INFO] 继续采样：读取已有文件 {backend_file}")
        backend = HDFBackend(backend_file, read_only=False)
    else:
        print(f"[INFO] 新建采样：创建新文件 {backend_file}")
        backend = HDFBackend(backend_file)
        backend.reset(nwalkers, ndim)

    if backend.iteration == 0:
        assert initial_guess is not None, "初始值必须提供"
        print("[INFO] 从头开始采样")
        p0 = initial_guess + 1e-3 * np.random.randn(nwalkers, ndim)
    else:
        print(f"[INFO] 从第 {backend.iteration} 步继续采样")
        p0 = None  # emcee 会从后端自动恢复

    # 定义目标函数：固定参数传递
    def logpost_wrap(theta):
        return log_posterior_func(theta, data_df, interp_list)

    with multiprocessing.get_context("spawn").Pool(processes=processes) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, logpost_wrap, pool=pool, backend=backend
        )
        sampler.run_mcmc(p0, nsteps, progress=True)

    print("[INFO] 采样完成")
    return sampler
