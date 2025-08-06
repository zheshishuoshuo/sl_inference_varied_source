import os
import numpy as np
import emcee
from emcee.backends import HDFBackend
import multiprocessing
from tqdm import tqdm
from functools import partial  # ✅ 新增
from likelihood import log_posterior, set_context, initializer_for_pool

def run_mcmc(data_df,
            #  interp_list,
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
        p0 = None

    # ✅ 使用 partial 绑定 log_posterior_func 的剩余参数
    logpost = partial(log_posterior_func)

    # with multiprocessing.get_context("spawn").Pool(processes=processes) as pool:
    #     sampler = emcee.EnsembleSampler(
    #         nwalkers, ndim, logpost, pool=pool, backend=backend, initializer=initializer_for_pool,
    #     )
    #     sampler.run_mcmc(p0, nsteps, progress=True)

    print("[INFO] 采样完成")
    return sampler
