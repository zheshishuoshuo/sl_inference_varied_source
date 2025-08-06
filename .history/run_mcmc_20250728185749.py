import os
import numpy as np
import emcee
from emcee.backends import HDFBackend
import multiprocessing
from functools import partial
from likelihood import log_posterior, initializer_for_pool  # ⬅ 你已经写了它！

def run_mcmc(
    data_df,
    logMstar_interp_list,
    detJ_interp_list,
    use_interp,
    log_posterior_func,
    backend_file="chains_eta.h5",
    nwalkers=50,
    nsteps=3000,
    ndim=4,
    initial_guess=None,
    resume=True,
    processes=None
):
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

    # ✅ 只传入 eta 参数，数据由 initializer 设置为每个子进程的全局变量
    logpost = partial(log_posterior_func)

    # with multiprocessing.get_context("spawn").Pool(
    #     processes=processes,
    #     initializer=initializer_for_pool,
    #     initargs=(data_df, logMstar_interp_list, detJ_interp_list, use_interp)
    # ) as pool:
    #     sampler = emcee.EnsembleSampler(
    #         nwalkers, ndim, logpost, pool=pool, backend=backend
    #     )
    #     sampler.run_mcmc(p0, nsteps, progress=True)

    with multiprocessing.get_context("spawn").Pool(
        processes=processes,
        initializer=initializer_for_pool,
        initargs=(data_df, logMstar_interp_list, detJ_interp_list, use_interp)
    ) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, logpost, pool=pool, backend=backend
        )
        sampler.run_mcmc(p0, nsteps, progress=True)



    print("[INFO] 采样完成")
    return sampler
