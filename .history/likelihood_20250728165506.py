from likelihood_utils import selection_function, mag_likelihood
from lens_model import LensModel
from lens_solver import solve_lens_parameters_from_obs, compute_detJ
from cached_A import cached_A_interp
from scipy.stats import norm
import numpy as np

def log_prior(eta): ...
def log_likelihood(...): ...
def log_posterior(...): ...
def likelihood_single_fast_optimized(...): ...





def log_likelihood(
    eta, data_df, A_interp, logMstar_interp_list=None, detJ_interp_list=None,
    use_interp=False, **kwargs
):
    mu_DM, sigma_DM, mu_alpha, sigma_alpha = eta

    # 参数范围检查
    if sigma_DM <= 0 or sigma_alpha <= 0 or sigma_DM > 2.0 or sigma_alpha > 2.0:
        print(f"[DEBUG] Sigma 超出范围: sigma_DM = {sigma_DM}, sigma_alpha = {sigma_alpha}")
        return -np.inf

    try:
        A_eta = cached_A_interp(safe_round(mu_DM), safe_round(sigma_DM))
        if not np.isfinite(A_eta) or A_eta <= 0:
            print(f"[WARN] 非法 A_eta: {A_eta} for eta = {eta}")
            return -np.inf
    except Exception as e:
        print(f"[ERROR] A_interp 失败 at eta = {eta}, error = {repr(e)}")
        return -np.inf

    logL = 0.0
    # for idx, row in data_df.iterrows():
    for i, (_, row) in enumerate(data_df.iterrows()):

        try:
            di = tuple(row[col] for col in [
                'xA', 'xB', 'logMsps_observed', 'logRe',
                'magnitude_observedA', 'magnitude_observedB'
            ])
        except Exception as e:
            print(f"[ERROR] 读取第 {idx} 行数据失败: {repr(e)}")
            return -np.inf

        try:
            if use_interp:
                try:
                    # logMstar_interp = logMstar_interp_list[idx]
                    # detJ_interp = detJ_interp_list[idx]
                    logMstar_interp = logMstar_interp_list[i]
                    detJ_interp = detJ_interp_list[i]

                except IndexError as e:
                    print(f"[ERROR] 插值器列表越界: lens {idx}, total lenses = {len(logMstar_interp_list)}")
                    return -np.inf
            else:
                logMstar_interp = None
                detJ_interp = None

            L_i = likelihood_single_fast_optimized(
                di, eta,
                logMstar_interp=logMstar_interp,
                detJ_interp=detJ_interp,
                use_interp=use_interp,
                **kwargs
            )

            if not np.isfinite(L_i):
                print(f"[WARN] 非有限似然值: lens {idx}, L_i = {L_i}")
                return -np.inf
            if L_i <= 0:
                print(f"[WARN] 非正似然值: lens {idx}, L_i = {L_i}")
                return -np.inf

            logL += np.log(L_i / A_eta)

        except Exception as e:
            print(f"[ERROR] likelihood_single_fast_optimized 失败 at lens {idx}, eta = {eta}")
            print(f"        错误类型: {type(e).__name__}, 信息: {e}")
            return -np.inf

    return logL


# === 后验函数 ===
def log_posterior(eta, data_df, **kwargs):
    lp = log_prior(eta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(eta, data_df, **kwargs)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll



# warmup cache (提前填充 A_interp 缓存)
for mu in np.linspace(12.0, 13.0, 20):
    for sigma in np.linspace(0.05, 0.5, 20):
        _ = cached_A_interp(safe_round(mu), safe_round(sigma))

        
        

# ==== 主程序入口 ====
if __name__ == '__main__':
    # from your_module import mock_observed_data, A_interp  # 请替换为你自己的数据模块

    ndim = 4
    nwalkers = 8
    nsteps = 5000
    initial = np.array([12.5, 0.2, 0.05, 0.1])
    backend_file = "mcmc_checkpoint_eta_new_nomutab.h5"

    data_df = mock_observed_data  # 可替换为更多 lens
    logMh_grid = np.linspace(11.5, 14.0, 100)
    logMstar_list, detJ_list = build_interp_list_for_lenses(data_df, logMh_grid, zl=0.3, zs=2.0)

    if os.path.exists(backend_file):
        backend = HDFBackend(backend_file)
        print(f"[INFO] 继续从 {backend.iteration} 步开始采样")
        p0 = None
    else:
        backend = HDFBackend(backend_file)
        backend.reset(nwalkers, ndim)
        p0 = initial + 1e-3 * np.random.randn(nwalkers, ndim)

    with multiprocessing.get_context("fork").Pool(processes=multiprocessing.cpu_count()//2) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior,
            args=(data_df,),
            kwargs=dict(
                A_interp=A_interp,
                logMstar_interp_list=logMstar_list,
                detJ_interp_list=detJ_list,
                use_interp=True,
                gridN=30, zl=0.3, zs=2.0, ms=26.0, sigma_m=0.1, m_lim=26.5
            ),
            pool=pool,
            backend=backend,
        )

        print("[INFO] 开始并行采样...")
        sampler.run_mcmc(p0, nsteps, progress=True)

        samples = sampler.get_chain(discard=1000, thin=10, flat=True)
        log_probs = sampler.get_log_prob(discard=1000, thin=10, flat=True)
        np.savez("mcmc_results_eta_checkpoint.npz", samples=samples, log_probs=log_probs)
        print(f"[INFO] A_interp cache hits: {cached_A_interp.cache_info()}")


