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


