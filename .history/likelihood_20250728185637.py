from utils import selection_function, mag_likelihood
from lens_model import LensModel
from lens_solver import solve_lens_parameters_from_obs, compute_detJ
from cached_A import cached_A_interp
from scipy.stats import norm
import numpy as np
from cached_A import safe_round
# def log_prior(eta): ...
# def log_likelihood(...): ...
# def log_posterior(...): ...
# def likelihood_single_fast_optimized(...): ...


# === 全局缓存变量 ===
# likelihood.py 顶部
_context = {
    "data_df": None,
    "logMstar_interp_list": None,
    "detJ_interp_list": None,
    "use_interp": False
}

def set_context(data_df, logMstar_interp_list, detJ_interp_list, use_interp=False):
    _context["data_df"] = data_df
    _context["logMstar_interp_list"] = logMstar_interp_list
    _context["detJ_interp_list"] = detJ_interp_list
    _context["use_interp"] = use_interp


def initializer_for_pool(data_df_, logMstar_list_, detJ_list_, use_interp_):
    global _data_df, _logMstar_interp_list, _detJ_interp_list, _use_interp
    _data_df = data_df_
    _logMstar_interp_list = logMstar_list_
    _detJ_interp_list = detJ_list_
    _use_interp = use_interp_




def log_prior(eta):
    mu_DM, sigma_DM, mu_alpha, sigma_alpha = eta
    if not (9 < mu_DM < 15 and 0 < sigma_DM < 5 and -0.2 < mu_alpha < 0.3 and 0.0 < sigma_alpha < 1):
        return -np.inf
    return 0.0  # flat prior

def likelihood_single_fast_optimized(
    di, eta, gridN=35, zl=0.3, zs=2.0, ms=26.0, sigma_m=0.1, m_lim=26.5,
    logMstar_interp=None, detJ_interp=None, use_interp=False
):
    xA_obs, xB_obs, logM_sps_obs, logRe_obs, m1_obs, m2_obs = di
    mu_DM, sigma_DM, mu_alpha, sigma_alpha = eta

    logMh_grid = np.linspace(mu_DM - 4*sigma_DM, mu_DM + 4*sigma_DM, gridN)
    logalpha_grid = np.linspace(mu_alpha - 4*sigma_alpha, mu_alpha + 4*sigma_alpha, gridN)
    Z = np.zeros((gridN, gridN))

    for i, logMh in enumerate(logMh_grid):
        try:
            if use_interp:
                logM_star = float(logMstar_interp(logMh))
                detJ = float(detJ_interp(logMh))
            else:
                logM_star, _ = solve_lens_parameters_from_obs(xA_obs, xB_obs, logRe_obs, logMh, zl, zs)
                detJ = compute_detJ(xA_obs, xB_obs, logRe_obs, logMh, zl, zs)
        except:
            continue

        try:
            model = LensModel(logM_star=logM_star, logM_halo=logMh, logRe=logRe_obs, zl=zl, zs=zs)
            muA, muB = model.mu_from_rt(xA_obs), model.mu_from_rt(xB_obs)
            selA = selection_function(muA, m_lim, ms, sigma_m)
            selB = selection_function(muB, m_lim, ms, sigma_m)
            p_magA = mag_likelihood(m1_obs, muA, ms, sigma_m)
            p_magB = mag_likelihood(m2_obs, muB, ms, sigma_m)
        except:
            continue

        for j, logalpha in enumerate(logalpha_grid):
            p_Mstar = norm.pdf(logM_sps_obs, loc=logM_star - logalpha, scale=0.1)
            p_logMh = norm.pdf(logMh, loc=mu_DM, scale=sigma_DM)
            p_logalpha = norm.pdf(logalpha, loc=mu_alpha, scale=sigma_alpha)

            Z[i, j] = p_Mstar * p_logMh * p_logalpha * detJ * selA * selB * p_magA * p_magB

    integral = np.trapz(np.trapz(Z, logalpha_grid, axis=1), logMh_grid)
    return max(integral, 1e-300)



def __log_likelihood(
    eta, data_df, logMstar_interp_list=None, detJ_interp_list=None,
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
            print(f"[ERROR] 读取第 {i} 行数据失败: {repr(e)}")
            return -np.inf

        try:
            if use_interp:
                try:
                    # logMstar_interp = logMstar_interp_list[idx]
                    # detJ_interp = detJ_interp_list[idx]
                    logMstar_interp = logMstar_interp_list[i]
                    detJ_interp = detJ_interp_list[i]

                except IndexError as e:
                    print(f"[ERROR] 插值器列表越界: lens {i}, total lenses = {len(logMstar_interp_list)}")
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
                print(f"[WARN] 非有限似然值: lens {i}, L_i = {L_i}")
                return -np.inf
            if L_i <= 0:
                print(f"[WARN] 非正似然值: lens {i}, L_i = {L_i}")
                return -np.inf

            logL += np.log(L_i / A_eta)

        except Exception as e:
            print(f"[ERROR] likelihood_single_fast_optimized 失败 at lens {i}, eta = {eta}")
            print(f"        错误类型: {type(e).__name__}, 信息: {e}")
            return -np.inf

    return logL

def log_likelihood(eta, **kwargs):
    # global _data_df, _logMstar_interp_list, _detJ_interp_list, _use_interp
    _data_df = _context["data_df"]
    _logMstar_interp_list = _context["logMstar_interp_list"]
    _detJ_interp_list = _context["detJ_interp_list"]
    _use_interp = _context["use_interp"]

    mu_DM, sigma_DM, mu_alpha, sigma_alpha = eta

    if sigma_DM <= 0 or sigma_alpha <= 0 or sigma_DM > 2.0 or sigma_alpha > 2.0:
        return -np.inf

    try:
        A_eta = cached_A_interp(safe_round(mu_DM), safe_round(sigma_DM))
        if not np.isfinite(A_eta) or A_eta <= 0:
            return -np.inf
    except Exception:
        return -np.inf

    logL = 0.0
    for i, (_, row) in enumerate(_data_df.iterrows()):
        try:
            di = tuple(row[col] for col in [
                'xA', 'xB', 'logMsps_observed', 'logRe',
                'magnitude_observedA', 'magnitude_observedB'
            ])
        except:
            return -np.inf

        logMstar_interp = _logMstar_interp_list[i] if _use_interp else None
        detJ_interp = _detJ_interp_list[i] if _use_interp else None

        try:
            L_i = likelihood_single_fast_optimized(
                di, eta,
                logMstar_interp=logMstar_interp,
                detJ_interp=detJ_interp,
                use_interp=_use_interp,
                **kwargs
            )
            if not np.isfinite(L_i) or L_i <= 0:
                return -np.inf
            logL += np.log(L_i / A_eta)
        except:
            return -np.inf

    return logL


# === 后验函数 ===
def __log_posterior(eta, data_df, **kwargs):
    lp = log_prior(eta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(eta, data_df, **kwargs)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def log_posterior(eta):
    lp = log_prior(eta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(eta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll



