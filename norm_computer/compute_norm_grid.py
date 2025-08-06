# === Imports ===
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.stats import norm, skewnorm
from scipy.special import erf
from ..mock_generator.lens_model import LensModel
from ..mock_generator.lens_solver import solve_single_lens
from ..mock_generator.mass_sampler import MODEL_PARAMS, sample_m_s

MODEL_P = MODEL_PARAMS["deVauc"]
# === Utils ===
# to check. use it to model the relation between logM_sps and logRe
# def logRe_of_logMsps(logMsps):
#     """根据 logM_sps 估计 logRe"""
#     a = 0.6181
#     b = -6.2756
#     return a * logMsps + b


def logRe_of_logMsps(logMsps, model='deVauc'):
    """
    使用 Sonnenfeld+2019 的 Re–M* 关系估计 logRe 的均值
    """
    p = MODEL_PARAMS[model]
    return p['mu_R0'] + p['beta_R'] * (logMsps - 11.4)


# === Sample Generation ===

def generate_lens_samples_no_alpha(
    n_samples=1000,
    seed=42,
    mu_DM=13.0,
    sigma_DM=0.2,
    n_sigma=3,
    alpha_s=-1.3,
    m_s_star=24.5,
):
    """
    生成透镜样本，不使用 alpha_sps。
    输出：(logM_star, logRe, logMh, beta)，以及 Mh 的采样范围
    """
    rng = np.random.default_rng(seed)
    logMstar = rng.normal(11.4, 0.3, n_samples)
    logRe = rng.normal(1 + 0.8 * (logMstar - 11.4), 0.15, n_samples)
    Mh_min = mu_DM - n_sigma * sigma_DM
    Mh_max = mu_DM + n_sigma * sigma_DM
    logMh = rng.uniform(Mh_min, Mh_max, n_samples)
    beta = rng.uniform(0.0, 1.0, n_samples)
    m_s = sample_m_s(alpha_s, m_s_star, size=n_samples, rng=rng)
    return list(zip(logMstar, logRe, logMh, beta, m_s)), (Mh_min, Mh_max)

# === Core Computation ===
def compute_A_phys_eta(
    mu_DM_cnst,
    beta_DM,
    xi_DM,
    sigma_DM,
    samples,
    Mh_range,
    zl=0.3,
    zs=2.0,
    sigma_m=0.1,
    m_lim=26.5,
):
    """
    计算 A(η)：物理归一化因子，考虑所有先验权重
    """
    Mh_min, Mh_max = Mh_range
    q_Mh = 1.0 / (Mh_max - Mh_min)

    # === 星质量分布参数 ===
    a_skew = 10 ** MODEL_P['log_s_star']
    mu_star = MODEL_P['mu_star']
    sigma_star = MODEL_P['sigma_star']
    sigma_Re = MODEL_P['sigma_R']

    # === 将样本转换为 NumPy 数组 ===
    samples_array = np.asarray(samples)
    if samples_array.size == 0:
        return 0.0
    logMstar_array, logRe_array, logMh_array, beta_array, ms_array = samples_array.T

    # === DM 条件均值和权重 (向量化) ===
    logRe_model_array = logRe_of_logMsps(logMstar_array)
    mu_DM_i_array = (
        mu_DM_cnst
        + beta_DM * (logMstar_array - 11.4)
        + xi_DM * (logRe_array - logRe_model_array)
    )

    p_logMh = norm.pdf(logMh_array, loc=mu_DM_i_array, scale=sigma_DM)
    p_logMstar = skewnorm.pdf(
        logMstar_array, a=a_skew, loc=mu_star, scale=sigma_star
    )
    p_logRe = norm.pdf(logRe_array, loc=logRe_model_array, scale=sigma_Re)
    w_array = (p_logMh * p_logMstar * p_logRe) / q_Mh

    # === 对每个透镜求解放大率 ===
    n = len(samples_array)
    muA_array = np.full(n, np.nan)
    muB_array = np.full(n, np.nan)

    for i, (logMstar, logRe, logMh, beta) in enumerate(samples_array):
        try:
            model = LensModel(logMstar, logMh, logRe, zl=zl, zs=zs)
            xA, xB = solve_single_lens(model, beta_unit=beta)
            muA_array[i] = model.mu_from_rt(xA)
            muB_array[i] = model.mu_from_rt(xB)
        except Exception:
            continue

    # === 选择函数 (向量化) ===
    valid_mask = (
        (muA_array > 0)
        & (muB_array > 0)
        & np.isfinite(muA_array * muB_array)
    )

    sel_prob_array = np.zeros(n)
    if np.any(valid_mask):
        magA = ms_array[valid_mask] - 2.5 * np.log10(muA_array[valid_mask])
        magB = ms_array[valid_mask] - 2.5 * np.log10(muB_array[valid_mask])

        selA = 0.5 * (1 + erf((m_lim - magA) / (np.sqrt(2) * sigma_m)))
        selB = 0.5 * (1 + erf((m_lim - magB) / (np.sqrt(2) * sigma_m)))
        sel_prob_array[valid_mask] = selA * selB

    total = np.sum(sel_prob_array * w_array)
    valid = np.count_nonzero(valid_mask)

    return total / valid if valid > 0 else 0.0
# === 单点计算任务 ===

def single_A_eta_entry(args, seed=42):
    muDM, sigmaDM, beta_DM, xi_DM, n_samples, n_sigma = args
    samples, Mh_range = generate_lens_samples_no_alpha(
        n_samples=n_samples,
        mu_DM=muDM,
        sigma_DM=sigmaDM,
        n_sigma=n_sigma,
        seed=seed,
    )
    A_eta = compute_A_phys_eta(
        mu_DM_cnst=muDM,
        beta_DM=beta_DM,
        xi_DM=xi_DM,
        sigma_DM=sigmaDM,
        samples=samples,
        Mh_range=Mh_range
    )
    return {
        'mu_DM': muDM,
        'sigma_DM': sigmaDM,
        'beta_DM': beta_DM,
        'xi_DM': xi_DM,
        'A_phys': A_eta
    }

# === 并行构建 A_phys 表格 ===

# def build_A_phys_table_parallel_4D(muDM_grid, sigmaDM_grid, betaDM_grid, xiDM_grid,
#                                     n_samples=1000, n_sigma=3,
#                                     filename='A_phys_table_4D.csv', nproc=None, batch_size=1000):
#     if nproc is None:
#         nproc = max(1, cpu_count() - 1)
#     filename = os.path.join(os.path.dirname(__file__), '..', 'tables', filename)
#     done_set = set()
#     if os.path.exists(filename):
#         df_done = pd.read_csv(filename)
#         done_set = set(zip(df_done['mu_DM'], df_done['sigma_DM'],
#                            df_done['beta_DM'], df_done['xi_DM']))
#         print(f"[INFO] 已完成 {len(done_set)} 个点，将跳过这些")

#     args_list = [
#         (mu, sigma, beta, xi, n_samples, n_sigma)
#         for mu in muDM_grid
#         for sigma in sigmaDM_grid
#         for beta in betaDM_grid
#         for xi in xiDM_grid
#         if (mu, sigma, beta, xi) not in done_set
#     ]
#     print(f"[INFO] 共需计算 {len(args_list)} 个 A(eta) 点")

#     with Pool(nproc) as pool:
#         buffer = []
#         with open(filename, 'a') as f:
#             if os.stat(filename).st_size == 0:
#                 f.write('mu_DM,sigma_DM,beta_DM,xi_DM,A_phys\n')

#             for result in tqdm(pool.imap_unordered(single_A_eta_entry, args_list), total=len(args_list)):
#                 buffer.append(f"{result['mu_DM']},{result['sigma_DM']},{result['beta_DM']},{result['xi_DM']},{result['A_phys']}\n")
#                 if len(buffer) >= batch_size:
#                     f.writelines(buffer)
#                     f.flush()
#                     buffer = []

#             if buffer:
#                 f.writelines(buffer)
#                 f.flush()

#     print(f"[INFO] 所有任务完成，结果已保存到 {filename}")


def build_A_phys_table_parallel_4D(muDM_grid, sigmaDM_grid, betaDM_grid, xiDM_grid,
                                    n_samples=1000, n_sigma=3,
                                    filename='A_phys_table_4D.csv', nproc=None, batch_size=1000,
                                    prec=6):
    """
    构建 A_phys(eta) 插值表（四维），并行运行。
    使用浮点精度量化避免 (mu, sigma, beta, xi) 比较失败。
    """
    from functools import partial

    if nproc is None:
        nproc = max(1, cpu_count() - 4)

    # 包裹标量 xiDM_grid 为列表
    if np.isscalar(xiDM_grid):
        xiDM_grid = [float(xiDM_grid)]

    # === 浮点量化 key 工具 ===
    def key4(mu, s, b, x): return (round(mu, prec), round(s, prec), round(b, prec), round(x, prec))

    # === 已完成点集 ===
    done_set = set()
    if os.path.exists(filename):
        df_done = pd.read_csv(filename)
        done_set = set(key4(*row) for row in df_done[['mu_DM','sigma_DM','beta_DM','xi_DM']].to_numpy())
        print(f"[INFO] 已完成 {len(done_set)} 个点，将跳过这些")

    # === 待计算参数列表 ===
    args_list = [
        (mu, sigma, beta, xi, n_samples, n_sigma)
        for mu in muDM_grid
        for sigma in sigmaDM_grid
        for beta in betaDM_grid
        for xi in xiDM_grid
        if key4(mu, sigma, beta, xi) not in done_set
    ]
    print(f"[INFO] 共需计算 {len(args_list)} 个 A(eta) 点")

    # === 并行计算 ===
    with Pool(nproc) as pool:
        buffer = []
        with open(filename, 'a') as f:
            if os.stat(filename).st_size == 0:
                f.write('mu_DM,sigma_DM,beta_DM,xi_DM,A_phys\n')

            for result in tqdm(pool.imap_unordered(single_A_eta_entry, args_list), total=len(args_list)):
                buffer.append(f"{result['mu_DM']},{result['sigma_DM']},{result['beta_DM']},{result['xi_DM']},{result['A_phys']}\n")
                if len(buffer) >= batch_size:
                    f.writelines(buffer)
                    f.flush()
                    buffer = []

            if buffer:
                f.writelines(buffer)
                f.flush()

    print(f"[INFO] 所有任务完成，结果已保存到 {filename}")
    
# === 主程序入口 ===

if __name__ == "__main__":
    muDM_grid    = np.linspace(12, 13, 100)      # Δμ = 0.0345
    sigmaDM_grid = np.linspace(0.1, 0.5, 100)    # Δσ = 0.0138
    betaDM_grid  = np.linspace(1.0, 3.0, 100)    # Δβ = 0.069
    xiDM_grid    = 0                            # 固定为 0


    build_A_phys_table_parallel_4D(
        muDM_grid, sigmaDM_grid, betaDM_grid, xiDM_grid,
        n_samples=2000,
        filename="A_phys_table_4D.csv"
    )
