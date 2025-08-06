# === Imports ===
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.stats import norm
from scipy.special import erf
from ..lens_model import LensModel
from ..lens_solver import solve_single_lens

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

def generate_lens_samples_no_alpha(n_samples=1000, seed=42, mu_DM=13.0, sigma_DM=0.2, n_sigma=3):
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
    return list(zip(logMstar, logRe, logMh, beta)), (Mh_min, Mh_max)

# === Core Computation ===

def compute_A_phys_eta(mu_DM_cnst, beta_DM, xi_DM, sigma_DM, samples, Mh_range,
                       zl=0.3, zs=2.0, ms=26.0, sigma_m=0.1, m_lim=26.5):
    """
    计算 A(eta)，兼容 halo mass 模型：
        logMh ~ N(mu_DM + beta_DM * (logM* - 11.4) + xi_DM * (logRe - logRe_model(logM*)), sigma_DM)
    """
    Mh_min, Mh_max = Mh_range
    q_Mh = 1.0 / (Mh_max - Mh_min)

    total = 0.0
    valid = 0

    for logMstar, logRe, logMh, beta in samples:
        logRe_model = logRe_of_logMsps(logMstar)
        mu_DM_i = mu_DM_cnst + beta_DM * (logMstar - 11.4) + xi_DM * (logRe - logRe_model)
        p_Mh_i = norm.pdf(logMh, loc=mu_DM_i, scale=sigma_DM)
        w_i = p_Mh_i / q_Mh

        try:
            model = LensModel(logMstar, logMh, logRe, zl=zl, zs=zs)
            xA, xB = solve_single_lens(model, beta_unit=beta)
            muA, muB = model.mu_from_rt(xA), model.mu_from_rt(xB)
        except Exception:
            continue

        if muA <= 0 or muB <= 0 or not np.isfinite(muA * muB):
            continue

        magA = ms - 2.5 * np.log10(muA)
        magB = ms - 2.5 * np.log10(muB)

        sel_prob1 = 0.5 * (1 + erf((m_lim - magA) / (np.sqrt(2) * sigma_m)))
        sel_prob2 = 0.5 * (1 + erf((m_lim - magB) / (np.sqrt(2) * sigma_m)))
        sel_prob = sel_prob1 * sel_prob2

        total += sel_prob * w_i
        valid += 1

    return total / valid if valid > 0 else 0.0

# === 单点计算任务 ===

def single_A_eta_entry(args, seed=42):
    muDM, sigmaDM, beta_DM, xi_DM, n_samples, n_sigma = args
    samples, Mh_range = generate_lens_samples_no_alpha(
        n_samples=n_samples, mu_DM=muDM, sigma_DM=sigmaDM, n_sigma=n_sigma, seed=seed
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

def build_A_phys_table_parallel_4D(muDM_grid, sigmaDM_grid, betaDM_grid, xiDM_grid,
                                    n_samples=1000, n_sigma=3,
                                    filename='A_phys_table_4D.csv', nproc=None, batch_size=1000):
    if nproc is None:
        nproc = max(1, cpu_count() - 1)

    done_set = set()
    if os.path.exists(filename):
        df_done = pd.read_csv(filename)
        done_set = set(zip(df_done['mu_DM'], df_done['sigma_DM'],
                           df_done['beta_DM'], df_done['xi_DM']))
        print(f"[INFO] 已完成 {len(done_set)} 个点，将跳过这些")

    args_list = [
        (mu, sigma, beta, xi, n_samples, n_sigma)
        for mu in muDM_grid
        for sigma in sigmaDM_grid
        for beta in betaDM_grid
        for xi in xiDM_grid
        if (mu, sigma, beta, xi) not in done_set
    ]
    print(f"[INFO] 共需计算 {len(args_list)} 个 A(eta) 点")

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
    muDM_grid    = np.linspace(12.0, 13.2, 10)
    sigmaDM_grid = np.linspace(0.1, 0.3, 10)
    betaDM_grid  = np.linspace(0.0, 1.0, 10)
    xiDM_grid    = np.linspace(0.0, 0.5, 10)

    build_A_phys_table_parallel_4D(
        muDM_grid, sigmaDM_grid, betaDM_grid, xiDM_grid,
        n_samples=1000,
        filename="A_phys_table_4D.csv"
    )
