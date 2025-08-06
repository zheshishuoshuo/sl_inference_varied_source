import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import erf
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from ..lens_model import LensModel
from l..ens_solver import solve_single_lens

# ==== 常数 ====
m_lim = 26.5
sigma_m = 0.1
ms = 26

# ==== 样本生成函数 ====
def generate_lens_samples_no_alpha(n_samples=1000, seed=42, mu_DM=13.0, sigma_DM=0.2, n_sigma=3):
    rng = np.random.default_rng(seed)
    logMstar = rng.normal(11.4, 0.3, n_samples)
    logRe = rng.normal(1 + 0.8 * (logMstar - 11.4), 0.15, n_samples)
    
    Mh_min = mu_DM - n_sigma * sigma_DM
    Mh_max = mu_DM + n_sigma * sigma_DM
    logMh = rng.uniform(Mh_min, Mh_max, n_samples)
    
    beta = rng.uniform(0.0, 1.0, n_samples)
    return list(zip(logMstar, logRe, logMh, beta)), (Mh_min, Mh_max)

# ==== A_phys 计算函数 ====
def compute_A_phys_eta(mu_DM, sigma_DM, samples, Mh_range, zl=0.3, zs=2.0):
    Mh_min, Mh_max = Mh_range
    q_Mh = 1.0 / (Mh_max - Mh_min)

    logMh_array = np.array([s[2] for s in samples])
    p_Mh_array = norm.pdf(logMh_array, mu_DM, sigma_DM)
    w_array = p_Mh_array / q_Mh

    total = 0.0
    valid = 0

    for i, (logMstar, logRe, logMh, beta) in enumerate(samples):
        try:
            model = LensModel(logMstar, logMh, logRe, zl=zl, zs=zs)
            xA, xB = solve_single_lens(model, beta_unit=beta)
            muA, muB = model.mu_from_rt(xA), model.mu_from_rt(xB)
        except:
            continue

        if muA <= 0 or muB <= 0 or not np.isfinite(muA * muB):
            continue

        magA = ms - 2.5 * np.log10(muA)
        magB = ms - 2.5 * np.log10(muB)

        sel_prob1 = 0.5 * (1 + erf((m_lim - magA) / (np.sqrt(2) * sigma_m)))
        sel_prob2 = 0.5 * (1 + erf((m_lim - magB) / (np.sqrt(2) * sigma_m)))
        sel_prob = sel_prob1 * sel_prob2

        w = w_array[i]
        total += sel_prob * w
        valid += 1

    return total / valid if valid > 0 else 0.0

# ==== 单点任务 ====
def single_A_eta_entry(args):
    muDM, sigmaDM, n_samples, n_sigma = args
    samples, Mh_range = generate_lens_samples_no_alpha(
        n_samples=n_samples, mu_DM=muDM, sigma_DM=sigmaDM, n_sigma=n_sigma
    )
    A_eta = compute_A_phys_eta(
        mu_DM=muDM, sigma_DM=sigmaDM, samples=samples, Mh_range=Mh_range
    )
    return {'mu_DM': muDM, 'sigma_DM': sigmaDM, 'A_phys': A_eta}

# ==== 并行构建表格 ====
def build_A_phys_table_parallel(muDM_grid, sigmaDM_grid, n_samples=1000, n_sigma=3,
                                 filename='A_phys_table100.csv', nproc=None, batch_size=1000):
    if nproc is None:
        nproc = max(1, cpu_count() - 1)

    done_set = set()
    if os.path.exists(filename):
        df_done = pd.read_csv(filename)
        done_set = set(zip(df_done['mu_DM'], df_done['sigma_DM']))
        print(f"[INFO] 已完成 {len(done_set)} 个点，将跳过这些")

    args_list = [
        (muDM, sigmaDM, n_samples, n_sigma)
        for muDM in muDM_grid
        for sigmaDM in sigmaDM_grid
        if (muDM, sigmaDM) not in done_set
    ]
    print(f"[INFO] 共需计算 {len(args_list)} 个点")

    with Pool(nproc) as pool:
        buffer = []
        with open(filename, 'a') as f:
            if os.stat(filename).st_size == 0:
                f.write('mu_DM,sigma_DM,A_phys\n')

            for result in tqdm(pool.imap_unordered(single_A_eta_entry, args_list), total=len(args_list)):
                buffer.append(f"{result['mu_DM']},{result['sigma_DM']},{result['A_phys']}\n")

                if len(buffer) >= batch_size:
                    f.writelines(buffer)
                    f.flush()
                    buffer = []

            if buffer:
                f.writelines(buffer)
                f.flush()
    print(f"[INFO] 任务完成，结果保存在 {filename}")

# ==== 主程序 ====
if __name__ == "__main__":
    muDM_grid = np.linspace(9, 16, 1000)
    sigmaDM_grid = np.linspace(0, 1, 1000)
    build_A_phys_table_parallel(muDM_grid, sigmaDM_grid, nproc=cpu_count())
