import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from lens_model import LensModel
from solve_lens_parameters_from_obs import solve_lens_parameters_from_obs
from scipy.stats import norm
import os
import csv


# === 函数定义部分 ===

def logRe_of_logMsps(logMsps):
    mu_R_0 = 0.774
    beta_R = 0.977
    sigma_R = 0.112
    mu_R = mu_R_0 + beta_R * (logMsps - 11.4)
    return norm.rvs(loc=mu_R, scale=sigma_R)


def generate_lens_samples_no_alpha(n_samples, mu_DM, sigma_DM, beta_DM, xi_DM):
    logMsps = np.random.normal(loc=11.4, scale=0.3, size=n_samples)
    logRe = logRe_of_logMsps(logMsps)
    mu_r = 0.774 + 0.977 * (logMsps - 11.4)
    mu_h = mu_DM + beta_DM * (logMsps - 11.4) + xi_DM * (logRe - mu_r)
    logMh = np.random.normal(loc=mu_h, scale=sigma_DM)
    beta = np.random.uniform(0.01, 0.6, size=n_samples)
    return logMsps, logRe, logMh, beta


def compute_A_phys_eta(mu_DM, sigma_DM, beta_DM, xi_DM, n_samples=1000, n_sigma=3):
    count = 0
    logMsps, logRe, logMh, beta = generate_lens_samples_no_alpha(
        n_samples, mu_DM, sigma_DM, beta_DM, xi_DM
    )

    for i in range(n_samples):
        try:
            model = LensModel(logMsps[i], logMh[i], logRe[i], zl=0.3, zs=2.0)
            result = solve_lens_parameters_from_obs(model, beta[i])
            if result is not None:
                count += 1
        except:
            continue

    A_phys = count / n_samples
    return A_phys


# === 主程序部分 ===

if __name__ == "__main__":
    muDM_grid = np.linspace(9.0, 15.0, 100)
    sigmaDM_grid = np.linspace(0.05, 0.5, 10)
    betaDM_grid = [2.04]
    xiDM_grid = [0.0]
    n_samples = 1000
    n_sigma = 3
    filename = "A_phys_table_4D.csv"
    nproc = max(1, cpu_count() - 1)
    batch_size = 1000

    done_set = set()
    if os.path.exists(filename):
        df_done = pd.read_csv(filename)
        done_set = set(zip(df_done['mu_DM'], df_done['sigma_DM'],
                           df_done['beta_DM'], df_done['xi_DM']))
        print(f"[INFO] 已完成 {len(done_set)} 个点，将跳过这些")

    args_list = [
        (mu, sigma, beta, xi)
        for mu in muDM_grid
        for sigma in sigmaDM_grid
        for beta in betaDM_grid
        for xi in xiDM_grid
        if (mu, sigma, beta, xi) not in done_set
    ]

    def compute_and_pack(args):
        mu, sigma, beta, xi = args
        A = compute_A_phys_eta(mu, sigma, beta, xi, n_samples, n_sigma)
        return dict(mu_DM=mu, sigma_DM=sigma, beta_DM=beta, xi_DM=xi, A_phys=A)

    with Pool(processes=nproc) as pool, open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['mu_DM', 'sigma_DM', 'beta_DM', 'xi_DM', 'A_phys'])
        if os.stat(filename).st_size == 0:
            writer.writeheader()

        for i in tqdm(range(0, len(args_list), batch_size)):
            batch = args_list[i:i+batch_size]
            results = pool.map(compute_and_pack, batch)
            writer.writerows(results)
            f.flush()
            print(f"[INFO] 已写入 {i+len(batch)} / {len(args_list)}")
