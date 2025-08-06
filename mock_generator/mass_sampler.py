import numpy as np
from scipy.stats import skewnorm, norm


# ==============================
# 模型参数（来源：Sonnenfeld+2019）
# ==============================
MODEL_PARAMS = {
    'deVauc': {
        'mu_star': 11.252,
        'sigma_star': 0.202,
        'log_s_star': 0.17,
        'mu_R0': 0.774,
        'beta_R': 0.977,
        'sigma_R': 0.112,
        'mu_h0': 12.91,
        'beta_h': 2.04,
        'xi_h': 0.0,
        'sigma_h': 0.37
    },
    'SerExp': {
        'mu_star': 11.274,
        'sigma_star': 0.254,
        'log_s_star': 0.31,
        'mu_R0': 0.854,
        'beta_R': 1.218,
        'sigma_R': 0.129,
        'mu_h0': 12.83,
        'beta_h': 1.73,
        'xi_h': -0.03,
        'sigma_h': 0.32
    },
    'Sersic': {
        'mu_star': 11.249,
        'sigma_star': 0.285,
        'log_s_star': 0.44,
        'mu_R0': 0.855,
        'beta_R': 1.366,
        'sigma_R': 0.147,
        'mu_h0': 12.79,
        'beta_h': 1.70,
        'xi_h': -0.14,
        'sigma_h': 0.35
    }
}

# ==============================
# 源星等分布采样
# ==============================
def sample_m_s(alpha_s, m_s_star, size=1, rng=None, m_min=20.0, m_max=30.0):
    """Sample source magnitudes from the Schechter-like distribution.

    The probability density is proportional to

    .. math::

        [10^{-0.4(m_s - m_s^*)}]^{\alpha_s + 1}\, \exp[-10^{-0.4(m_s - m_s^*)}].

    Parameters
    ----------
    alpha_s : float
        The power-law slope of the luminosity function.
    m_s_star : float
        The characteristic magnitude.
    size : int, optional
        Number of samples to draw, by default 1.
    rng : ``np.random.Generator``, optional
        Random-number generator for reproducibility.
    m_min, m_max : float, optional
        Allowed magnitude range for rejection sampling.

    Returns
    -------
    ndarray or float
        Sampled magnitudes.  A scalar is returned if ``size==1``.
    """

    if rng is None:
        rng = np.random.default_rng()

    def pdf(m):
        L = 10 ** (-0.4 * (m - m_s_star))
        return L ** (alpha_s + 1) * np.exp(-L)

    # Find an upper bound for the PDF within [m_min, m_max]
    grid = np.linspace(m_min, m_max, 1000)
    p_max = pdf(grid).max()

    samples = np.empty(size)
    i = 0
    while i < size:
        m_try = rng.uniform(m_min, m_max)
        u = rng.uniform(0, p_max)
        if u < pdf(m_try):
            samples[i] = m_try
            i += 1

    return samples if size > 1 else samples[0]

# ==============================
# 生成 logM_star_sps 样本（skew-normal）
# ==============================
def mstar_gene(n, model='deVauc', rng=None):
    p = MODEL_PARAMS[model]
    a = 10**p['log_s_star']
    dist = skewnorm(a=a, loc=p['mu_star'], scale=p['sigma_star'])
    return dist.rvs(size=n, random_state=rng)

# ==============================
# 生成 logRe（给定 logM_star_sps）
# ==============================
def logRe_given_logM(logM_star_sps, model='deVauc', rng=None):
    p = MODEL_PARAMS[model]
    mu_Re = p['mu_R0'] + p['beta_R'] * (logM_star_sps - 11.4)
    return norm.rvs(loc=mu_Re, scale=p['sigma_R'], size=len(logM_star_sps), random_state=rng)

# ==============================
# 生成 logMh（给定 logM_star_sps 和 logRe）
# ==============================
def logMh_given_logM_logRe(logM_star_sps, logRe, model='deVauc', rng=None):
    p = MODEL_PARAMS[model]
    mu_r = p['mu_R0'] + p['beta_R'] * (logM_star_sps - 11.4)
    mu_h = p['mu_h0'] + p['beta_h'] * (logM_star_sps - 11.4) + p['xi_h'] * (logRe - mu_r)
    return norm.rvs(loc=mu_h, scale=p['sigma_h'], size=len(logM_star_sps), random_state=rng)

# ==============================
# 主函数：生成完整样本
# ==============================
def generate_samples(
    n_samples,
    model='deVauc',
    random_state=None,
    alpha_s=-1.3,
    m_s_star=24.5,
):
    """
    生成星系参数样本，包括：
    - logM_star_sps: stellar mass (SPS)
    - logRe: effective radius
    - logMh: halo mass
    - z: redshift (固定为 1)
    - gamma_in: 初始内密度坡度（固定为 1）
    - C: halo 浓度参数（固定为 20）

    参数:
        n_samples: 样本数量
        model: 使用的结构模型，支持 'deVauc', 'SerExp', 'Sersic'
        random_state: 可选整数或 np.random.Generator

    返回:
        dict，包含字段 logM_star_sps, logRe, logMh, m_s, z, gamma_in, C
    """
    if isinstance(random_state, (int, type(None))):
        rng = np.random.default_rng(random_state)
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        raise ValueError("random_state must be int, None, or np.random.Generator")

    logM_star_sps = mstar_gene(n_samples, model=model, rng=rng)
    logRe = logRe_given_logM(logM_star_sps, model=model, rng=rng)
    logMh = logMh_given_logM_logRe(logM_star_sps, logRe, model=model, rng=rng)
    m_s = sample_m_s(alpha_s, m_s_star, size=n_samples, rng=rng)

    return {
        'logM_star_sps': logM_star_sps,
        'logRe': logRe,
        'logMh': logMh,
        'm_s': m_s,
        'z': np.ones(n_samples),
        'gamma_in': np.ones(n_samples),
        'C': np.ones(n_samples) * 20
    }


##############
# to be fixed "logMh"
##############


# ==============================
# 可选测试代码（直接运行本文件触发）
# ==============================
if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")  # 或 "Qt5Agg"，取决于你系统
    import matplotlib.pyplot as plt



    # === 定义：理论边际分布计算函数 ===
    def theoretical_logRe_pdf(x_logRe, params, n_grid=1000):
        m_grid = np.linspace(10.5, 12.5, n_grid)
        w_m = skewnorm.pdf(m_grid, a=10**params['log_s_star'],
                        loc=params['mu_star'], scale=params['sigma_star'])
        w_m /= np.trapz(w_m, m_grid)
        
        pdf_Re = np.zeros_like(x_logRe)
        for mi, wi in zip(m_grid, w_m):
            mu = params['mu_R0'] + params['beta_R'] * (mi - 11.4)
            pdf_Re += wi * norm.pdf(x_logRe, loc=mu, scale=params['sigma_R'])
        return pdf_Re

    def theoretical_logMh_pdf(x_logMh, params, n_grid=100):
        m_grid = np.linspace(10.5, 12.5, n_grid)
        w_m = skewnorm.pdf(m_grid, a=10**params['log_s_star'],
                        loc=params['mu_star'], scale=params['sigma_star'])
        w_m /= np.trapz(w_m, m_grid)
        
        r_grid = np.linspace(0.5, 2.5, n_grid)
        dr = r_grid[1] - r_grid[0]

        pdf_Mh = np.zeros_like(x_logMh)
        for mi, wi in zip(m_grid, w_m):
            mu_r = params['mu_R0'] + params['beta_R'] * (mi - 11.4)
            w_r = norm.pdf(r_grid, loc=mu_r, scale=params['sigma_R'])

            for rj, wj in zip(r_grid, w_r):
                mu_h = (params['mu_h0']
                        + params['beta_h'] * (mi - 11.4)
                        + params['xi_h'] * (rj - mu_r))
                pdf_Mh += wi * wj * norm.pdf(x_logMh, loc=mu_h, scale=params['sigma_h']) * dr
        return pdf_Mh

    # === 生成样本 ===
    model_name = 'deVauc'
    params = MODEL_PARAMS[model_name]
    samples = generate_samples(100000, model=model_name)

    logM_star_sps = samples['logM_star_sps']
    logRe = samples['logRe']
    logMh = samples['logMh']

    # === 绘图 ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. logM_star 分布
    x_logM = np.linspace(10.5, 12.5, 500)
    pdf_logM = skewnorm.pdf(x_logM, a=10**params['log_s_star'], loc=params['mu_star'], scale=params['sigma_star'])
    axes[0].hist(logM_star_sps, bins=100, density=True, alpha=0.6, color='orange', label='Sampled')
    axes[0].plot(x_logM, pdf_logM, 'r--', lw=2, label='Theoretical')
    axes[0].set_xlabel('$\\log M_*$')
    axes[0].set_ylabel('PDF')
    axes[0].legend()
    axes[0].set_title('Distribution of $\\log M_*$')

    # 2. logRe 分布
    x_logRe = np.linspace(0, 2.5, 500)
    pdf_logRe_theory = theoretical_logRe_pdf(x_logRe, params)
    pdf_logRe_theory /= np.trapz(pdf_logRe_theory, x_logRe)
    axes[1].hist(logRe, bins=100, density=True, alpha=0.6, color='blue', label='Sampled')
    axes[1].plot(x_logRe, pdf_logRe_theory, 'r--', lw=2, label='Theoretical')
    axes[1].set_xlabel('$\\log R_e$')
    axes[1].set_ylabel('PDF')
    axes[1].legend()
    axes[1].set_title('Distribution of $\\log R_e$')

    # 3. logMh 分布
    x_logMh = np.linspace(11.0, 15.0, 500)
    pdf_logMh_theory = theoretical_logMh_pdf(x_logMh, params)
    pdf_logMh_theory /= np.trapz(pdf_logMh_theory, x_logMh)

    axes[2].hist(logMh, bins=100, density=True, alpha=0.6, color='green', label='Sampled')
    axes[2].plot(x_logMh, pdf_logMh_theory, 'r--', lw=2, label='Theoretical')
    axes[2].set_xlabel('$\\log M_h$')
    axes[2].set_ylabel('PDF')
    axes[2].legend()
    axes[2].set_title('Distribution of $\\log M_h$')

    plt.tight_layout()
    plt.show()
