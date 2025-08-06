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
# 生成 logM_star_sps 样本（skew-normal）
# ==============================
def mstar_gene(n, model='deVauc', random_state=None):
    p = MODEL_PARAMS[model]
    a = 10**p['log_s_star']
    dist = skewnorm(a=a, loc=p['mu_star'], scale=p['sigma_star'])
    return dist.rvs(size=n, random_state=random_state)

# ==============================
# 生成 logRe（给定 logM_star_sps）
# ==============================
def logRe_given_logM(logM_star_sps, model='deVauc', random_state=None):
    p = MODEL_PARAMS[model]
    mu_Re = p['mu_R0'] + p['beta_R'] * (logM_star_sps - 11.4)
    return norm.rvs(loc=mu_Re, scale=p['sigma_R'], size=len(logM_star_sps), random_state=random_state)

# ==============================
# 生成 logMh（给定 logM_star_sps 和 logRe）
# ==============================
def logMh_given_logM_logRe(logM_star_sps, logRe, model='deVauc', random_state=None):
    p = MODEL_PARAMS[model]
    mu_r = p['mu_R0'] + p['beta_R'] * (logM_star_sps - 11.4)
    mu_h = p['mu_h0'] + p['beta_h'] * (logM_star_sps - 11.4) + p['xi_h'] * (logRe - mu_r)
    return norm.rvs(loc=mu_h, scale=p['sigma_h'], size=len(logM_star_sps), random_state=random_state)

# ==============================
# 主函数：生成完整样本（接口不变）
# ==============================
def generate_samples(n_samples, model='deVauc', random_state=None):
    """
    生成星系参数样本，包括：
    - logM_star: stellar mass sps
    - logRe: effective radius
    - logMh: halo mass
    - z: redshift (固定为 1)
    - gamma_in: 初始内密度坡度（固定为 1）
    - C: halo 浓度参数（固定为 20）
    
    参数:
        n_samples: 样本数量
        model: 使用的结构模型，支持 'deVauc', 'SerExp', 'Sersic'
        random_state: 随机种子
    
    返回:
        dict，包含字段 logM_star, logRe, logMh, z, gamma_in, C
    """
    logM_star_sps = mstar_gene(n_samples, model=model, random_state=random_state)
    logRe = logRe_given_logM(logM_star_sps, model=model, random_state=random_state)
    logMh = logMh_given_logM_logRe(logM_star_sps, logRe, model=model, random_state=random_state)

    return {
        'logM_star': logM_star_sps,
        'logRe': logRe,
        'logMh': logMh,
        'z': np.ones(n_samples),
        'gamma_in': np.ones(n_samples),
        'C': np.ones(n_samples) * 20
    }

# ==============================
# 可选测试代码（直接运行本文件触发）
# ==============================
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    samples = generate_samples(10000, model='deVauc')
    plt.hist(samples['logM_star'], bins=50, density=True, alpha=0.6, color='orange', label='Sampled $\\log M_*$')
    
    x = np.linspace(10.5, 12.5, 500)
    params = MODEL_PARAMS['deVauc']
    pdf = skewnorm.pdf(x, a=10**params['log_s_star'], loc=params['mu_star'], scale=params['sigma_star'])
    plt.plot(x, pdf, 'k--', lw=2, label='Sonnenfeld+19 PDF')
    
    plt.xlabel('$\\log M_*$')
    plt.ylabel('PDF')
    plt.legend()
    plt.title('Comparison of $\\log M_*$ Distribution')
    plt.show()
