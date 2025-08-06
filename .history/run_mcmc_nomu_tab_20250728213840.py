# %%
import os
import sys
import warnings
from itertools import product
from time import time
import numpy as np
import emcee
import multiprocessing
import os
from emcee.backends import HDFBackend
import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits as pyfits
from scipy.integrate import quad
from scipy.interpolate import interp1d, splev, splint, splrep
from scipy.optimize import brentq, leastsq, minimize_scalar
from scipy.special import erf
from scipy.stats import norm, rv_continuous, truncnorm
import numpy as np
import emcee
import multiprocessing
from functools import lru_cache
from tqdm import tqdm

import sl_cosmology
from sl_cosmology import Dang, G, M_Sun, Mpc, c, rhoc, yr
from sl_profiles import deVaucouleurs as deV
from sl_profiles import nfw
import numpy as np
from scipy.stats import norm
from scipy.special import erf
from scipy.interpolate import RegularGridInterpolator
import emcee
import matplotlib.pyplot as plt
from mass_sampler import generate_samples
import numpy as np


def arcsec_to_kpc(arcsec, z_l, Dang_func):

    D_d = Dang_func(z_l)  # in Mpc
    arcsec2rad = np.deg2rad(1. / 3600.)  # 1 arcsecond in radians
    kpc = arcsec * arcsec2rad * D_d * 1000.  # kpc conversion (in Mpc)
    return kpc
def kpc_to_arcsec(kpc, z_l, Dang_func):

    D_d = Dang_func(z_l)  # in Mpc
    kpc2rad = kpc / (D_d * 1000.)  # Convert kpc to radians
    arcsec = np.rad2deg(kpc2rad) * 3600.  # Convert radians to arcseconds
    return arcsec


class LensModel:
    """
    LensModel 类：
    所有内部计算单位：
    - M_star, M_halo：Msun（太阳质量）
    - Re, rs, x, R：kpc
    - Sigma, s_cr：Msun / kpc^2
    - alpha(x)：kpc（也提供 arcsec 接口）
    - theta：arcsec（角度单位）
    - kappa, gamma：无量纲
    """

    def __init__(self, logM_star, logM_halo, logRe, zl, zs):
        self.zl, self.zs = zl, zs
        self.logM_star = logM_star  # [Msun]
        self.logM_halo = logM_halo  # [Msun]
        self.logRe = logRe            # [kpc]
        self.M_star = 10**logM_star           # [Msun]
        self.M_halo = 10**logM_halo           # [Msun]
        self.Re = 10**logRe                   # [kpc]
        self._setup_cosmology()               # 计算 s_cr 等
        self._setup_profiles()                # NFW + deV 分布
        self._setup_splines()                 # 插值缓存

    def _setup_cosmology(self):
        dd = Dang(self.zl)                    # [Mpc]
        ds = Dang(self.zs)                    # [Mpc]
        dds = Dang(self.zs, self.zl)          # [Mpc]
        kpc = Mpc / 1000.                     # [kpc/Mpc]
        self.s_cr = c**2 / (4*np.pi*G) * ds/dds/dd / Mpc / M_Sun * kpc**2  # [Msun / kpc^2]
        self.rhoc_z = rhoc(self.zl)           # [Msun / Mpc^3]

    def _setup_profiles(self):
        r200 = (self.M_halo * 3./(4*np.pi*200*self.rhoc_z))**(1./3.) * 1000.  # [kpc]
        self.rs = 10.0            # [kpc]
        self.nfw_norm = self.M_halo / nfw.M3d(r200, self.rs)

        self.R2d = np.logspace(-3, 2, 1001)     # [R/Re] 无单位
        self.Rkpc = self.R2d * self.rs        # [kpc]
        Sigma = self.nfw_norm * nfw.Sigma(self.Rkpc, self.rs)  # [Msun / kpc^2]
        self._Sigma = Sigma

    def _setup_splines(self):
        self.sigma_spline = splrep(self.Rkpc, self._Sigma)                # Sigma(R)
        self.sigmaR_spline = splrep(self.Rkpc, self._Sigma * self.Rkpc)  # Sigma(R) * R
        kappa_array = splev(self.Rkpc, self.sigma_spline) / self.s_cr
        self.kappaR_spline = splrep(self.Rkpc, kappa_array * self.Rkpc)  # for gamma

    def alpha_theta(self, theta):  # [arcsec] → [arcsec]
        x = arcsec_to_kpc(theta, self.zl, Dang)        # [kpc]
        alphakpc = self.alpha(x)                       # [kpc]
        return kpc_to_arcsec(alphakpc, self.zl, Dang)  # [arcsec]

    def alpha(self, x):  # [kpc] → [kpc]
        m2d_halo = 2*np.pi * splint(0., abs(x), self.sigmaR_spline)  # [Msun]
        m2d_star = self.M_star * deV.fast_M2d(abs(x)/self.Re)        # [Msun]
        return (m2d_halo + m2d_star) / (np.pi * x * self.s_cr)       # [kpc]
    

    def kappa(self, x):  # [kpc] → [dimensionless]
        sigma_halo = splev(abs(x), self.sigma_spline)                      # [Msun / kpc^2]
        sigma_star = self.M_star * deV.Sigma(abs(x), self.Re)             # [Msun / kpc^2]
        return (sigma_halo + sigma_star) / self.s_cr                      # [dimensionless]
    

    def kappa_star(self, x):  # [kpc] → [dimensionless]
        sigma_star = self.M_star * deV.Sigma(abs(x), self.Re)  # [Msun / kpc^2]
        return sigma_star / self.s_cr


    def gamma(self, x):
        R = abs(x)
        integrand = lambda r: self.kappa(r) * r
        integral, _ = quad(integrand, 0, R, epsabs=1e-8, epsrel=1e-8)
        mean_kappa = 2 * integral / R**2
        return mean_kappa - self.kappa(R)



    def einstein_radius(self):  # [kpc]
        def zerofunc(x): return self.alpha(x) - x
        xmin = 1.01 * self.Rkpc[0]
        xmax = 0.99 * self.Rkpc[-1]
        if zerofunc(xmin)<0:
            return 0.  # 没有爱因斯坦半径
        elif zerofunc(xmax) > 0:
            return np.inf  # 无穷大爱因斯坦半径（无界
        else:
            return brentq(zerofunc, xmin, xmax)
    
    
    def mu_from_rt(self, x):
        """
        使用 μ_r 与 μ_t 计算放大率 (径向/切向放大率法)
        μ_r⁻¹ = 1 + α/x - 2κ
        μ_t⁻¹ = 1 - α/x
        μ = |μ_r · μ_t|
        """
        xabs = abs(x)
        alpha_val = self.alpha(xabs)
        kappa_val = self.kappa(xabs)

        inv_mur = 1. + alpha_val / xabs - 2. * kappa_val
        inv_mut = 1. - alpha_val / xabs

        if inv_mur * inv_mut == 0:
            return np.inf
        return np.abs(1. / (inv_mur * inv_mut))

    def mu_from_kg(self, x):
        """
        使用 κ 与 γ 计算放大率 (广义公式)
        μ = 1 / [ (1 - κ)^2 - γ^2 ]
        """
        kappa_val = self.kappa(x)
        gamma_val = self.gamma(x)
        denom = (1. - kappa_val)**2 - gamma_val**2
        if denom == 0:
            return np.inf
        return np.abs(1. / denom)

    
    def solve_xradcrit(self):  # [kpc]
        """求解径向临界曲线的位置 x_radcrit（图像平面，单位：kpc）"""
        def radial_invmag(x):
            return 1. + self.alpha(x)/x - 2.*self.kappa(x)

        xmin = 1.01 * self.Rkpc[0]
        xmax = 0.99 * self.Rkpc[-1]

        if radial_invmag(xmin) * radial_invmag(xmax) > 0:
            return None  # 没有临界曲线
        return brentq(radial_invmag, xmin, xmax)


    def solve_ycaustic(self):  # [kpc]
        """由 x_radcrit 推出径向 caustic 的 y 值（源平面，单位：kpc）"""
        xradcrit = self.solve_xradcrit()
        if xradcrit is None:
            return None
        return  self.alpha(xradcrit) - xradcrit 

    def solve_ycaustic_arcsec(self):
        """返回 ycaustic，单位为 arcsec"""
        y_kpc = self.solve_ycaustic()
        if y_kpc is None:
            return None
        return kpc_to_arcsec(y_kpc, self.zl, Dang)



def solve_single_lens(model, beta_unit):
   caustic_max_at_lens_plane = model.solve_xradcrit()  # [kpc]
   caustic_max_at_source_plane = model.solve_ycaustic()  # [kpc]
   beta = beta_unit * caustic_max_at_source_plane  # [kpc]
   einstein_radius = model.einstein_radius()  # [kpc]

   def lens_equation(x):
       return model.alpha(x) - x + beta
   
   xA = brentq(lens_equation, einstein_radius, 100*einstein_radius)
   xB = brentq(lens_equation, -einstein_radius, -caustic_max_at_lens_plane)
   # print(caustic_max_at_lens_plane, caustic_max_at_source_plane, beta)
   return xA, xB

# calculate lens properties for a single lens

def lens_properties(model, beta_unit=0.5, logalpha_sps=0.1):
   xA, xB = solve_single_lens(model, beta_unit)
   kappaA = model.kappa(xA)
   kappaB = model.kappa(xB)
   gammaA = model.gamma(xA)
   gammaB = model.gamma(xB)
   magnificationA = model.mu_from_rt(xA)
   magnificationB = model.mu_from_rt(xB)
   kappa_starA = model.kappa_star(xA)
   kappa_starB = model.kappa_star(xB)
   alphaA = model.alpha(xA)
   alphaB = model.alpha(xB)  
       
   sA = 1- kappa_starA/kappaA
   sB = 1- kappa_starB/kappaB  

   logMsps = model.logM_star - logalpha_sps
   scatter_Mstar = 0.1  # [Msun] 源质量的散射
   logMsps_observed = logMsps + np.random.normal(loc=0.0, scale=scatter_Mstar)  # 添加噪声

   einstein_radius = model.einstein_radius()  # [kpc]
   einstein_radius_arcsec = kpc_to_arcsec(einstein_radius, model.zl, Dang)  # [arcsec]

   return {
       'xA': xA, 'xB': xB, 'beta': beta_unit,
       'kappaA': kappaA, 'kappaB': kappaB,
       'gammaA': gammaA, 'gammaB': gammaB,
       'magnificationA': magnificationA, 'magnificationB': magnificationB,
       'kappa_starA': kappa_starA, 'kappa_starB': kappa_starB,
       'alphaA': alphaA, 'alphaB': alphaB,
       'sA': sA, 'sB': sB,
       'logMsps_observed': logMsps_observed,  # [Msun]
       'logMsps': logMsps,  # [Msun]
       'einstein_radius_kpc': einstein_radius,
       'einstein_radius_arcsec': einstein_radius_arcsec
   }

   
   
# add source properties

def observed_data(input_df, caustic= False):
   """
   计算 lens 的属性，并返回包含源属性的字典
   """
   mag_source = input_df['mag_source'].values[0]  # [mag]
   maximum_magnitude = input_df['maximum_magnitude'].values[0]  # [mag]
   beta_unit = input_df['beta_unit'].values[0]  # [kpc]
   logalpha_sps = input_df['logalpha_sps'].values[0]
   logM_star = input_df['logM_star'].values[0]  # [Msun]
   logM_halo = input_df['logM_halo'].values[0]  # [Msun]
   logRe = input_df['logRe'].values[0]  # [kpc]
   zl = input_df['zl'].values[0]  # [redshift]
   zs = input_df['zs'].values[0]  # [redshift]

   model = LensModel(logM_star=logM_star, logM_halo=logM_halo, logRe=logRe, zl=zl, zs=zs)
   properties = lens_properties(model, beta_unit, logalpha_sps)

   # magnificationA, magnificationB = properties['magnificationA'], properties['magnificationB']
   
   scatter_mag = 0.1  # [mag] 源光度的散射
   properties['scatter_mag'] = scatter_mag
   magnitude_observedA = mag_source - 2.5 * np.log10(properties['magnificationA']) + np.random.normal(loc=0.0, scale=scatter_mag)
   magnitude_observedB = mag_source - 2.5 * np.log10(properties['magnificationB']) + np.random.normal(loc=0.0, scale=scatter_mag)

   # no observed error
   if magnitude_observedA > maximum_magnitude or magnitude_observedB > maximum_magnitude:
       properties['is_lensed'] = False
   else:
       properties['is_lensed'] = True

   # 添加源属性
   properties['magnitude_observedA'] = magnitude_observedA  # [mag]
   properties['magnitude_observedB'] = magnitude_observedB  # [mag]
   properties['mag_source'] = mag_source  # [mag]
   properties['maximum_magnitude'] = maximum_magnitude  # [mag]
   properties['beta_unit'] = beta_unit  # [kpc]
   properties['logalpha_sps'] = logalpha_sps  # [Msun]
   properties['logM_star'] = model.logM_star
   properties['logM_halo'] = model.logM_halo
   properties['logRe'] = model.logRe
   properties['zl'] = model.zl
   properties['zs'] = model.zs

   if caustic:
       properties['ycaustic_kpc'] = model.solve_ycaustic()
       properties['ycaustic_arcsec'] = model.solve_ycaustic_arcsec()
       properties['xradcrit_kpc'] = model.solve_xradcrit()
       properties['xradcrit_arcsec'] = kpc_to_arcsec(properties['xradcrit_kpc'], model.zl, Dang)
   
   return properties



n_samples = 100

beta_scalefree_samp = np.random.rand(n_samples)**0.5

alpha_sps = np.random.normal(loc=1.2, scale=0.2, size=n_samples)

logalpha_sps_sample = np.log10(alpha_sps)
# 生成样本
samples = generate_samples(n_samples)

# 转换为 numpy array（按列顺序）
import numpy as np

lens_data = np.vstack([
    samples['logM_star'],
    samples['logMh'],
    samples['logRe'],
    beta_scalefree_samp,
    logalpha_sps_sample,
]).T





lens_results = []

# 常数参数（可替换为数组）
mag_source = 26
maximum_magnitude = 26.5
zl = 0.3
zs = 2.0

# tqdm 包裹 zip 迭代器
for logM_star, logMh, logRe, logalpha_sps, beta_unit in tqdm(zip(
        samples['logM_star'],
        samples['logMh'],
        samples['logRe'],
        logalpha_sps_sample,
        beta_scalefree_samp),
        total=len(samples['logM_star']),
        desc="Processing lenses"):

    input_df = pd.DataFrame({
        'logM_star': [logM_star],
        'logM_halo': [logMh],
        'logRe': [logRe],
        'beta_unit': [beta_unit],
        'mag_source': [mag_source],
        'maximum_magnitude': [maximum_magnitude],
        'logalpha_sps': [logalpha_sps],
        'zl': [zl],
        'zs': [zs]
    })

    result = observed_data(input_df, caustic=False)
    lens_results.append(result)

# 转为 DataFrame
df_lens = pd.DataFrame(lens_results)
df_lens


mock_lens_data = df_lens[df_lens['is_lensed'] == True]
mock_lens_data


mock_observed_data = df_lens[df_lens['is_lensed'] == True][['xA', 'xB', 'logMsps_observed', 'logRe', 'magnitude_observedA', 'magnitude_observedB']].copy()
mock_observed_data




# def generate_lens_samples_no_alpha(n_samples=1000, seed=42, mu_DM=13.0, sigma_DM=0.2, n_sigma=3):
#     rng = np.random.default_rng(seed)
#     logMstar = rng.normal(11.4, 0.3, n_samples)
#     logRe = rng.normal(1 + 0.8 * (logMstar - 11.4), 0.15, n_samples)
    
#     Mh_min = mu_DM - n_sigma * sigma_DM
#     Mh_max = mu_DM + n_sigma * sigma_DM
#     logMh = rng.uniform(Mh_min, Mh_max, n_samples)
    
#     beta = rng.uniform(0.0, 1.0, n_samples)
#     return list(zip(logMstar, logRe, logMh, beta)), (Mh_min, Mh_max)



# # ==== 2. alpha 样本 ====
# def generate_alpha_samples(n_samples=1000, mu_alpha=0.0, sigma_alpha=0.3, seed=123):
#     rng = np.random.default_rng(seed)
#     alpha = rng.uniform(-3.6, 2.9, n_samples)  # proposal range
#     weights = norm.pdf(alpha, loc=mu_alpha, scale=sigma_alpha)
#     return alpha, weights


# def compute_A_phys_eta(mu_DM, sigma_DM, samples, Mh_range, zl=0.3, zs=2.0):
#     Mh_min, Mh_max = Mh_range
#     q_Mh = 1.0 / (Mh_max - Mh_min)

#     logMh_array = np.array([s[2] for s in samples])  # 取出所有 logMh
#     p_Mh_array = norm.pdf(logMh_array, mu_DM, sigma_DM)
#     w_array = p_Mh_array / q_Mh  # 提前计算所有采样点的 p/q 权重

#     total = 0.0
#     valid = 0

#     for i, (logMstar, logRe, logMh, beta) in enumerate(samples):
#         try:
#             model = LensModel(logMstar, logMh, logRe, zl=zl, zs=zs)
#             xA, xB = solve_single_lens(model, beta_unit=beta)
#             muA, muB = model.mu_from_rt(xA), model.mu_from_rt(xB)
#         except:
#             continue

#         if muA <= 0 or muB <= 0 or not np.isfinite(muA * muB):
#             continue

#         magA = ms - 2.5 * np.log10(muA)
#         magB = ms - 2.5 * np.log10(muB)

#         sel_prob1 = 0.5 * (1 + erf((m_lim - magA) / (np.sqrt(2) * sigma_m)))
#         sel_prob2 = 0.5 * (1 + erf((m_lim - magB) / (np.sqrt(2) * sigma_m)))
#         sel_prob = sel_prob1 * sel_prob2

#         w = w_array[i]  # 快速读取预计算权重
#         total += sel_prob * w
#         valid += 1

#     return total / valid if valid > 0 else 0.0




def solve_lens_parameters_from_obs(xA_obs, xB_obs, logRe_obs, logM_halo, zl, zs):

    Re = 10**logRe_obs  # [kpc]
    rs = 10.0  # [kpc] NFW scale radius
    M_halo = 10**logM_halo  # [Msun]

    dd = Dang(zl)  # [Mpc]
    ds = Dang(zs)  # [Mpc]
    dds = Dang(zs, zl)  # [Mpc]
    kpc = Mpc / 1000.  # [kpc/Mpc]
    s_cr = c**2 / (4*np.pi*G) * ds/dds/dd / Mpc / M_Sun * kpc**2  # [Msun / kpc^2]
    rhoc_z = rhoc(zl)  # [Msun / Mpc^3]
    r200 = (M_halo * 3./(4*np.pi*200*rhoc_z))**(1./3.) * 1000.  # [kpc]
    nfw_norm = M_halo / nfw.M3d(r200, rs)
    R2d = np.logspace(-3, 2, 1001)  # [R/Re] 无单位
    Rkpc = R2d * rs  # [kpc]
    Sigma = nfw_norm * nfw.Sigma(Rkpc, rs)  # [Msun / kpc^2]
    # sigma_spline = splrep(Rkpc, Sigma)  # Sigma(R)
    sigmaR_spline = splrep(Rkpc, Sigma * Rkpc)  # Sigma(R) * R

    def alpha_star_unit(x):
        m2d_star =  deV.fast_M2d(abs(x)/Re)        # [Msun]
        return m2d_star / (np.pi * x * s_cr) 
    
    def alpha_halo(x):
        m2d_halo = 2*np.pi * splint(0., abs(x), sigmaR_spline)
        return m2d_halo / (np.pi * x * s_cr)

    M_star_solved = ((xA_obs -xB_obs) + alpha_halo(xB_obs)-alpha_halo(xA_obs))/ (alpha_star_unit(xA_obs) - alpha_star_unit(xB_obs))  # [Msun]
    beta_solved = -(alpha_star_unit(xA_obs)*(xB_obs-alpha_halo(xB_obs)) -alpha_star_unit(xB_obs)*(xA_obs-alpha_halo(xA_obs))) / (alpha_star_unit(xB_obs) - alpha_star_unit(xA_obs))  # [kpc]

    another_solution_beta = M_star_solved*alpha_star_unit(xA_obs) +alpha_halo(xA_obs) - xA_obs  # [kpc]
    
    # logbeta_solved = np.log10(beta_solved)  # [kpc]
    if M_star_solved <= 0:
        # print(f"Warning at Mh = {logM_halo}, xA = {xA_obs}, xB = {xB_obs}, Re = {Re}, beta = {beta_solved}")
        raise ValueError(f"Invalid M_star_solved = {M_star_solved}, must be > 0")
    logM_star_solved = np.log10(M_star_solved)  # [Msun]
    model = LensModel(logM_star=logM_star_solved, logM_halo=logM_halo, logRe=np.log10(Re), zl=zl, zs=zs)
    caustic_max_at_lens_plane = model.solve_xradcrit()  # [kpc]
    caustic_max_at_source_plane = model.solve_ycaustic()  # [kpc]
    # beta = beta_unit * caustic_max_at_source_plane  # [kpc]
    beta_unit = beta_solved / caustic_max_at_source_plane  # [kpc]
    # print('truebeta',beta_solved, another_solution_beta,beta_unit)

    return logM_star_solved, beta_unit  


    
def load_A_phys_interpolator(filename='A_phys_table.csv'):
    df = pd.read_csv(filename)
    
    # 重新构建二维网格
    muDM_unique = np.sort(df['mu_DM'].unique())
    sigmaDM_unique = np.sort(df['sigma_DM'].unique())
    
    A_grid = df.pivot(index='mu_DM', columns='sigma_DM', values='A_phys').values
    
    interp = RegularGridInterpolator((muDM_unique, sigmaDM_unique), A_grid,
                                     bounds_error=False, fill_value=None)
    return interp


A_interp = load_A_phys_interpolator("A_phys_table100.csv")






def selection_function(mu, m_lim, ms, sigma_m):
    """选择函数（单个图像）"""
    return 0.5 * (1 + erf((m_lim - ms + 2.5 * np.log10(np.abs(mu))) / (np.sqrt(2) * sigma_m)))

def mag_likelihood(m_obs, mu, ms, sigma_m):
    """星等的似然函数"""
    mag_model = ms - 2.5 * np.log10(np.abs(mu))
    return norm.pdf(m_obs, loc=mag_model, scale=sigma_m)

def compute_detJ(theta1_obs, theta2_obs, logRe_obs, logMh, zl=0.3, zs=2.0):
    delta = 1e-4  # arcsec

    logM0, beta0 = solve_lens_parameters_from_obs(theta1_obs, theta2_obs, logRe_obs, logMh, zl, zs)
    logM1, beta1 = solve_lens_parameters_from_obs(theta1_obs + delta, theta2_obs, logRe_obs, logMh, zl, zs)
    logM2, beta2 = solve_lens_parameters_from_obs(theta1_obs, theta2_obs + delta, logRe_obs, logMh, zl, zs)

    dlogM_dtheta1 = (logM1 - logM0) / delta
    dlogM_dtheta2 = (logM2 - logM0) / delta
    dbeta_dtheta1 = (beta1 - beta0) / delta
    dbeta_dtheta2 = (beta2 - beta0) / delta

    J = np.array([[dlogM_dtheta1, dlogM_dtheta2],
                  [dbeta_dtheta1, dbeta_dtheta2]])
    return np.abs(np.linalg.det(J))

def log_prior(eta):
    mu_DM, sigma_DM, mu_alpha, sigma_alpha = eta
    if not (9 < mu_DM < 15 and 0 < sigma_DM < 5 and -0.2 < mu_alpha < 0.3 and 0.0 < sigma_alpha < 1):
        return -np.inf
    return 0.0  # flat prior




# -----------
# tabulate

import numpy as np
from scipy.interpolate import interp1d
import h5py

def solve_lens_tabulate(logMh_grid, xA, xB, logRe, zl=0.3, zs=2.0):
    logMstar_list = []
    for logMh in logMh_grid:
        try:
            logMstar, _ = solve_lens_parameters_from_obs(xA, xB, logRe, logMh, zl, zs)
        except Exception:
            logMstar = np.nan
        logMstar_list.append(logMstar)
    logMstar_array = np.array(logMstar_list)
    valid = ~np.isnan(logMstar_array)
    return interp1d(logMh_grid[valid], logMstar_array[valid], kind='linear', fill_value='extrapolate')

def detJ_tabulate(logMh_grid, xA, xB, logRe, zl=0.3, zs=2.0):
    detJ_list = []
    for logMh in logMh_grid:
        try:
            detJ = compute_detJ(xA, xB, logRe, logMh, zl, zs)
        except Exception:
            detJ = 0.0
        detJ_list.append(detJ)
    detJ_array = np.array(detJ_list)
    valid = detJ_array > 0
    return interp1d(logMh_grid[valid], detJ_array[valid], kind='linear', fill_value=0.0)



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


def build_interp_list_for_lenses(data_df, logMh_grid, zl=0.3, zs=2.0):
    logMstar_list, detJ_list = [], []

    for idx, row in data_df.iterrows():
        xA, xB, logRe = row['xA'], row['xB'], row['logRe']
        try:
            logMstar_interp = solve_lens_tabulate(logMh_grid, xA, xB, logRe, zl, zs)
            detJ_interp = detJ_tabulate(logMh_grid, xA, xB, logRe, zl, zs)
        except Exception as e:
            print(f"[ERROR] 插值器构建失败: lens {idx}, xA={xA:.3f}, xB={xB:.3f}, logRe={logRe:.3f}")
            print(f"[ERROR] 失败原因: {e}")
            raise RuntimeError(f"lens #{idx} 插值器构建失败，终止程序。")

        logMstar_list.append(logMstar_interp)
        detJ_list.append(detJ_interp)

    # === 安全性断言：确保每个 lens 都成功生成了插值器 ===
    assert len(logMstar_list) == len(data_df) == len(detJ_list), \
        f"[FATAL] 插值器数量与 lens 数不一致！lens 数={len(data_df)}, logMstar={len(logMstar_list)}, detJ={len(detJ_list)}"

    return logMstar_list, detJ_list


# === 安全取整，避免缓存 key 精度差异 ===
def safe_round(x, ndigits=4):
    return round(float(x), ndigits)

# === 缓存包装的 A_interp ===
@lru_cache(maxsize=512)
def cached_A_interp(mu_DM, sigma_DM):
    return A_interp((mu_DM, sigma_DM))


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


