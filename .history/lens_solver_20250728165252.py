from lens_model import LensModel
from scipy.interpolate import splrep, splint
from sl_cosmology import Dang, Mpc, c, G, M_Sun, rhoc
import numpy as np
from sl_profiles import nfw




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
