from lens_solver import solve_lens_parameters_from_obs, compute_detJ
from scipy.interpolate import interp1d
import numpy as np

def solve_lens_tabulate(...): ...
def detJ_tabulate(...): ...
def build_interp_list_for_lenses(data_df, logMh_grid, zl=0.3, zs=2.0): ...


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

