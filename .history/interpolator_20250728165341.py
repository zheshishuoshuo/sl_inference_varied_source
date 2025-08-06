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

