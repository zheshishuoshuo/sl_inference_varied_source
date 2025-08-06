import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator




def load_A_phys_interpolator(filename='A_phys_table.csv'):
    df = pd.read_csv(filename)
    
    # 重新构建二维网格
    muDM_unique = np.sort(df['mu_DM'].unique())
    sigmaDM_unique = np.sort(df['sigma_DM'].unique())
    
    A_grid = df.pivot(index='mu_DM', columns='sigma_DM', values='A_phys').values
    
    interp = RegularGridInterpolator((muDM_unique, sigmaDM_unique), A_grid,
                                     bounds_error=False, fill_value=None)
    return interp


prec = 'low'  # 默认精度

if prec == 'low':
    A_interp = load_A_phys_interpolator("./tables/A_phys_table100.csv")
elif prec == 'high':
    A_interp = load_A_phys_interpolator("./table/A_phys_table1000.csv")


# === A_interp wrapper ===
def cached_A_interp(mu_DM, sigma_DM):

    return A_interp((mu_DM, sigma_DM))
