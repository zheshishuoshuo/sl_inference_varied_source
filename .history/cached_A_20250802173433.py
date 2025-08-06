import numpy as np
import pandas as pd
import os
from scipy.interpolate import RegularGridInterpolator




def load_A_phys_interpolator_4d(filename='A_phys_table_4D_dummy.csv'):
    df = pd.read_csv(filename)

    mu_unique = np.sort(df['mu_DM'].unique())
    sigma_unique = np.sort(df['sigma_DM'].unique())
    beta_unique = np.sort(df['beta_DM'].unique())
    xi_unique = np.sort(df['xi_DM'].unique())

    shape = (
        len(mu_unique),
        len(sigma_unique),
        len(beta_unique),
        len(xi_unique),
    )
    values = (
        df.set_index(['mu_DM', 'sigma_DM', 'beta_DM', 'xi_DM'])
        .sort_index()['A_phys']
        .values.reshape(shape)
    )

    interp = RegularGridInterpolator(
        (mu_unique, sigma_unique, beta_unique, xi_unique),
        values,
        bounds_error=False,
        fill_value=None,
    )
    return interp


prec = 'low'  # 默认精度

if prec == 'low':
    A_interp = load_A_phys_interpolator_4d(
        os.path.join(os.path.dirname(__file__), 'tables', 'A_phys_table_4D.csv')
    )
elif prec == 'high':
    A_interp = load_A_phys_interpolator_4d(
        os.path.join(os.path.dirname(__file__), 'tables', 'A_phys_table_4D.csv')
    )


# === A_interp wrapper ===
def cached_A_interp(mu0, sigma, beta, xi):

    return A_interp((mu0, sigma, beta, xi))
