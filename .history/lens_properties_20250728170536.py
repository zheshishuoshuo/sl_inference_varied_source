from lens_solver import solve_single_lens
from lens_model import kpc_to_arcsec, LensModel
from sl_cosmology import Dang
import numpy as np

def lens_properties(model, beta_unit=0.5, logalpha_sps=0.1):
    xA, xB = solve_single_lens(model, beta_unit)
    ...
    return { ... }

def observed_data(input_df, caustic=False):
    # unpack DataFrame
    logM_star = input_df['logM_star'].values[0]
    ...
    model = LensModel(...)
    properties = lens_properties(model, beta_unit, logalpha_sps)

    scatter_mag = 0.1
    ...
    return properties
