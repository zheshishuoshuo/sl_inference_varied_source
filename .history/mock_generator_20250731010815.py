import numpy as np
import pandas as pd
from lens_properties import observed_data
from tqdm import tqdm
from mass_sampler import generate_samples

# SPS PARAMETER
# M_star = alpha_sps * M_sps
# logM_star = log_alpha_sps + logM_sps


def run_mock_simulation(n_samples, mag_source=26.0, maximum_magnitude=26.5, zl=0.3, zs=2.0, if_source=False):
    beta_scalefree_samp = np.random.rand(n_samples)**0.5
    alpha_sps = np.random.normal(loc=1.2, scale=0.2, size=n_samples)
    logalpha_sps_sample = np.log10(alpha_sps)
    samples = generate_samples(n_samples)
    logM_star_sample = samples['logM_star_sps'] +logalpha_sps_sample

    lens_results = []
    for i in tqdm(range(n_samples), desc="Processing lenses"):
        input_df = pd.DataFrame({
            'logM_star_sps': [samples['logM_star_sps'][i]],
            'logM_star': [logM_star_sample[i]],
            'logM_halo': [samples['logMh'][i]],
            'logRe': [samples['logRe'][i]],
            'beta_unit': [beta_scalefree_samp[i]],
            'mag_source': [mag_source],
            'maximum_magnitude': [maximum_magnitude],
            'logalpha_sps': [logalpha_sps_sample[i]],
            'zl': [zl],
            'zs': [zs]
        })
        result = observed_data(input_df, caustic=False)
        lens_results.append(result)

    df_lens = pd.DataFrame(lens_results)
    mock_lens_data = df_lens[df_lens['is_lensed']]
    mock_observed_data = mock_lens_data[['xA', 'xB', '', 'logRe', 'magnitude_observedA', 'magnitude_observedB']].copy()
    if if_source:
        return df_lens, mock_lens_data, mock_observed_data
    else:
        return mock_lens_data, mock_observed_data
