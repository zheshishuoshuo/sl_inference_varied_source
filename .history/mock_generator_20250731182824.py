import numpy as np
import pandas as pd
from lens_properties import observed_data
from tqdm import tqdm
from mass_sampler import generate_samples

# SPS PARAMETER
# M_star = alpha_sps * M_sps
# logM_star = log_alpha_sps + logM_sps


# def run_mock_simulation(n_samples, mag_source=26.0, maximum_magnitude=26.5, zl=0.3, zs=2.0, if_source=False):
#     beta_scalefree_samp = np.random.rand(n_samples)**0.5
#     alpha_sps = np.random.normal(loc=1.2, scale=0.2, size=n_samples)
#     logalpha_sps_sample = np.log10(alpha_sps)
#     samples = generate_samples(n_samples)
#     logM_star_sample = samples['logM_star_sps'] +logalpha_sps_sample

#     lens_results = []
#     for i in tqdm(range(n_samples), desc="Processing lenses"):
#         input_df = pd.DataFrame({
#             'logM_star_sps': [samples['logM_star_sps'][i]],
#             'logM_star': [logM_star_sample[i]],
#             'logM_halo': [samples['logMh'][i]],
#             'logRe': [samples['logRe'][i]],
#             'beta_unit': [beta_scalefree_samp[i]],
#             'mag_source': [mag_source],
#             'maximum_magnitude': [maximum_magnitude],
#             'logalpha_sps': [logalpha_sps_sample[i]],
#             'zl': [zl],
#             'zs': [zs]
#         })
#         result = observed_data(input_df, caustic=False)
#         lens_results.append(result)

#     df_lens = pd.DataFrame(lens_results)
#     mock_lens_data = df_lens[df_lens['is_lensed']]
#     mock_observed_data = mock_lens_data[['xA', 'xB', 'logM_star_sps_observed', 'logRe', 'magnitude_observedA', 'magnitude_observedB']].copy()
#     if if_source:
#         return df_lens, mock_lens_data, mock_observed_data
#     else:
#         return mock_lens_data, mock_observed_data
    


# if __name__ == "__main__":
#     n_samples = 1000
#     mock_lens_data, mock_observed_data = run_mock_simulation(n_samples)
#     print(mock_lens_data.head())
#     print(mock_observed_data.head())

import numpy as np
import pandas as pd
from lens_properties import observed_data
from tqdm import tqdm
from mass_sampler import generate_samples
import multiprocessing


def simulate_single_lens(i, samples, beta_samp, logalpha_sps_sample, mag_source, maximum_magnitude, zl, zs):
    input_df = pd.DataFrame({
        'logM_star_sps': [samples['logM_star_sps'][i]],
        'logM_star': [samples['logM_star_sps'][i] + logalpha_sps_sample[i]],
        'logM_halo': [samples['logMh'][i]],
        'logRe': [samples['logRe'][i]],
        'beta_unit': [beta_samp[i]],
        'mag_source': [mag_source],
        'maximum_magnitude': [maximum_magnitude],
        'logalpha_sps': [logalpha_sps_sample[i]],
        'zl': [zl],
        'zs': [zs]
    })
    return observed_data(input_df, caustic=False)


def run_mock_simulation(n_samples, mag_source=26.0, maximum_magnitude=26.5,
                        zl=0.3, zs=2.0, if_source=False, process=None):
    beta_samp = np.random.rand(n_samples)**0.5
    alpha_sps = np.random.normal(loc=1.2, scale=0.2, size=n_samples)
    logalpha_sps_sample = np.log10(alpha_sps)
    samples = generate_samples(n_samples)

    if process is None or process == 0:
        # ===== 串行计算 =====
        lens_results = []
        for i in tqdm(range(n_samples), desc="Processing lenses"):
            input_df = pd.DataFrame({
                'logM_star_sps': [samples['logM_star_sps'][i]],
                'logM_star': [samples['logM_star_sps'][i] + logalpha_sps_sample[i]],
                'logM_halo': [samples['logMh'][i]],
                'logRe': [samples['logRe'][i]],
                'beta_unit': [beta_samp[i]],
                'mag_source': [mag_source],
                'maximum_magnitude': [maximum_magnitude],
                'logalpha_sps': [logalpha_sps_sample[i]],
                'zl': [zl],
                'zs': [zs]
            })
            result = observed_data(input_df, caustic=False)
            lens_results.append(result)

    else:
        # ===== 并行计算 =====
        args = [(i, samples, beta_samp, logalpha_sps_sample, mag_source,
                 maximum_magnitude, zl, zs) for i in range(n_samples)]

        with multiprocessing.get_context("spawn").Pool(process) as pool:
            lens_results = list(tqdm(
                pool.starmap(simulate_single_lens, args),
                total=n_samples, desc=f"Processing lenses (process={process})"
            ))

    df_lens = pd.DataFrame(lens_results)
    mock_lens_data = df_lens[df_lens['is_lensed']]
    mock_observed_data = mock_lens_data[[
        'xA', 'xB', 'logM_star_sps_observed', 'logRe',
        'magnitude_observedA', 'magnitude_observedB'
    ]].copy()

    if if_source:
        return df_lens, mock_lens_data, mock_observed_data
    else:
        return mock_lens_data, mock_observed_data

if __name__ == "__main__":
    # 串行
mock_lens_data, mock_observed_data = run_mock_simulation(1000, process=0)

# 默认行为（串行）
mock_lens_data, mock_observed_data = run_mock_simulation(1000)

# 并行，使用 8 核
mock_lens_data, mock_observed_data = run_mock_simulation(1000, process=8)
