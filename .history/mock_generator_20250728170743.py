import numpy as np
import pandas as pd
from lens_properties import observed_data
from tqdm import tqdm

import numpy as np

def generate_samples(n_samples, seed=None):
    """
    生成包含 logM_star, logMh, logRe 的模拟样本。

    参数：
    - n_samples: 样本数量
    - seed: 随机种子（可选）

    返回：
    - dict：包含 'logM_star', 'logMh', 'logRe' 三个 key，每个是 ndarray
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. logM_star
    logM_star = np.random.normal(loc=11.4, scale=0.3, size=n_samples)

    # 2. logRe（经验关系：随 stellar mass 增加而增大）
    logRe = np.random.normal(loc=1 + 0.8 * (logM_star - 11.4), scale=0.15)

    # 3. logMh（halo mass 与 SPS stellar mass 的经验关系）
    logMh = np.random.normal(loc=13.0 + 1.5 * (logM_star - 11.4), scale=0.2)

    return {
        "logM_star": logM_star,
        "logRe": logRe,
        "logMh": logMh
    }


def run_mock_simulation(n_samples, mag_source=26.0, maximum_magnitude=26.5, zl=0.3, zs=2.0):
    beta_scalefree_samp = np.random.rand(n_samples)**0.5
    alpha_sps = np.random.normal(loc=1.2, scale=0.2, size=n_samples)
    logalpha_sps_sample = np.log10(alpha_sps)
    samples = generate_samples(n_samples)

    lens_results = []
    for i in tqdm(range(n_samples), desc="Processing lenses"):
        input_df = pd.DataFrame({
            'logM_star': [samples['logM_star'][i]],
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
    mock_observed_data = mock_lens_data[['xA', 'xB', 'logMsps_observed', 'logRe', 'magnitude_observedA', 'magnitude_observedB']].copy()
    return mock_lens_data, mock_observed_data
