from functools import lru_cache





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



# === 安全取整，避免缓存 key 精度差异 ===
def safe_round(x, ndigits=4):
    return round(float(x), ndigits)

# === 缓存包装的 A_interp ===
@lru_cache(maxsize=512)
def cached_A_interp(mu_DM, sigma_DM):
    return A_interp((mu_DM, sigma_DM))