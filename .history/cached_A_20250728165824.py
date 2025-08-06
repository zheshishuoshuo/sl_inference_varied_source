from functools import lru_cache


# === 安全取整，避免缓存 key 精度差异 ===
def safe_round(x, ndigits=4):
    return round(float(x), ndigits)

# === 缓存包装的 A_interp ===
@lru_cache(maxsize=512)
def cached_A_interp(mu_DM, sigma_DM):
    return A_interp((mu_DM, sigma_DM))