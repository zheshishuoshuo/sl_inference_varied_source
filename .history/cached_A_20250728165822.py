from functools import lru_cache

def safe_round(x, ndigits=4): ...
@lru_cache(maxsize=512)
def cached_A_interp(mu_DM, sigma_DM):
    from your_module import A_interp  # 延迟导入避免循环依赖
    return A_interp((mu_DM, sigma_DM))
