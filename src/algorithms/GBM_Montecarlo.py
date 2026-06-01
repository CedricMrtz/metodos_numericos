import numpy as np
from typing import Optional

def gbm_montecarlo_predict(
    prices: np.ndarray,
    future_days: int = 30,
    simulations: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> dict:
    prices = np.asarray(prices, dtype=float)
 
    if seed is not None:
        np.random.seed(seed)

    log_returns = np.log(prices[1:] / prices[:-1])
 
    mu = np.mean(log_returns)
    sigma = np.std(log_returns, ddof=1)
 
    print(f"[gbm_montecarlo] μ (daily drift) = {mu:.6f}, "
          f"σ (daily vol) = {sigma:.6f}, "
          f"S₀ = {prices[-1]:.2f}")
 
    S0 = prices[-1]
    dt = 1.0 
 
    Z = np.random.standard_normal((simulations, future_days))
 
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
 
    log_paths = np.cumsum(increments, axis=1)    
    all_paths = S0 * np.exp(log_paths)
 
    mean_path  = np.mean(all_paths, axis=0)
 
    alpha = 1.0 - confidence
    lower_band = np.percentile(all_paths, 100 * alpha / 2, axis=0)
    upper_band = np.percentile(all_paths, 100 * (1 - alpha / 2), axis=0)
 
    return {
        "mean": mean_path,
        "lower": lower_band,
        "upper": upper_band,
        "all_paths": all_paths,
        "mu": mu,
        "sigma": sigma,
        "S0": S0,
    }
