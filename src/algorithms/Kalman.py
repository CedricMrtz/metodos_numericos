import numpy as np
import pandas as pd

def kalman(prices: pd.Series, process_var: float = 1e-5, measurement_var: float = 0.1) -> str:
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    n = len(prices)
    vals = prices.values
    x_est = np.zeros(n)   # estimación del estado
    p_est = np.zeros(n)   # estimación del error
 
    x_est[0] = vals[0]
    p_est[0] = 1.0
 
    for i in range(1, n):
        x_pred = x_est[i - 1]
        p_pred = p_est[i - 1] + process_var
 
        K = p_pred / (p_pred + measurement_var)
 
        x_est[i] = x_pred + K * (vals[i] - x_pred)
        p_est[i] = (1 - K) * p_pred
 
    smoothed = pd.Series(x_est, index=prices.index, name="Kalman")
 
    last_raw  = vals[-1]
    last_smoothed = x_est[-1]
    last_gain = K  # última ganancia
    noise = last_raw - last_smoothed
 
    lines = [
        f"\n{'='*50}",
        f"  Filtro de Kalman (1D)",
        f"{'='*50}",
        f"  Proceso σ²     : {process_var}",
        f"  Medición σ²    : {measurement_var}",
        f"  Precio raw     : {last_raw:.4f}",
        f"  Precio filtrado: {last_smoothed:.4f}",
        f"  Ruido estimado : {noise:+.4f}",
        f"  Última ganancia K: {last_gain:.6f}",
        f"\n  Últimos 5 días (raw vs filtrado):",
    ]
    tail = pd.DataFrame({"Raw": prices, "Kalman": smoothed}).tail(5)
    lines.append(tail.to_string())
    
    return "\n".join(lines), smoothed