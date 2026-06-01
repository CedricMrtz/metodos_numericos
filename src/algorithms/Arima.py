from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def arima(prices: pd.Series, order: tuple = (5, 1, 2), forecast_days: int = 30) -> tuple:
    model   = ARIMA(prices, order=order)
    result  = model.fit()
    forecast = result.forecast(steps=forecast_days)
 
    last_price = prices.iloc[-1]
    last_date  = prices.index[-1]
    if forecast.iloc[-1] > last_price:
        trend = "alcista"
    else:
        trend = "bajista"
 
    lines = [
        f"\n{'='*50}",
        f"  ARIMA{order}",
        f"{'='*50}",
        f"  Ultimo precio  : {last_price:.4f}  (t={last_date})",
        f"  AIC            : {result.aic:.2f}",
        f"  Tendencia pred : {trend}",
        f"\n  Predicción próximos {forecast_days} días:",
    ]
    for i, val in enumerate(forecast.values, 1):
        chg = val - last_price
        lines.append(f"t+{i:02d} : {val:.4f}  ({chg:+.4f})")
    
    return "\n".join(lines), forecast.values