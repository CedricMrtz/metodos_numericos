from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def arima(prices: pd.Series, order: tuple = (5, 1, 2), forecast_days: int = 10) -> str:
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
        f"  Último precio  : {last_price:.4f}  ({last_date.date()})",
        f"  AIC            : {result.aic:.2f}",
        f"  Tendencia pred : {trend}",
        f"\n  Predicción próximos {forecast_days} días:",
    ]
    for i, (date, val) in enumerate(forecast.items(), 1):
        chg = val - last_price
        lines.append(f"t+{i:02d} ({date.date()}): {val:.4f}  ({chg:+.4f})")
    
    print("\n".join(lines))
    return "\n".join(lines)