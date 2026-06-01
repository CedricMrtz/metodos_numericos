import pandas as pd

def sma_ema(prices: pd.Series, period: int = 20) -> tuple:
    sma = prices.rolling(window=period).mean()
    ema = prices.ewm(span=period, adjust=False).mean()
 
    last_price = prices.iloc[-1]
    last_sma = sma.iloc[-1]
    last_ema = ema.iloc[-1]
 
    if last_price > last_sma:
        signal_sma = "COMPRA FUERTE"
    else:
        signal_sma = "VENTA FUERTE" 

    if last_price > last_ema:
        signal_ema = "COMPRA FUERTE"
    else:
        signal_ema = "VENTA FUERTE"
 
    lines = [
        f"\n{'='*50}",
        f"  SMA / EMA  (periodo = {period})",
        f"{'='*50}",
        f"  Precio actual : {last_price:.4f}",
        f"  SMA({period}) : {last_sma:.4f}  → señal {signal_sma}",
        f"  EMA({period}) : {last_ema:.4f}  → señal {signal_ema}",
        f"\n  Últimos 5 días:",
    ]
    tail = pd.DataFrame({"Precio": prices, f"SMA_{period}": sma, f"EMA_{period}": ema}).tail(5)
    lines.append(tail.to_string())

    return "\n".join(lines), sma.values, ema.values