import numpy as np
from algorithms.linealRegression import linear_regression_predict
from algorithms.CubicSplineInterpolation import cubic_spline_predict
from algorithms.GBM_Montecarlo import gbm_montecarlo_predict
from utils.extract_stock import extract_stock


DEFAULT_CSV = "data/SP500_Historical_Data.csv"
DEFAULT_TICKER = "AAPL"
DEFAULT_OUTPUT = "processed_data/AAPL.csv"
DEFAULT_FUTURE = 30
DEFAULT_SIMULATIONS = 1000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 42


def run_on_prices(prices, future_days=30, simulations=1000, confidence=0.95, seed=42):
    print("=" * 60)
    print("1. LINEAR REGRESSION")
    lr_pred = linear_regression_predict(prices, future_days=future_days)
    print(f"   Day+1  = ${lr_pred[0]:.2f}   Day+{future_days} = ${lr_pred[-1]:.2f}")

    print("=" * 60)
    print("2. CUBIC SPLINE")
    cs_pred = cubic_spline_predict(prices, future_days=future_days)
    print(f"   Day+1  = ${cs_pred[0]:.2f}   Day+{future_days} = ${cs_pred[-1]:.2f}")

    print("=" * 60)
    print("3. GBM + MONTE CARLO")
    gbm = gbm_montecarlo_predict(
        prices,
        future_days=future_days,
        simulations=simulations,
        confidence=confidence,
        seed=seed,
    )
    print(f"   Mean Day+1  = ${gbm['mean'][0]:.2f}   Day+{future_days} = ${gbm['mean'][-1]:.2f}")
    print(f"   {int(confidence*100)}% CI Day+{future_days}: [${gbm['lower'][-1]:.2f}, ${gbm['upper'][-1]:.2f}]")
    print(f"   Paths shape: {gbm['all_paths'].shape}")


def main():
    csv_path = DEFAULT_CSV
    ticker = DEFAULT_TICKER
    output = DEFAULT_OUTPUT
    future = DEFAULT_FUTURE
    simulations = DEFAULT_SIMULATIONS
    confidence = DEFAULT_CONFIDENCE
    seed = DEFAULT_SEED

    if csv_path:
        try:
            df = extract_stock(csv_path, ticker, output)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return

        if df.empty:
            print(f"No rows found for ticker {ticker} in {csv_path}")
            return

        if "Close" not in df.columns:
            print("CSV does not contain a 'Close' column needed for prices")
            return

        prices = df["Close"].values
        run_on_prices(prices, future_days=future, simulations=simulations, confidence=confidence, seed=seed)
    else:
        np.random.seed(0)
        n_hist = 252   # one trading year
        synthetic_prices = 150.0 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n_hist))
        run_on_prices(synthetic_prices, future_days=future, simulations=simulations, confidence=confidence, seed=seed)


if __name__ == "__main__":
    main()
