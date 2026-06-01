import pandas as pd

def extract_stock(input_csv, ticker, output_csv=None):
    df = pd.read_csv(input_csv)

    stock_df = df[df["Ticker"] == ticker].copy()

    if output_csv:
        stock_df.to_csv(output_csv, index=False)

    return stock_df
