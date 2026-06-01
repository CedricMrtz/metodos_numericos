import numpy as np

def linear_regression_predict(prices, future_days=30):
    y = np.array(prices, dtype=float)

    x = np.arange(len(y))

    X = np.column_stack([
        np.ones(len(x)),
        x
    ])

    beta = np.linalg.inv(X.T @ X) @ X.T @ y

    intercept = beta[0]
    slope = beta[1]

    future_x = np.arange(
        len(y),
        len(y) + future_days
    )

    predictions = (
        intercept +
        slope * future_x
    )

    return predictions
