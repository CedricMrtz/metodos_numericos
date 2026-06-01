import numpy as np

def cubic_spline_predict(prices: np.ndarray, future_days: int = 30) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    n = len(prices)

    MAX_KNOTS = 200
    if n > MAX_KNOTS:
        idx = np.round(np.linspace(0, n - 1, MAX_KNOTS)).astype(int)
        t_knots = idx.astype(float)
        y_knots = prices[idx]
    else:
        t_knots = np.arange(n, dtype=float)
        y_knots = prices.copy()

    m = len(t_knots)   # number of knots
    h = np.diff(t_knots)  # interval widths 

    A = np.zeros((m, m))
    r = np.zeros(m)

    A[0, 0] = 1.0   
    A[-1, -1] = 1.0        

    for i in range(1, m - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i]     = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        r[i] = 6.0 * ((y_knots[i + 1] - y_knots[i]) / h[i]
                     - (y_knots[i]     - y_knots[i - 1]) / h[i - 1])

    sigma = np.linalg.solve(A, r)  

    t_last = t_knots[-1]
    t_future = np.arange(t_last + 1, t_last + 1 + future_days, dtype=float)

    predictions = np.empty(future_days)

    for k, t_eval in enumerate(t_future):
        if t_eval >= t_knots[-1]:
            seg = m - 2       
        elif t_eval <= t_knots[0]:
            seg = 0
        else:
            seg = np.searchsorted(t_knots, t_eval, side="right") - 1
            seg = int(np.clip(seg, 0, m - 2))

        dx = t_eval - t_knots[seg]
        hi = h[seg]
        yi, yi1 = y_knots[seg], y_knots[seg + 1]
        si, si1 = sigma[seg], sigma[seg + 1]

        a = yi
        b = (yi1 - yi) / hi - hi * (2 * si + si1) / 6.0
        c = si / 2.0
        d = (si1 - si) / (6.0 * hi)

        predictions[k] = a + b * dx + c * dx**2 + d * dx**3

    return predictions

