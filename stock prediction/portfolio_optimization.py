
# Import libraries
import numpy as np
import yfinance as yf
from scipy.optimize import minimize


# Portfolio optimization using Modern Portfolio Theory (MPT)
def get_optimal_portfolio(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    log_returns = np.log(data / data.shift(1))

    def minimize_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))

    num_assets = len(tickers)
    initial_weights = np.random.random(num_assets)
    initial_weights /= np.sum(initial_weights)

    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    optimal_weights = minimize(minimize_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimal_weights.x

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "FB"]
optimal_portfolio = get_optimal_portfolio(tickers, "2010-01-01", "2021-09-30")