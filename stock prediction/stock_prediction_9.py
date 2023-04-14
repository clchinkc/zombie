

"""
Chaos theory is a branch of mathematics that focuses on the behavior of complex, nonlinear dynamical systems. It deals with systems that appear to be random or unpredictable, even though their underlying rules are deterministic. The study of chaos theory has applications in various disciplines, including physics, meteorology, engineering, biology, and economics.

In chaos theory, a small change in the initial conditions of a system can result in drastically different outcomes, a phenomenon known as sensitive dependence on initial conditions or the "butterfly effect." This sensitivity makes long-term predictions in chaotic systems difficult, as even the slightest variation in initial conditions can lead to vastly different outcomes.

Key concepts in chaos theory include:

Attractors: In a dynamical system, an attractor is a set of values towards which the system tends to evolve, regardless of the starting conditions. There are different types of attractors, such as fixed points, limit cycles, and strange attractors. Strange attractors are often associated with chaotic systems and have a fractal structure.

Fractals: Fractals are geometric shapes that exhibit self-similarity, meaning they look similar at different scales. They can be found in many natural phenomena, like coastlines, clouds, and snowflakes. Fractals are often used to model chaotic systems, as they can represent the complex behavior of these systems more effectively than traditional geometric shapes.

Bifurcation: Bifurcation is a term used in chaos theory to describe the process by which a small change in a system's parameters causes the system to undergo a qualitative or topological change in its behavior. Bifurcation diagrams are used to visualize the transitions between different types of behavior in a system, such as periodic, chaotic, or stable.

Lyapunov exponent: The Lyapunov exponent is a measure of the rate of separation of initially close trajectories in a dynamical system. A positive Lyapunov exponent indicates that the system is chaotic, as nearby trajectories diverge exponentially over time.

Applying chaos theory to stock prices can help in understanding the seemingly unpredictable and complex behavior of financial markets. While stock prices may appear random, chaos theory suggests that there could be underlying patterns or structures governing their movement. Here are some ways chaos theory can be applied to stock prices:

Identifying patterns: By analyzing stock price data through the lens of chaos theory, researchers can identify patterns or structures that might otherwise be hidden in the noise. These patterns could reveal relationships between different stocks, sectors, or economic indicators and help inform investment strategies.

Nonlinear dynamics: Financial markets are complex, nonlinear systems that can exhibit chaotic behavior. Using techniques from chaos theory, such as fractals and attractors, analysts can study the dynamics of stock prices and gain insights into their behavior. This can help investors identify periods of stability, volatility, or potential shifts in market trends.

Forecasting: While chaos theory implies that long-term predictions in chaotic systems are inherently difficult, it can still provide insights for short-term forecasts. By understanding the underlying dynamics of stock prices and their sensitivity to initial conditions, investors can make more informed decisions about market entry and exit points, and risk management.

Risk management: Chaos theory can help investors and portfolio managers assess the level of risk associated with their investments. By analyzing the behavior of stock prices in the context of chaos theory, they can better understand the potential for large, unpredictable fluctuations and make more informed decisions about diversification and risk management strategies.

Trading algorithms: Some quantitative traders use chaos theory to develop trading algorithms that exploit the patterns and structures found in stock price data. These algorithms may be able to identify short-term trading opportunities based on the chaotic behavior of financial markets.

It is important to note that while chaos theory can provide valuable insights into the behavior of stock prices, it does not guarantee success in predicting or profiting from market movements. Financial markets are influenced by various factors, such as economic conditions, investor sentiment, and global events, which can all contribute to the complexity and unpredictability of stock prices.
"""

"""
import numpy as np
import pandas as pd
import yfinance as yf
from hurst import compute_Hc


def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close']

def analyze_hurst_exponent(stock_data):
    H, c, data = compute_Hc(stock_data, kind='price', simplified=True)
    return H

if __name__ == "__main__":
    # Example: Apple Inc. stock (AAPL) from 2020-01-01 to 2021-09-01
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2021-09-01'
    
    stock_data = get_stock_data(ticker, start_date, end_date)
    hurst_exponent = analyze_hurst_exponent(stock_data)

    print(f"The Hurst exponent for {ticker} is {hurst_exponent}")

    if hurst_exponent < 0.5:
        print("The stock price exhibits anti-persistent behavior.")
    elif hurst_exponent == 0.5:
        print("The stock price exhibits a random walk.")
    else:
        print("The stock price exhibits persistent behavior.")

# If H < 0.5, the stock price exhibits anti-persistent behavior, meaning it tends to revert to the mean.
# If H = 0.5, the stock price exhibits a random walk, meaning it's difficult to predict future prices.
# If H > 0.5, the stock price exhibits persistent behavior, meaning it tends to trend in the same direction.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# Fetch historical stock price data
ticker = 'AAPL'
start_date = '2021-01-01'
end_date = '2021-12-31'

data = yf.download(ticker, start=start_date, end=end_date)
data['Normalized'] = data['Close'] / data['Close'].iloc[0]

print("Finished fetching data")

# Define the Lorenz system
def lorenz_system(t, xyz, sigma, beta, rho):
    x, y, z = xyz
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

# Function to fit the data
def fit_function(t, sigma, beta, rho, x0, y0, z0):
    sol = solve_ivp(lorenz_system, (t[0], t[-1]), [x0, y0, z0], t_eval=t, args=(sigma, beta, rho))
    return sol.y[0]

# Preprocess the data
time = np.linspace(0, 1, len(data))
normalized_prices = data['Normalized'].values

print("Finished preprocessing data")

# Fit the data to the Lorenz system
popt, _ = curve_fit(fit_function, time, normalized_prices, p0=(10, 8/3, 28, 1, 1, 1), method='trf')

print("Finished fitting data")

# Predict future stock prices
sigma, beta, rho, x0, y0, z0 = popt
t_future = np.linspace(0, 1.2, int(len(data) * 1.2))
predicted_prices = fit_function(t_future, sigma, beta, rho, x0, y0, z0)

print("Finished predicting data")

# Plot the results
plt.plot(time, normalized_prices, label='Historical Data')
plt.plot(t_future, predicted_prices, label='Predicted Data', linestyle='--')
plt.xlabel("Time")
plt.ylabel("Normalized Stock Price")
plt.legend()
plt.show()

print("Finished plotting data")

