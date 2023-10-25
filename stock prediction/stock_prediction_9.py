import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from hurst import compute_Hc
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


# Data Acquisition
def fetch_data(ticker, start_date, end_date):
    """ Fetch stock data using yfinance. """
    try:
        return yf.download(ticker, start=start_date, end=end_date)['Close']
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.Series()

# Preprocessing
def detrend_normalize(series):
    """ Detrend and normalize the data """
    trend = series.rolling(window=5).mean()
    detrended = series - trend
    normalized = (detrended - detrended.min()) / (detrended.max() - detrended.min())
    return normalized.dropna(), trend.dropna()


def time_delay_embedding(series, dimension, delay):
    """ Reconstruct the phase space using the time delay embedding method """
    vectors = []
    for i in range(len(series) - (dimension - 1) * delay):
        vector = [series[i + j * delay] for j in range(dimension)]
        vectors.append(vector)
    return np.array(vectors)

# Time-Series Analysis and Feature Engineering
def calculate_rolling_hurst(stock_data, window_size=252):
    """ Calculate the rolling Hurst exponent """
    hurst_values = []
    for i in range(len(stock_data) - window_size):
        H = compute_Hc(stock_data[i:i+window_size], kind='price', simplified=True)[0]
        hurst_values.append(H)
    return hurst_values

def lorenz_system(t, xyz, sigma, beta, rho):
    """ Lorenz System dynamics """
    x, y, z = xyz
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def get_initial_guess(phase_space):
    """ Generate an initial guess for Lorenz parameters based on phase space data. """
    means = np.mean(phase_space, axis=0)
    return [means[0], means[1], means[2], phase_space[0][0], phase_space[0][1], phase_space[0][2]]

def fit_to_lorenz(time, phase_space, p0):
    """ Fit the phase space data to the Lorenz system """
    def fit_function(t, sigma, beta, rho, x0, y0, z0):
        sol = solve_ivp(lorenz_system, (t[0], t[-1]), [x0, y0, z0], t_eval=t, args=(sigma, beta, rho))
        return sol.y[0]
    popt, _ = curve_fit(fit_function, time, phase_space[:, 0], p0=p0, method='trf')
    return popt

def predict_using_lorenz(t_future, popt):
    """ Predict future values using the Lorenz system """
    sigma, beta, rho, x0, y0, z0 = popt
    sol = solve_ivp(lorenz_system, (t_future[0], t_future[-1]), [x0, y0, z0], t_eval=t_future, args=(sigma, beta, rho))
    return sol.y[0]

def predict_using_clusters(series, clusters, next_n):
    """ Predict future values using the clustered phase space """
    dimension = 2
    delay = 1
    last_cluster = clusters[-1]
    next_points = []
    
    for _ in range(next_n):
        similar_clusters = np.where(clusters[:-1] == last_cluster)[0]
        if len(similar_clusters) == 0:
            break
        next_cluster = clusters[similar_clusters + 1]
        next_point = series[similar_clusters + (dimension - 1)].mean()
        next_points.append(next_point)
        last_cluster = np.random.choice(next_cluster)
    
    return next_points

def extrapolate_trend(series, next_n):
    """ Extrapolate the trend using linear regression for the next 'next_n' points. """
    trend = series.rolling(window=5).mean().dropna()
    x = np.arange(len(trend)).reshape(-1, 1)
    y = trend.values
    model = LinearRegression().fit(x, y)
    future_x = np.arange(len(trend), len(trend) + next_n).reshape(-1, 1)
    return model.predict(future_x)

def restore_scale_trend(original_series, trend, normalized_series, predicted_points):
    """ Restore the original scale and trend of the predicted values """
    scale = (original_series.max() - original_series.min()) / (normalized_series.max() - normalized_series.min())
    restored_points = [predicted_points[i] * scale + trend.values[i] for i in range(len(predicted_points))]
    return restored_points


def consolidate_predictions(lorenz_pred, cluster_pred, hurst_value):
    lorenz_weight = hurst_value
    return lorenz_pred * lorenz_weight + cluster_pred * (1 - lorenz_weight)

def visualize_combined_data(time, data, lorenz_predictions, cluster_predictions, combined_predictions, hurst_values, window_size):
    """ Visualize actual, Lorenz-predicted, cluster-predicted, and combined stock prices """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time, data.values, label='Actual Data', color='blue')
    ax.plot(time[-len(lorenz_predictions):], lorenz_predictions, label='Lorenz Predictions', linestyle='-.', color='orange')
    ax.plot(time[-len(cluster_predictions):], cluster_predictions, label='Cluster Predictions', linestyle=':', color='purple')
    ax.plot(time[-len(combined_predictions):], combined_predictions, label='Combined Predictions', linestyle='--', color='green')
    
    behavior_colors = ['blue', 'yellow', 'red']
    for i, H in enumerate(hurst_values):
        ax.axvspan(time[i], time[i+window_size], facecolor=behavior_colors[int(np.round(H))], alpha=0.1)
    # behaviors = ["Trending", "Random Walk", "Mean Reverting"]
    # ax.text(0.02, 0.85, f"Behavior Regions:\n{behaviors[0]}: H > 0.5\n{behaviors[1]}: H = 0.5\n{behaviors[2]}: H < 0.5", transform=ax.transAxes)
    
    ax.legend(loc="upper left")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    plt.title(f"Stock Price & Predictions with Behavior Regions")
    plt.show()

if __name__ == "__main__":
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2021-12-31'
    window_size = 252

    # Data Acquisition
    data = fetch_data(ticker, start_date, end_date)

    # Preprocessing
    normalized_data, trend_data = detrend_normalize(data)
    phase_space = time_delay_embedding(normalized_data, dimension=3, delay=1)
    
    # Time-Series Analysis
    hurst_values = calculate_rolling_hurst(data, window_size=window_size)

    # Lorenz System Prediction
    initial_guess = get_initial_guess(phase_space)
    popt = fit_to_lorenz(np.arange(len(phase_space)), phase_space, p0=initial_guess)
    lorenz_predictions = predict_using_lorenz(np.arange(len(data), len(data) + window_size), popt)
    
    # Clustering-based Prediction
    clustering_model = KMeans(n_clusters=5, n_init="auto")
    clusters = clustering_model.fit_predict(phase_space)
    cluster_predictions = predict_using_clusters(normalized_data, clusters, window_size)

    # Combining Predictions
    combined_predictions = []
    for lorenz_value, cluster_value, hurst_value in zip(lorenz_predictions, cluster_predictions, hurst_values):
        combined_predictions.append(consolidate_predictions(lorenz_value, cluster_value, hurst_value))
    
    # Restore Scale and Trend
    restored_lorenz_predictions = restore_scale_trend(data, trend_data, normalized_data, lorenz_predictions)
    restored_cluster_predictions = restore_scale_trend(data, trend_data, normalized_data, cluster_predictions)
    restored_combined_predictions = restore_scale_trend(data, trend_data, normalized_data, combined_predictions)

    # Visualization
    visualize_combined_data(data.index, data, restored_lorenz_predictions, restored_cluster_predictions, restored_combined_predictions, hurst_values, window_size)

