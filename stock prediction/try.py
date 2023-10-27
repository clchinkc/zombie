import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from hurst import compute_Hc
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Close', 'Open', 'High', 'Low', 'Volume']]
def detrend_normalize(data):
    trends = []
    detrended_data = np.zeros_like(data)
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        trend = np.polyfit(np.arange(len(data[:, i])), data[:, i], 1)[0] * np.arange(len(data[:, i]))
        trends.append(trend)
        detrended_data[:, i] = data[:, i] - trend
        normalized_data[:, i] = detrended_data[:, i] / np.linalg.norm(detrended_data[:, i])
    return normalized_data, trends
def time_delay_embedding(data, dimension, delay):
    N, M = data.shape  # N: time steps, M: number of features
    X = np.zeros((dimension * M, N - (dimension-1)*delay))
    for j in range(M):
        for i in range(dimension):
            X[j * dimension + i] = data[i*delay : i*delay + X.shape[1], j]
    return X
def extrapolate_trends(trends, next_n):
    future_trends = []
    for trend in trends:
        X = np.arange(len(trend)).reshape(-1, 1)
        y = trend.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        future_x = np.arange(len(trend), len(trend) + next_n).reshape(-1, 1)
        future_trend = reg.predict(future_x).reshape(-1)
        future_trends.append(future_trend)
    return np.array(future_trends).T
def denormalize_trend(data_normalized_detrended, trends):
    denormalized_data = np.zeros_like(data_normalized_detrended)
    for i in range(data_normalized_detrended.shape[1]):
        std_val = np.std(data_normalized_detrended[:, i])
        denormalized_data[:, i] = data_normalized_detrended[:, i] * std_val + trends[i]
    return denormalized_data
def predict_using_dmd_modes(X, DMD_modes, DMD_freqs, forecast_steps=50):
    """Predict future states using DMD modes and frequencies."""
    b = np.linalg.lstsq(DMD_modes, X[:, 0], rcond=None)[0]
    time_dynamics = np.zeros((DMD_modes.shape[1], forecast_steps))
    
    for t in range(forecast_steps):
        time_dynamics[:, t] = np.exp(DMD_freqs * t) * b
    
    predictions = np.abs(DMD_modes @ time_dynamics)
    return predictions
def lorenz_system(t, xyz, sigma, beta, rho):
    """ Lorenz System dynamics """
    x, y, z = xyz
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
def fit_to_lorenz(embedded_data, forecast_steps):
    num_features = embedded_data.shape[0] // 3  # Assuming embedding dimension of 3
    all_predictions = []

    for i in range(num_features):
        feature_embedded_data = embedded_data[i*3:i*3+3, :]
        initial_condition = feature_embedded_data[:, -1]

        def cost_function(params):
            sol = solve_ivp(lorenz_system, [0, forecast_steps], initial_condition, args=params, t_eval=np.arange(forecast_steps))
            return (sol.y - feature_embedded_data[:, :forecast_steps]).flatten()

        res = least_squares(cost_function, x0=[10, 28, 8/3])
        sol = solve_ivp(lorenz_system, [0, forecast_steps], initial_condition, args=res.x, t_eval=np.arange(forecast_steps))
        all_predictions.append(sol.y[0])

    return np.array(all_predictions).T
def predict_using_clusters(embedded_data, forecast_steps):
    num_features = embedded_data.shape[0]
    all_predictions = []

    for i in range(num_features):
        feature_embedded_data = embedded_data[i:i+1, :].T
        kmeans = KMeans(n_clusters=3, n_init="auto").fit(feature_embedded_data)
        cluster_centers = kmeans.cluster_centers_

        predictions = [feature_embedded_data[-1, :]]
        for _ in range(forecast_steps - 1):
            distances = np.linalg.norm(cluster_centers - predictions[-1].reshape(-1, 1), axis=1)
            predictions.append(cluster_centers[np.argmin(distances)])
        
        all_predictions.append(np.array(predictions)[:, 0])

    return np.array(all_predictions).T
def consolidate_predictions(lorenz_pred, cluster_pred, hurst_value):
    num_features = lorenz_pred.shape[1]
    combined = np.zeros_like(lorenz_pred)

    for i in range(num_features):
        combined[:, i] = lorenz_pred[:, i] * hurst_value + cluster_pred[:, i] * (1 - hurst_value)
    
    return combined
def visualize_combined_data_multi(data, dmd_predictions, lorenz_predictions, cluster_predictions, combined_predictions, trends, hurst_values, window_size):
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    feature_names = ['Close', 'Open', 'High', 'Low', 'Volume']

    for i, ax in enumerate(axes):
        ax.plot(data[:, i], label='Actual ' + feature_names[i], color='black')
        ax.plot(np.arange(len(data), len(data) + len(dmd_predictions)), dmd_predictions[:, i], label='DMD Predictions', color='red')
        ax.plot(np.arange(len(data), len(data) + len(lorenz_predictions)), lorenz_predictions[:, i], label='Lorenz Predictions', color='blue')
        ax.plot(np.arange(len(data), len(data) + len(cluster_predictions)), cluster_predictions[:, i], label='Cluster Predictions', color='green')
        ax.plot(np.arange(len(data), len(data) + len(combined_predictions)), combined_predictions[:, i], label='Combined Predictions', color='purple')

        ax.set_ylabel(feature_names[i])
        ax.legend(loc='upper left')
        ax.grid(True)
        ax.set_xlabel('Time')
    plt.tight_layout()
    plt.show()

# Main section of the code
ticker = "AAPL"
start_date = "2010-01-01"
end_date = "2020-12-31"

data = fetch_data(ticker, start_date, end_date)
normalized_data, trend = detrend_normalize(data)
hurst_values = calculate_rolling_hurst(normalized_data)
latest_hurst_value = hurst_values[-1]
embedding_dimension = 3
delay = 1
embedded_data = time_delay_embedding(normalized_data, embedding_dimension, delay)

forecast_steps = 50

X, X_prime = construct_data_matrices(embedded_data)
A_tilde = compute_dmd_matrix(X)
DMD_modes, DMD_freqs = compute_dmd_modes_and_freqs(X, A_tilde)
dmd_pred = predict_using_dmd_modes(X, DMD_modes, DMD_freqs, forecast_steps=forecast_steps)
lorenz_pred = fit_to_lorenz(embedded_data, forecast_steps)
cluster_pred = predict_using_clusters(embedded_data, forecast_steps)

final_prediction = consolidate_predictions(lorenz_pred, cluster_pred, latest_hurst_value)

extrapolated_trend = extrapolate_trend(trend, forecast_steps)
denormalized_prediction = denormalize_trend(final_prediction, data[:, 0].min(), data[:, 0].max(), extrapolated_trend)

print(f"Final Prediction: {denormalized_prediction}")

plt.figure(figsize=(14, 7))
plt.plot(data[:, 0], label="Actual 'Close' Prices")
plt.plot(np.arange(len(data), len(data) + forecast_steps), denormalized_prediction, label="Predicted 'Close' Prices", linestyle='--')
plt.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.grid(True)
plt.show()
