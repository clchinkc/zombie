import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from hurst import compute_Hc
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import svd

# 1. Data Loading and Preprocessing

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].values

def detrend_normalize(data):
    trend = np.polyfit(np.arange(len(data)), data, 1)[0] * np.arange(len(data))
    detrended = data - trend
    normalized = detrended / np.linalg.norm(detrended)
    return normalized, trend

def time_delay_embedding(data, delay, dimension):
    N = len(data)
    X = np.zeros((dimension, N - (dimension-1)*delay))
    for i in range(dimension):
        X[i] = data[i*delay : i*delay + X.shape[1]]
    return X

# 2. Dynamic Mode Decomposition (DMD) Analysis

def construct_data_matrix(feature_matrix, k=2):
    rows, cols = feature_matrix.shape
    X = np.zeros(((k * cols), (rows - k + 1)))
    for i in range(rows - k + 1):
        X[:, i] = feature_matrix[i:i+k].flatten()
    return X

def compute_dmd_matrix(X):
    U, Sigma, Vh = svd(X, full_matrices=False)
    S_inv = np.diag(1 / Sigma)
    A_tilde = U.T @ X @ Vh.T @ S_inv
    return A_tilde

def find_dominant_modes(eigenvalues, eigenvectors, num_modes):
    dominant_indices = np.argsort(np.abs(eigenvalues))[-num_modes:]
    dominant_eigenvalues = eigenvalues[dominant_indices]
    dominant_eigenvectors = eigenvectors[:, dominant_indices]
    return dominant_eigenvalues, dominant_eigenvectors

def predict_future_prices(A_tilde, X, forecast_steps=50):
    state = X[:, -1].reshape(-1, 1)  # Ensure state is a column vector
    
    # Initialize the future_prices list with the last known price
    future_prices = [state[0][0]]

    for _ in range(forecast_steps - 1):  # -1 since we already have the last known price
        state = A_tilde @ state
        future_prices.append(state[0][0])  # Extracting the future price from the state

    return np.array(future_prices)


# 3. Advanced Time-Series Analysis

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

def fit_to_lorenz(embedded_data, forecast_steps):
    data_len = embedded_data.shape[1]
    initial_condition = embedded_data[:, -1]  # Use the last known point as the initial condition

    def cost_function(params):
        sol = solve_ivp(lorenz_system, [0, data_len + forecast_steps], initial_condition, args=params, t_eval=np.arange(data_len + forecast_steps))
        return (sol.y[:, -forecast_steps:] - embedded_data[:, -forecast_steps:]).flatten()

    res = least_squares(cost_function, x0=[10, 28, 8/3])

    # Once we have the best fit parameters, we'll forecast the next values
    sol = solve_ivp(lorenz_system, [0, forecast_steps], initial_condition, args=res.x, t_eval=np.arange(forecast_steps))
    return sol.y[0]  # Return x-values from Lorenz system


def predict_using_clusters(embedded_data, forecast_steps):
    kmeans = KMeans(n_clusters=3, n_init="auto").fit(embedded_data.T)
    cluster_centers = kmeans.cluster_centers_
    
    predictions = [embedded_data[:, -1]]
    for _ in range(forecast_steps - 1):  
        distances = np.linalg.norm(cluster_centers - predictions[-1].reshape(-1, 1), axis=1)
        predictions.append(cluster_centers[np.argmin(distances)])
    return np.array(predictions).T[0]  # Return the x-values from the cluster centers


def reverse_preprocess(data_normalized_detrended, min_val, max_val, trend):
    data_detrended = data_normalized_detrended * (max_val - min_val) + min_val
    original_data = data_detrended + trend
    return original_data

def consolidate_predictions(lorenz_pred, cluster_pred, hurst_value):
    lorenz_weight = hurst_value
    return lorenz_pred * lorenz_weight + cluster_pred * (1 - lorenz_weight)

# 4. Visualization

def visualize_combined_data(data, dmd_predictions, lorenz_predictions, cluster_predictions, combined_predictions, trend, hurst_values, window_size):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    time = np.arange(len(data) - len(hurst_values))
    
    # Assuming Hurst values can range between 0 and 1
    for i, H in enumerate(hurst_values):
        if i + window_size < len(data):
            color = plt.cm.RdYlGn(H)  # Get color based on Hurst value
            ax.axvspan(i, i + window_size, facecolor=color, alpha=0.1)
    
    ax.plot(data, label='Actual Stock Prices', color='black')
    ax.plot(np.arange(len(data), len(data) + len(dmd_predictions)), dmd_predictions, label='DMD Predictions', color='red')
    ax.plot(np.arange(len(data), len(data) + len(lorenz_predictions)), lorenz_predictions, label='Lorenz Predictions', color='blue')
    ax.plot(np.arange(len(data), len(data) + len(cluster_predictions)), cluster_predictions, label='Cluster Predictions', color='green')
    ax.plot(np.arange(len(data), len(data) + len(combined_predictions)), combined_predictions, label='Combined Predictions', color='purple')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.set_title('Stock Price Predictions using DMD, Lorenz System, and Clustering')
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    # Data acquisition and preprocessing
    stock_data = fetch_data('AAPL', '2020-01-01', '2022-12-31')
    min_val = np.min(stock_data)
    max_val = np.max(stock_data)
    normalized_data, trend = detrend_normalize(stock_data)
    embedded_data = time_delay_embedding(normalized_data, 1, 3)
    data_matrix = construct_data_matrix(embedded_data.T, k=2)
    
    window_size = 252
    forecast_steps = 30

    # Advanced Time-Series Analysis
    hurst_values = calculate_rolling_hurst(stock_data, window_size=window_size)

    # DMD Analysis
    A_tilde = compute_dmd_matrix(data_matrix)
    dmd_predictions = predict_future_prices(A_tilde, data_matrix, forecast_steps)
    dmd_predictions = reverse_preprocess(dmd_predictions, min_val, max_val, trend[-forecast_steps:])
    
    # Lorenz System Prediction
    lorenz_predictions = fit_to_lorenz(embedded_data, forecast_steps)
    lorenz_predictions = reverse_preprocess(lorenz_predictions, min_val, max_val, trend[-forecast_steps:])

    # Clustering-based Prediction
    cluster_predictions = predict_using_clusters(embedded_data, forecast_steps)
    cluster_predictions = reverse_preprocess(cluster_predictions, min_val, max_val, trend[-forecast_steps:])

    # Consolidate Predictions
    combined_predictions = consolidate_predictions(lorenz_predictions, cluster_predictions, hurst_values[-1])

    # Visualization
    visualize_combined_data(stock_data, dmd_predictions, lorenz_predictions, cluster_predictions, combined_predictions, trend, hurst_values, window_size)


