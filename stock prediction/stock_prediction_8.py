
# Data acquisition: Import necessary libraries and fetch historical stock price data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
from alpha_vantage.timeseries import TimeSeries

# Replace 'YOUR_API_KEY' with your Alpha Vantage API key
api_key = 'YOUR_API_KEY'
symbol = 'MSFT' # Microsoft stock

ts = TimeSeries(key=api_key, output_format='pandas')
data, _ = ts.get_daily_adjusted(symbol, outputsize='full')
data = data['5. adjusted close'].sort_index()
"""

data = pd.read_csv('apple_stock_data.csv', index_col='Date', parse_dates=True)['Close']


# Preprocessing: Detrend and normalize the data.
def detrend_normalize(series):
    detrended = series - series.rolling(window=5).mean()
    normalized = (detrended - detrended.min()) / (detrended.max() - detrended.min())
    return normalized

normalized_data = detrend_normalize(data)
normalized_data = normalized_data.dropna()

# Phase space reconstruction: Reconstruct the phase space using the time delay embedding method.
def time_delay_embedding(series, dimension, delay):
    vectors = []
    for i in range(len(series) - (dimension - 1) * delay):
        vector = [series[i + j * delay] for j in range(dimension)]
        vectors.append(vector)
    return np.array(vectors)

dimension = 2
delay = 1
phase_space = time_delay_embedding(normalized_data, dimension, delay)



# Clustering: Cluster the phase space vectors using a clustering algorithm.
from sklearn.cluster import KMeans

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, n_init=100, max_iter=1000)
clusters = kmeans.fit_predict(phase_space)



# Prediction: Use the clustered phase space to make predictions.
def predict(series, clusters, next_n):
    last_cluster = clusters[-1]
    next_points = []

    for _ in range(next_n):
        similar_clusters = np.where(clusters[:-1] == last_cluster)[0]
        next_cluster = clusters[similar_clusters + 1]
        next_point = series[similar_clusters + (dimension - 1)].mean()
        next_points.append(next_point)
        last_cluster = next_cluster[np.argmax(series[similar_clusters + (dimension - 1)])]

    return next_points


future_days = 14
predicted_points = predict(normalized_data, clusters, future_days)



# Postprocessing: Restore the original scale and trend to the predicted points.
def restore_scale_trend(series, normalized_series, predicted_points):
    scale = (series.max() - series.min()) / (normalized_series.max() - normalized_series.min())
    trend = series.rolling(window=5).mean()
    restored_points = [predicted_points[i] * scale + trend[-i] for i in range(len(predicted_points))]
    return restored_points

restored_points = restore_scale_trend(data, normalized_data, predicted_points)



# Visualization: Plot the actual and predicted stock prices.
plt.figure(figsize=(14, 6))
plt.plot(data, label='Actual Price')
plt.plot(pd.Series(index=pd.date_range(data.index[-1], periods=future_days + 1, inclusive='right'), data=restored_points), label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
symbol = 'MSFT' # Microsoft stock
plt.title(f'{symbol} Stock Price Prediction Using Chaos Theory')
plt.legend()
plt.show()




"""
More sophisticated techniques for stock price prediction using chaos theory can involve a combination of advanced methods for data preprocessing, feature extraction, model selection, and post-processing. Some of these techniques include:

Advanced denoising: Instead of simple detrending and normalization, more advanced denoising techniques, such as wavelet denoising or empirical mode decomposition, can be used to better isolate the underlying patterns in the stock price data.

Optimal parameter selection: For time delay embedding, selecting the appropriate dimension and delay is critical. Techniques like the false nearest neighbors (FNN) method and the mutual information method can help in finding optimal values for these parameters.

Nonlinear feature extraction: Extracting relevant features from the reconstructed phase space can be crucial for accurate predictions. Some possible methods include Lyapunov exponents, correlation dimension, and recurrence quantification analysis.

Ensemble methods and hybrid models: Instead of relying on a single model or clustering algorithm, ensemble methods can be employed, such as bagging and boosting, to improve prediction accuracy. Furthermore, hybrid models that combine chaos theory with other techniques like machine learning, deep learning, or statistical models can potentially yield better results.

Model validation and selection: Rigorous validation methods, such as cross-validation or walk-forward validation, can be used to assess the performance of different models and select the best one for prediction. Additionally, performance metrics like mean absolute error (MAE), mean squared error (MSE), and R-squared can help in evaluating and comparing models.

Post-processing and risk management: Techniques like Kalman filtering or Bayesian estimation can be used to refine the predictions and manage uncertainties. Incorporating a risk management framework that accounts for the chaotic nature of stock prices can help in managing investment risks.

Adapting to changing market conditions: Since financial markets are dynamic and evolving, it's essential to continuously monitor and update the models to adapt to changing market conditions. This can involve updating parameters, retraining the model, or even changing the model altogether.
"""