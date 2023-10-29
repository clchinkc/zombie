import datetime
from re import L

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# 1. Data Loading and Preprocessing
def load_data(filename):
    return pd.read_csv(filename, parse_dates=['Date'], index_col='Date')

def generate_features(data):
    data['PrevOpen'] = data['Open'].shift(1, fill_value=np.mean(data['Open']))
    data['PrevHigh'] = data['High'].shift(1, fill_value=np.mean(data['High']))
    data['PrevLow'] = data['Low'].shift(1, fill_value=np.mean(data['Low']))
    data['PrevClose'] = data['Close'].shift(1, fill_value=np.mean(data['Close']))
    data['MovingAverage'] = data['Close'].rolling(window=7, min_periods=1).mean()
    data['RSI'] = compute_rsi(data, 14)
    data['MACD'], data['Signal_Line'], data['MACD_Histogram'] = compute_macd(data)
    data['Upper_Bollinger_Band'], data['Lower_Bollinger_Band'] = compute_bollinger_bands(data)
    return data.dropna()

def compute_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def compute_macd(data):
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def compute_bollinger_bands(data):
    sma = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    upper_band = sma + (rolling_std * 2)
    lower_band = sma - (rolling_std * 2)
    return upper_band, lower_band

# 3. Splitting and Normalization
def split_data(data, test_size=0.3):
    return train_test_split(data, test_size=test_size, shuffle=False)

def normalize_data(data, min_vals=None, max_vals=None):
    if min_vals is None or max_vals is None:
        min_vals = data.min()
        max_vals = data.max()
    return (data - min_vals) / (max_vals - min_vals), min_vals, max_vals

def perform_grey_relational_analysis(train_data, features):
    def grey_relational_coefficient(reference, comparison, rho=0.5):
        abs_diff = np.abs(reference - comparison)
        max_diff = np.max(abs_diff)
        min_diff = np.min(abs_diff)
        return (min_diff + rho * max_diff) / (abs_diff + rho * max_diff + 1e-10)

    normalized_reference = train_data['Close']
    normalized_comparisons = [train_data[feature] for feature in features]
    coefficients = [grey_relational_coefficient(normalized_reference, seq) for seq in normalized_comparisons]
    weights = np.mean(coefficients, axis=1)
    aggregated_sequence = np.average(normalized_comparisons, axis=0, weights=weights)
    return aggregated_sequence

def create_windowed_data_for_features(data, features, window_size=5):
    X = []
    y = []
    
    for i in range(len(data) - window_size + 1):
        x_window = []
        for feature in features:
            x_window.extend(data[feature].iloc[i:i+window_size].values)
        X.append(x_window)
        y.append(data['Close'].iloc[i + window_size - 1])

    return np.array(X), np.array(y)

def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model

def predict_using_model(model, test, features, window_size=5, num_future_predictions=10):
    feature_windows = {feature: list(test[feature].iloc[-window_size:].values) for feature in features}
    predictions = []

    total_length = len(test) + num_future_predictions
    for i in range(total_length):
        x_window = []
        for feature in features:
            x_window.extend(feature_windows[feature])
        
        prediction = model.predict([x_window])
        predictions.append(prediction[0])

        for feature in features:
            feature_windows[feature].pop(0)
            feature_windows[feature].append(prediction[0])

    return predictions


def denormalize(data, min_vals, max_vals):
    data_np = np.array(data)
    min_vals_np = np.array(min_vals)
    max_vals_np = np.array(max_vals)
    return data_np * (max_vals_np - min_vals_np) + min_vals_np

def visualize_results(train, test, future_dates=[], **prediction_sets):
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train['Close'], label='Training Close Prices', color='blue')
    plt.plot(test.index, test['Close'], label='Actual Test Close Prices', color='green')
    
    for label, preds in prediction_sets.items():
        # Split the predictions into test and future predictions based on lengths
        test_preds = preds[:len(test)]
        future_preds = preds[len(test):]

        # Plot test predictions
        plt.plot(test.index, test_preds, label=f'{label} Test Predictions', color='orange')

        # If there are future predictions, plot them with a distinct color
        if future_dates:
            plt.plot(future_dates, future_preds, label=f'{label} Future Predictions', linestyle='--', color='red')
    
    plt.title('Stock Price Prediction Comparison')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.tight_layout()
    plt.show()



# Example usage:
data = load_data('apple_stock_data.csv')
data = generate_features(data)
train, test = split_data(data)
normalized_train, train_min, train_max = normalize_data(train)
normalized_test, _, _ = normalize_data(test, train_min, train_max)

features = ['PrevOpen', 'PrevHigh', 'PrevLow', 'PrevClose', 'MovingAverage']
X_train, y_train = create_windowed_data_for_features(normalized_train, features, window_size=7)
X_test, y_test = create_windowed_data_for_features(normalized_test, features, window_size=7)

# model = LinearRegression()
# model = Lasso(alpha=0.1)
# model = RandomForestRegressor(n_estimators=100)
model = GradientBoostingRegressor(n_estimators=100)

model = train_model(X_train, y_train, model)
predictions = predict_using_model(model, normalized_test, features, window_size=7, num_future_predictions=5)
predictions = denormalize(predictions, train_min['Close'], train_max['Close'])

# Generate future dates
num_dates = 7
next_date = data.index[-1] + datetime.timedelta(days=1)
future_dates = [next_date + datetime.timedelta(days=i) for i in range(num_dates)]

visualize_results(train, test, future_dates=future_dates, model=predictions)