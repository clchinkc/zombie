import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(filename):
    data = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
    
    # Generate additional features
    data['PrevOpen'] = data['Open'].shift(1, fill_value=np.mean(data['Open']))
    data['PrevHigh'] = data['High'].shift(1, fill_value=np.mean(data['High']))
    data['PrevLow'] = data['Low'].shift(1, fill_value=np.mean(data['Low']))
    data['PrevClose'] = data['Close'].shift(1, fill_value=np.mean(data['Close']))
    data['MovingAverage'] = data['Close'].rolling(window=7, min_periods=1).mean()
    
    # Relative Strength Index
    data['RSI'] = compute_rsi(data, 14) # not used
    
    # Moving Average Convergence Divergence
    data['MACD'], data['Signal_Line'], data['MACD_Histogram'] = compute_macd(data) # not used
    
    # Bollinger Bands
    data['Upper_Bollinger_Band'], data['Lower_Bollinger_Band'] = compute_bollinger_bands(data) # not used
    
    processed_data = data.dropna()
    return processed_data

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

def split_and_normalize_data(data):
    train_data, test_data = train_test_split(data, test_size=0.3, shuffle=False)
    
    min_vals = train_data.min()
    max_vals = train_data.max()
    
    train_data = (train_data - min_vals) / (max_vals - min_vals)
    test_data = (test_data - min_vals) / (max_vals - min_vals)
    
    return train_data, test_data, min_vals, max_vals

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

def create_windowed_data(data, window_size=5):
    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:i+window_size])
    return np.array(X)

def train_and_predict(train, test, window_size=5):
    # Generate windowed training data
    X_train = create_windowed_data(train['GRA'].values, window_size)
    y_train = train['Close'].iloc[window_size-1:]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = []

    # Use last `window_size` values from train data as the initial window
    window = list(train['GRA'].iloc[-window_size:].values)
    
    for i in range(len(test)):
        # Use the window to predict the next value
        prediction = model.predict([window])
        predictions.append(prediction[0])
        
        # Move the window: drop the oldest value and append the prediction
        window.pop(0)
        window.append(prediction[0])

    return predictions

def evaluate_predictions(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def visualize_results(train, test, predictions):
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train['Close'], label='Training Close Prices', color='blue')
    plt.plot(test.index, test['Close'], label='Actual Test Close Prices', color='green')
    plt.plot(test.index, predictions, label='Predicted Test Close Prices', linestyle='--', color='red')
    plt.fill_between(test.index, test['Close'], predictions, color='grey', alpha=0.5)
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

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

def train_and_predict_using_features(train, test, features, regression_type='linear', window_size=5, alpha=1.0):
    X_train, y_train = create_windowed_data_for_features(train, features, window_size)
    
    if regression_type == 'linear':
        model = LinearRegression()
    elif regression_type == 'lasso':
        model = Lasso(alpha=alpha)
    elif regression_type == 'forest':
        model = RandomForestRegressor(n_estimators=100)
    else:
        raise ValueError(f"Unsupported regression type: {regression_type}")
        
    model.fit(X_train, y_train)
    
    feature_windows = {feature: list(train[feature].iloc[-window_size:].values) for feature in features}
    predictions = []
    for i in range(len(test)):
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
    return data * (max_vals - min_vals) + min_vals

data = load_and_preprocess_data('apple_stock_data.csv')
train, test, min_vals, max_vals = split_and_normalize_data(data)

# Train and predict using GRA
features = ['PrevOpen', 'PrevHigh', 'PrevLow', 'PrevClose', 'MovingAverage']
train['GRA'] = perform_grey_relational_analysis(train, features)
predictions_GRA = train_and_predict(train, test, window_size=7)
predictions_GRA = denormalize(np.array(predictions_GRA), min_vals['Close'], max_vals['Close'])

# Train and predict using Lasso regression
lasso_predictions = train_and_predict_using_features(train, test, features, regression_type='lasso', alpha=1.0, window_size=7)
lasso_predictions = denormalize(np.array(lasso_predictions), min_vals['Close'], max_vals['Close'])

# Train and predict using Random Forest regression
rf_predictions = train_and_predict_using_features(train, test, features, regression_type='forest', window_size=7)
rf_predictions = denormalize(np.array(rf_predictions), min_vals['Close'], max_vals['Close'])

denormalized_train = denormalize(train['Close'].values, min_vals['Close'], max_vals['Close'])
denormalized_test = denormalize(test['Close'].values, min_vals['Close'], max_vals['Close'])

mse_GRA = evaluate_predictions(denormalized_test, predictions_GRA)
lasso_mse = evaluate_predictions(denormalized_test, lasso_predictions)
rf_mse = evaluate_predictions(denormalized_test, rf_predictions)

print(f'GRA MSE: {mse_GRA:.2f}')
print(f'Lasso MSE: {lasso_mse:.2f}')
print(f'Random Forest MSE: {rf_mse:.2f}')

# Visualize the results
plt.figure(figsize=(12,6))
plt.plot(train.index, denormalized_train, label='Training Close Prices', color='blue')
plt.plot(test.index, denormalized_test, label='Actual Test Close Prices', color='green')
plt.plot(test.index, predictions_GRA, label='GRA-Based Predictions', linestyle='--', color='red')
plt.plot(test.index, lasso_predictions, label='Lasso Predictions', linestyle=':', color='orange')
plt.plot(test.index, rf_predictions, label='Random Forest Predictions', linestyle='-.', color='purple')
plt.title('Stock Price Prediction Comparison')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.tight_layout()
plt.show()