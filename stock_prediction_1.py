
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_data(ticker):
    """Loads historical data for the given stock ticker."""
    ticker_data = yf.Ticker(ticker)
    return ticker_data.history(period='max')

def create_features(df):
    """Creates new features from the historical data."""
    # Add open, high, low, and volume features for the previous day
    # use the mean of previous 7 days if that day is missing
    df['prev_close'] = df['Close'].shift(1, fill_value=np.mean(df['Close']))
    df['prev_open'] = df['Open'].shift(1, fill_value=np.mean(df['Open']))
    df['prev_high'] = df['High'].shift(1, fill_value=np.mean(df['High']))
    df['prev_low'] = df['Low'].shift(1, fill_value=np.mean(df['Low']))
    df['prev_volume'] = df['Volume'].shift(1, fill_value=np.mean(df['Volume']))
    df['ma7'] = df['prev_close'].rolling(window=7, min_periods=1).mean()
    df['std7'] = df['prev_close'].rolling(window=7, min_periods=1).std()
    df['std7'] = df['std7'].fillna(df['std7'].mean())
    df['ma30'] = df['prev_close'].rolling(window=30, min_periods=1).mean()
    df['std30'] = df['prev_close'].rolling(window=30, min_periods=1).std()
    df['std30'] = df['std30'].fillna(df['std30'].mean())
    return df

def clean_data(df):
    """Cleans historical data by removing rows with missing values."""
    df = df.dropna()
    return df

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train):
    """Trains a linear regression model on the training set."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the performance of the model on the test set."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean squared error: {mse:.2f}")
    print(f"Coefficient of determination (R²): {r2:.2f}")

def make_prediction(model, last_7_days, days_to_predict):
    """Makes a prediction for the next `days_to_predict` days' closing prices."""
    predictions = []
    for i in range(days_to_predict):
        new_features = pd.DataFrame({
            'prev_close': last_7_days['prev_close'].rolling(window=7, min_periods=1).mean(),
            'prev_open': last_7_days['prev_open'].rolling(window=7, min_periods=1).mean(),
            'prev_high': last_7_days['prev_high'].rolling(window=7, min_periods=1).mean(),
            'prev_low': last_7_days['prev_low'].rolling(window=7, min_periods=1).mean(),
            'prev_volume': last_7_days['prev_volume'].rolling(window=7, min_periods=1).mean(),
            'ma7': last_7_days['prev_close'].rolling(window=7, min_periods=1).mean(),
            'std7': last_7_days['prev_close'].rolling(window=7, min_periods=1).std(),
            'ma30': last_7_days['prev_close'].rolling(window=30, min_periods=1).mean(),
            'std30': last_7_days['prev_close'].rolling(window=30, min_periods=1).std(),
        })
        new_features['std7'] = new_features['std7'].fillna(new_features['std7'].mean())
        new_features['std30'] = new_features['std30'].fillna(new_features['std30'].mean())
        prediction = model.predict(new_features)[0]
        predictions.append(prediction)
        last_7_days = pd.concat([last_7_days, new_features], ignore_index=True).iloc[-7:]
    return np.array(predictions)


if __name__ == '__main__':
    # Load data
    ticker = 'AAPL'
    # df = load_data(ticker)
    df = pd.read_csv('apple_stock_data.csv', index_col='Date', parse_dates=True)

    # Create features
    df = create_features(df)

    # Clean data
    df = clean_data(df)
    
    # Create X and y
    X = df[['prev_close', 'prev_open', 'prev_high', 'prev_low', 'prev_volume', 'ma7', 'std7', 'ma30', 'std30']]
    y = df['Close']

    # Split data into train and test set
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)


    # Make prediction for next 30 days' closing prices
    last_7_days = df.iloc[-7:]
    predictions = make_prediction(model, last_7_days, days_to_predict=30)
    dates = pd.date_range(start=df.index[-1], periods=31, freq='D')

    # Plot the predicted prices
    import matplotlib.pyplot as plt
    plt.plot(df.index[-100:], df['Close'][-100:], label='Actual')
    plt.plot(dates, np.concatenate([[df['Close'][-1]], predictions]), label='Predicted')
    plt.legend()
    plt.show()
