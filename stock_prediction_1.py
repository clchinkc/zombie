
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.model_selection import train_test_split


def load_data(ticker):
    """Loads historical data for the given stock ticker."""
    # ticker_data = yf.Ticker(ticker)
    # data = ticker_data.history(period='max')
    data = pd.read_csv('apple_stock_data.csv', index_col=0, parse_dates=True)
    return data

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
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean squared error: {mse:.2f}")
    r2 = r2_score(y_test, y_pred)
    print(f"Coefficient of determination (R²): {r2:.2f}")
    

def make_prediction(model, last_7_days, days_to_predict):
    """Makes a prediction for the next `days_to_predict` days' closing prices."""
    predictions = []
    for i in range(days_to_predict):
        prediction = model.predict(last_7_days)[0]
        predictions.append(prediction)
        new_features = pd.DataFrame({
            'prev_close': prediction,
            'prev_open': last_7_days['prev_open'].mean(),
            'prev_high': last_7_days['prev_high'].mean(),
            'prev_low': last_7_days['prev_low'].mean(),
            'prev_volume': last_7_days['prev_volume'].mean(),
            'ma7': last_7_days['ma7'].mean(),
            'std7': last_7_days['std7'].std(),
        }, index=[0])
        new_features['std7'] = new_features['std7'].fillna(new_features['std7'].mean())
        last_7_days = pd.concat([last_7_days, new_features], ignore_index=True)
        last_7_days = last_7_days.iloc[-7:]
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
    X = df[['prev_close', 'prev_open', 'prev_high', 'prev_low', 'prev_volume', 'ma7', 'std7']]
    y = df['Close']

    # Split data into train and test set
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)


    # Make prediction for next 30 days' closing prices
    last_7_days = X_test.iloc[-7:]
    predictions = make_prediction(model, last_7_days, days_to_predict=30)
    dates = pd.date_range(start=df.index[-1], periods=31, freq='D')

    # Plot the predicted prices
    import matplotlib.pyplot as plt
    plt.plot(df.index[-100:], df['Close'][-100:], label='Actual')
    plt.plot(dates, np.concatenate([[df['Close'][-1]], predictions]), label='Predicted')
    plt.legend()
    plt.show()
