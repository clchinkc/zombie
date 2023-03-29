
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def load_data(ticker):
    """Loads historical data for the given stock ticker."""
    # ticker_data = yf.Ticker(ticker)
    # data = ticker_data.history(period='max')
    stock_data = pd.read_csv('apple_stock_data.csv', index_col=0, parse_dates=True)
    return stock_data

def standardize_data(df):
    """Standardizes historical data by using StandardScaler."""
    scaler = StandardScaler()
    df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    df['Open'] = scaler.fit_transform(df['Open'].values.reshape(-1, 1))
    df['High'] = scaler.fit_transform(df['High'].values.reshape(-1, 1))
    df['Low'] = scaler.fit_transform(df['Low'].values.reshape(-1, 1))
    df['Volume'] = scaler.fit_transform(df['Volume'].values.reshape(-1, 1))
    return df

def create_features(df):
    """Creates new features from the historical data."""
    # Add open, high, low, and volume features for the previous day
    # use the mean of previous 7 days if that day is missing
    df['prev_close'] = df['Close'].shift(1)
    df['prev_open'] = df['Open'].shift(1)
    df['prev_high'] = df['High'].shift(1)
    df['prev_low'] = df['Low'].shift(1)
    df['prev_volume'] = df['Volume'].shift(1)
    df['ma7'] = df['prev_close'].rolling(window=7, min_periods=1).mean()
    df['std7'] = df['prev_close'].rolling(window=7, min_periods=1).std()
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def clean_data(df):
    """Cleans historical data by removing rows with missing values."""
    df = df.fillna(method='ffill')
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

def train_model(X_train, y_train):
    """Trains a linear regression model on the training set."""
    PolyRegression = Pipeline([('poly', PolynomialFeatures(degree=3)),
                                ('linear', LinearRegression(fit_intercept=False))])
    PolyRegression.fit(X_train, y_train)
    return PolyRegression

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
        new_features = pd.concat([last_7_days.iloc[1:], pd.DataFrame({
            'prev_close': prediction,
            'prev_open': last_7_days.iloc[-1]['prev_open'],
            'prev_high': last_7_days.iloc[-1]['prev_high'],
            'prev_low': last_7_days.iloc[-1]['prev_low'],
            'prev_volume': last_7_days.iloc[-1]['prev_volume'],
            'ma7': pd.concat([last_7_days['prev_close'], pd.Series([prediction])]).tail(7).mean(),
            'std7': pd.concat([last_7_days['prev_close'], pd.Series([prediction])]).tail(7).std(),
            'rsi': last_7_days['rsi'].mean()  # Ideally, you should recalculate the RSI, but for simplicity, you can use the mean here
            }, index=[0])], ignore_index=True)
        last_7_days = new_features.tail(7)
    return np.array(predictions)

def destandardize_data(df, predictions):
    """De-standardizes the predicted closing prices."""
    scaler = StandardScaler()
    scaler.fit(df['Close'].values.reshape(-1, 1))
    return scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

if __name__ == '__main__':
    # Load data
    ticker = 'AAPL'
    # df = load_data(ticker)
    df = pd.read_csv('apple_stock_data.csv', index_col='Date', parse_dates=True)
    
    standardized_data = standardize_data(df)

    # Create features
    featured_data = create_features(standardized_data)

    # Clean data
    clean_data = clean_data(featured_data)
    
    # Create X and y
    X = clean_data[['prev_close', 'prev_open', 'prev_high', 'prev_low', 'prev_volume', 'ma7', 'std7', 'rsi']]
    y = clean_data['Close']

    # Split data into train and test set
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)


    # Make prediction for next 30 days' closing prices
    last_7_days = X.tail(7)
    predictions = make_prediction(model, last_7_days, days_to_predict=30)
    destandardized_predictions = destandardize_data(df, predictions)
    dates = pd.date_range(start=df.index[-1], periods=31, freq='D')

    # Plot the predicted prices
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    plt.plot(df.index, df['Close'], label='Actual')
    plt.plot(dates, np.concatenate([[df['Close'][-1]], destandardized_predictions]), label='Predicted')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()
