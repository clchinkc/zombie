
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
    df['ma7'] = df['Close'].rolling(window=7).mean()
    df['std7'] = df['Close'].rolling(window=7).std()
    return df[['ma7', 'std7']], df['Close']

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

def make_prediction(model, last_7_days):
    """Makes a prediction for the next day's closing price."""
    new_features = pd.DataFrame({
        'ma7': [last_7_days['Close'].mean()],
        'std7': [last_7_days['Close'].std()]
    })
    return model.predict(new_features)[0]

if __name__ == '__main__':
    # Load data
    ticker = 'AAPL'
    df = load_data(ticker)

    # Create features
    X, y = create_features(df)

    # Split data into train and test set
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Make prediction for next day's closing price
    last_7_days = df.iloc[-7:]
    prediction = make_prediction(model, last_7_days)
    print(f"Next day's closing price prediction: {prediction:.2f}")
