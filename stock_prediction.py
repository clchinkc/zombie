
# Stock Prediction
# https://neptune.ai/blog/predicting-stock-prices-using-machine-learning
# https://www.youtube.com/watch?v=WcfKaZL4vpA
# stock prediction google search

"""
Project: Quantitative Analysis of Stock Data with Python
Overview:
In this project, we will use Python to perform a quantitative analysis of stock data to predict future stock price movements.

Specifically, we will be using the following steps:

Data Collection:
First, we will collect historical stock price data for the selected stock from a reliable source such as Yahoo Finance or Quandl or Hong Kong Stock Exchange and store it in a Pandas DataFrame.

Data Preparation:
We will clean and preprocess the data by removing any missing values, calculating returns, and calculating moving averages. We will also normalize the data to compare it across different time periods.

Data Analysis:
We will use the prepared data to calculate various metrics, such as daily and annualized returns, volatility, and risk using numpy, and compare them to the broader market using benchmarks such as the Hang Seng Index.

Data Visualization:
We will visualize the results using Matplotlib and create charts to gain insights into the stock's performance over time and various technical indicators, such as moving averages or Bollinger Bands, to help identify potential trading opportunities.

Feature Engineering:
We will calculate additional features such as returns, volatility, and trading volumes to enhance the analysis. We will explore various methods for feature selection, such as correlation analysis or principal component analysis, to identify the most important features for predicting stock price movements.

Machine Learning:
We will use the prepared data to create a machine learning model to predict future stock price movements based on the features.

Backtesting:
We will use the model to develop a trading strategy and backtest it to evaluate its performance over time, compare it to a benchmark model, such as a simple buy-and-hold strategy, and refine the model by adjusting the hyperparameters or using a different algorithm to improve its performance.

Reporting:
Finally, we will produce a report or dashboard summarizing the analysis and results to communicate the findings to stakeholders.

"""


"""
Describe the steps for stock prediction. Then design the code structure in python like classes and functions according to the steps.
Note that you should combine object-oriented programming and functional programming and you should use API whenever you need.
"""


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class StockPredictor:
    def __init__(self, symbol):
        self.symbol = symbol
        self.stock_data = None
        self.features = None
        self.target = None
        self.model = None
        self.predictions = None
    
    def load_stock_data(self, start_date, end_date):
        # Use yfinance API to fetch historical stock price data
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(start=start_date, end=end_date)
        # Store it in self.stock_data
        self.stock_data = data
        
    def plot_data(self):
        # Plot the close column as a line graph
        sns.lineplot(x=self.stock_data.index, y=self.stock_data["Close"])
        plt.show()
    
    def preprocess_data(self):
        # Clean the data and remove missing or outlier values
        # Scale or normalize the data
        # Store the preprocessed data in self.stock_data
        # Select the "Close" column as the target
        y = self.stock_data["Close"]
        # Select the "Open", "High", "Low", and "Volume" columns as the features
        X = self.stock_data[["Open", "High", "Low", "Volume"]]
        # Split the data into a training set (80%) and a test set (20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.features = {"train": X_train, "test": X_test}
        self.target = {"train": y_train, "test": y_test}
    
    def create_features(self):
        # Calculate technical indicators, such as moving averages or relative strength index (RSI)
        # Store the features in self.features
        self.features["train"] = self.features["train"].assign(
            # Calculate the 7-day rolling mean of the closing price
            rolling_mean_7=self.features["train"]["Close"].rolling(7).mean(),
            # Calculate the 30-day rolling mean of the closing price
            rolling_mean_30=self.features["train"]["Close"].rolling(30).mean(),
            # Calculate the 365-day rolling mean of the closing price
            rolling_mean_365=self.features["train"]["Close"].rolling(365).mean(),
            # Calculate the 7-day rolling standard deviation of the closing price
            rolling_std_7=self.features["train"]["Close"].rolling(7).std(),
            # Calculate the 30-day rolling standard deviation of the closing price
            rolling_std_30=self.features["train"]["Close"].rolling(30).std(),
            # Calculate the 365-day rolling standard deviation of the closing price
            rolling_std_365=self.features["train"]["Close"].rolling(365).std(),
        )
        self.features["test"] = self.features["test"].assign(
            # Calculate the 7-day rolling mean of the closing price
            rolling_mean_7=self.features["test"]["Close"].rolling(7).mean(),
            # Calculate the 30-day rolling mean of the closing price
            rolling_mean_30=self.features["test"]["Close"].rolling(30).mean(),
            # Calculate the 365-day rolling mean of the closing price
            rolling_mean_365=self.features["test"]["Close"].rolling(365).mean(),
            # Calculate the 7-day rolling standard deviation of the closing price
            rolling_std_7=self.features["test"]["Close"].rolling(7).std(),
            # Calculate the 30-day rolling standard deviation of the closing price
            rolling_std_30=self.features["test"]["Close"].rolling(30).std(),
            # Calculate the 365-day rolling standard deviation of the closing price
            rolling_std_365=self.features["test"]["Close"].rolling(365).std(),
        )
        # exponentially weighted moving average
        self.features["train"] = self.features["train"].assign(
            # Calculate the 7-day exponentially weighted moving average of the closing price
            ewma_7=self.features["train"]["Close"].ewm(span=7).mean(),
            # Calculate the 30-day exponentially weighted moving average of the closing price
            ewma_30=self.features["train"]["Close"].ewm(span=30).mean(),
            # Calculate the 365-day exponentially weighted moving average of the closing price
            ewma_365=self.features["train"]["Close"].ewm(span=365).mean(),
        )
        self.features["test"] = self.features["test"].assign(
            # Calculate the 7-day exponentially weighted moving average of the closing price
            ewma_7=self.features["test"]["Close"].ewm(span=7).mean(),
            # Calculate the 30-day exponentially weighted moving average of the closing price
            ewma_30=self.features["test"]["Close"].ewm(span=30).mean(),
            # Calculate the 365-day exponentially weighted moving average of the closing price
            ewma_365=self.features["test"]["Close"].ewm(span=365).mean(),
        )
        # Calculate the difference between the closing price and the 7-day rolling mean
        self.features["train"]["diff_rolling_mean_7"] = self.features["train"]["Close"] - self.features["train"]["rolling_mean_7"]
        self.features["test"]["diff_rolling_mean_7"] = self.features["test"]["Close"] - self.features["test"]["rolling_mean_7"]
        # Calculate the difference between the closing price and the 30-day rolling mean
        self.features["train"]["diff_rolling_mean_30"] = self.features["train"]["Close"] - self.features["train"]["rolling_mean_30"]
        self.features["test"]["diff_rolling_mean_30"] = self.features["test"]["Close"] - self.features["test"]["rolling_mean_30"]
        # Calculate the difference between the closing price and the 365-day rolling mean
        self.features["train"]["diff_rolling_mean_365"] = self.features["train"]["Close"] - self.features["train"]["rolling_mean_365"]
        self.features["test"]["diff_rolling_mean_365"] = self.features["test"]["Close"] - self.features["test"]["rolling_mean_365"]
        # Calculate the difference between the closing price and the 7-day exponentially weighted moving average
        self.features["train"]["diff_ewma_7"] = self.features["train"]["Close"] - self.features["train"]["ewma_7"]
        self.features["test"]["diff_ewma_7"] = self.features["test"]["Close"] - self.features["test"]["ewma_7"]
        # Calculate the difference between the closing price and the 30-day exponentially weighted moving average
        self.features["train"]["diff_ewma_30"] = self.features["train"]["Close"] - self.features["train"]["ewma_30"]
        self.features["test"]["diff_ewma_30"] = self.features["test"]["Close"] - self.features["test"]["ewma_30"]
        # Calculate the difference between the closing price and the 365-day exponentially weighted moving average
        self.features["train"]["diff_ewma_365"] = self.features["train"]["Close"] - self.features["train"]["ewma_365"]
        self.features["test"]["diff_ewma_365"] = self.features["test"]["Close"] - self.features["test"]["ewma_365"]
        # relative strength index
        self.features["train"] = self.features["train"].assign(
            # Calculate the 7-day relative strength index (RSI)
            rsi_7=ta.rsi(self.features["train"]["Close"], timeperiod=7),
            # Calculate the 30-day relative strength index (RSI)
            rsi_30=ta.rsi(self.features["train"]["Close"], timeperiod=30),
            # Calculate the 365-day relative strength index (RSI)
            rsi_365=ta.rsi(self.features["train"]["Close"], timeperiod=365),
        )
        self.features["test"] = self.features["test"].assign(
            # Calculate the 7-day relative strength index (RSI)
            rsi_7=ta.rsi(self.features["test"]["Close"], timeperiod=7),
            # Calculate the 30-day relative strength index (RSI)
            rsi_30=ta.rsi(self.features["test"]["Close"], timeperiod=30),
            # Calculate the 365-day relative strength index (RSI)
            rsi_365=ta.rsi(self.features["test"]["Close"], timeperiod=365),
        )
        # Calculate the difference between the closing price and the 7-day relative strength index (RSI)
        self.features["train"]["diff_rsi_7"] = self.features["train"]["Close"] - self.features["train"]["rsi_7"]
    
    def select_model(self, model):
        # Choose an appropriate machine learning algorithm
        # Set the model object to the selected algorithm
        self.model = model
    
    def train_model(self):
        # Train the selected model on the preprocessed data
        # Perform parameter tuning to optimize the model's performance
        # Store the trained model in self.model
        # Fit a linear regression model to the training data
        self.model.fit(self.features["train"], self.target["train"])
        self.predictions["test"] = self.model.predict(self.features["test"])
    
    def evaluate_model(self):
        # Evaluate the performance of the trained model on a test set of data
        # Print out the accuracy and other relevant metrics
        # Print the score and mean squared error
        score = self.model.score(self.features["test"], self.target["test"])
        mse = mean_squared_error(self.target["test"], self.predictions["test"])
        # Print the model's coefficients and intercept (slope and y-intercept)
        print(f"Coefficients: {self.model.coef_}")
        print(f"Intercept: {self.model.intercept_}")
        return score, mse
    
    def predict(self, date):
        # Use the trained model to predict the stock price on a given date
        # Return the predicted stock price
        # Get the last row of the features
        last_row = self.features["test"].iloc[-1]
        # Get the date of the last row
        last_date = last_row.name
        # Get the last date in the index
        last_date_in_index = self.stock_data.index[-1]
        # Check if the last date in the index is the same as the last date in the features
        if last_date == last_date_in_index:
            # If it is, then add 1 day to the date
            last_date += pd.Timedelta(days=1)
        # Create a new row with the same values as the last row
        new_row = last_row.copy()
        # Set the index of the new row to the new date
        new_row.name = last_date
        # Append the new row to the features
        self.features["test"] = self.features["test"].append(new_row)
        # Get the predicted price
        predicted_price = self.model.predict(self.features["test"])[-1]
        # Return the predicted price
        return predicted_price
    
    def plot_predictions(self):
        # Plot the predicted values alongside the actual values in a line graph
        sns.lineplot(x=self.stock_data.index, y=self.stock_data["Close"], label="Actual")
        sns.lineplot(x=self.predictions.index, y=self.predictions["Close"], label="Predicted")
        plt.show()

predictor = StockPredictor('AAPL')
predictor.load_stock_data('2010-01-01', '2022-03-12')
predictor.plot_data()
predictor.preprocess_data()
predictor.create_features()
predictor.select_model(LinearRegression())
predictor.train_model()
score, mse = predictor.evaluate_model()
print('Score:', score)
print('Mean squared error:', mse)
predicted_price = predictor.predict('2023-03-19')
print('Predicted stock price:', predicted_price)
predictor.plot_predictions()








# Plot the predicted values alongside with the actual values in a line graph using sns.lineplot
# sns.lineplot(data=y_test, x=X_test.index, y=y_test, label="Actual")
# sns.lineplot(data=predictions, x=X_test.index, y=predictions, label="Predicted")
# plt.show()

# Combine the actual and predicted values into a single DataFrame
# don't get the predictions of X_train, ignore the predictions of X_train
# training = pd.DataFrame({"close": y_train, "predicted": [None] * len(y_train)}, index=X_train.index)
# results = pd.DataFrame({"close": y_test, "predicted": predictions}, index=X_test.index)

# Merge the results with the original stock data
# stock_data_with_results = pd.concat([training, results], axis=0)

# Plot the data using seaborn
# sns.lineplot(data=stock_data_with_results[["close"]])
# sns.lineplot(data=stock_data_with_results[["predicted"]][len(X_train):])
# plt.show()

# add a list of None to the predictions
predictions = [None] * len(X_train) + predictions.tolist()

# make prediction a dataframe with the same index as stock_data["Close"]
predictions = pd.DataFrame(predictions, index=stock_data.index, columns=["Close"])

# plot the predicted value and true value in different colour in the similar plot with the first plot
sns.lineplot(x=stock_data.index, y=stock_data["Close"])
# plot it in the rightmost part of the plot
sns.lineplot(x=predictions.index, y=predictions["Close"])
plt.show()

# print stock_data["Close"]
print(stock_data["Close"][len(X_train):])
# print predictions
print(predictions)