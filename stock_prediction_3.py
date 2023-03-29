
"""
This code performs time-series analysis using KMeans clustering and Markov chain modeling to predict future stock prices. Here is an overview of what each section of the code does:

Step 1: Data Preprocessing

Reads in the stock data from a CSV file and drops any missing values.
Scales the Close prices using the MinMaxScaler.
Step 2: Defining the States

Defines the number of states to use in the Markov chain model.
Uses KMeans clustering to group the scaled Close prices into the specified number of clusters (i.e., states).
Assigns each data point to its corresponding state.
Step 3: Transition Probability Matrix

Constructs a transition probability matrix based on the frequency of transitions between the states in the data.
Normalizes the matrix so that each row sums to 1.
Step 4: Predicting the Future States

Calculates the probability of being in each state after one time step based on the current state probabilities and the transition probability matrix.
Step 5: Predicting the Future Prices

Calculates the estimated future price by taking a weighted average of the median prices of all the future states.
Inversely scales the estimated future price to get the final prediction.
"""


from abc import ABC
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels.tsa.api
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.svm import SVC
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class StockDataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = None
        self.train_data = None
        self.test_data = None

    def get_stock_data(self, symbol, start_date, end_date):
        # Use yfinance API to fetch historical stock price data
        ticker = yf.Ticker(symbol)
        stock_data = ticker.history(start=start_date, end=end_date)
        # Store it in a CSV file
        stock_data.to_csv(self.filepath)
        return stock_data

    def load_stock_data(self):
        # Load historical stock data from CSV file
        stock_data = pd.read_csv(self.filepath, index_col='Date', parse_dates=True)[['Close']]
        return stock_data

    def preprocess_data(self, stock_data, days):
        # Drop any missing values
        stock_data = stock_data.dropna()
        # Divide the data into training and testing sets
        self.train_data = stock_data.iloc[:-days]
        self.test_data = stock_data.iloc[-days:]
        self.scaler = MinMaxScaler()
        self.train_data = self.scaler.fit_transform(self.train_data)
        self.test_data = self.scaler.transform(self.test_data)
        return self.train_data, self.test_data, self.scaler

class StatesGetter:
    def __init__(self, stock_data, num_states):
        self.stock_data = stock_data
        self.num_states = num_states
        self.kmeans = None
        self.states = None

    def get_states(self):
        # Use KMeans clustering to group the scaled Close prices into the specified number of clusters (i.e., states)
        self.kmeans = KMeans(n_clusters=self.num_states, init='k-means++', n_init="auto", max_iter=1000, random_state=42)
        self.kmeans.fit(self.stock_data)
        self.states = self.kmeans.labels_
        return self.states

class Model(ABC):
    def train(self, stock_data, states):
        pass
    
    def predict(self, stock_data, states):
        pass


class MarkovChain(Model):
    def __init__(self):
        self.transition_prob_matrix = None

    def train(self, stock_data, states):
        hist = np.histogram2d(states[:-1], states[1:], bins=len(set(states)))[0]
        self.transition_prob_matrix = hist / hist.sum(axis=1, keepdims=True)
        
    def predict(self, stock_data, states):
        current_state_prob = np.array([1/len(set(states))]*len(set(states)))
        future_state_prob = np.dot(current_state_prob, self.transition_prob_matrix)
        future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
        estimated_future_price = np.dot(future_state_prob, future_prices)
        estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
        return estimated_future_price_2d

class MLP(Model):
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, solver='adam', random_state=42)

    def train(self, stock_data, states):
        self.model.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    def predict(self, stock_data, states):
        proba = self.model.predict_proba(stock_data[-1].reshape(-1, 1))
        future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
        estimated_future_price = np.dot(proba[-1], future_prices)
        estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
        return estimated_future_price_2d

class Forest(Model):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    def train(self, stock_data, states):
        self.model.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    def predict(self, stock_data, states):
        proba = self.model.predict_proba(stock_data[-1].reshape(-1, 1))
        future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
        estimated_future_price = np.dot(proba[-1], future_prices)
        estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
        return estimated_future_price_2d

class SVM(Model):
    def __init__(self):
        self.model = SVC(gamma="auto", kernel='rbf', probability=True, random_state=42)

    def train(self, stock_data, states):
        self.model.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    def predict(self, stock_data, states):
        proba = self.model.predict_proba(stock_data[-1].reshape(-1, 1))
        future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
        estimated_future_price = np.dot(proba[-1], future_prices)
        estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
        return estimated_future_price_2d

class Logistic(Model):
    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42)

    def train(self, stock_data, states):
        self.model.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    def predict(self, stock_data, states):
        proba = self.model.predict_proba(stock_data[-1].reshape(-1, 1))
        future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
        estimated_future_price = np.dot(proba[-1], future_prices)
        estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
        return estimated_future_price_2d

class KNN(Model):
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)

    def train(self, stock_data, states):
        self.model.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    def predict(self, stock_data, states):
        proba = self.model.predict_proba(stock_data[-1].reshape(-1, 1))
        future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
        estimated_future_price = np.dot(proba[-1], future_prices)
        estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
        return estimated_future_price_2d

class Gaussian(Model):
    def __init__(self):
        self.model = GaussianNB()

    def train(self, stock_data, states):
        self.model.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    def predict(self, stock_data, states):
        proba = self.model.predict_proba(stock_data[-1].reshape(-1, 1))
        future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
        estimated_future_price = np.dot(proba[-1], future_prices)
        estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
        return estimated_future_price_2d

class Exponential(Model):
    def __init__(self):
        self.model = None
        self.days = 0

    def train(self, stock_data, states):
        self.model = ExponentialSmoothing(stock_data, trend='add', seasonal='add', seasonal_periods=30)
        self.model_fit = self.model.fit()

    def predict(self, stock_data, states):
        # it will be called iteratively with the last predicted value
        predictions = self.model_fit.predict(start=len(stock_data) + self.days, end=len(stock_data) + self.days)
        self.days += 1
        return np.array([predictions]).reshape(-1, 1)

class StockPredictor:
    def __init__(self, model, filepath, num_states, days):
        self.model = model
        self.filepath = filepath
        self.num_states = num_states
        self.days = days

    def preprocess_data(self):
        stock_data_preprocessor = StockDataPreprocessor(self.filepath)
        stock_data = stock_data_preprocessor.load_stock_data()
        self.train_data, self.test_data, self.scaler = stock_data_preprocessor.preprocess_data(stock_data, self.days)

    def get_states(self):
        states_getter = StatesGetter(self.train_data, self.num_states)
        self.states = states_getter.get_states()

    def train(self):
        self.model.train(self.train_data, self.states)

    def predict(self):
        stock_prices = self.train_data
        self.predicted_prices = []
        for _ in range(self.days):
            last_price = self.model.predict(stock_prices, self.states)
            stock_prices = np.append(stock_prices, last_price, axis=0)[1:]
            self.predicted_prices.append(last_price)

    def evaluate(self):
        predicted_prices = self.scaler.inverse_transform(np.array(self.predicted_prices).reshape(-1, 1))
        actual_prices = self.scaler.inverse_transform(self.test_data)[-self.days:]
        residuals = actual_prices - predicted_prices
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        r2 = r2_score(actual_prices, predicted_prices)
        print("RMSE:", rmse)
        print("R2:", r2)

        # plot the residuals in a scatter plot
        plt.scatter(range(len(residuals)), residuals)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Time')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()

    def plot_prediction(self):
        predicted_prices = self.scaler.inverse_transform(np.array(self.predicted_prices).reshape(-1, 1))
        historical_prices = self.scaler.inverse_transform(np.array(self.train_data).reshape(-1, 1))
        actual_prices = self.scaler.inverse_transform(np.array(self.test_data).reshape(-1, 1))
        
        # Plot the predicted prices along with the historical data
        historical_dates = pd.read_csv(self.filepath)['Date'].values[-len(historical_prices):]
        dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None) for date in historical_dates]

        plt.plot(dates[:-len(actual_prices)], historical_prices[:-len(actual_prices)], label='Historical Prices')
        actual_prices = np.insert(actual_prices, 0, historical_prices[-len(actual_prices)], axis=0)
        plt.plot(dates[-len(actual_prices):], actual_prices, label='Actual Prices')
        predicted_prices = np.insert(predicted_prices, 0, historical_prices[-len(predicted_prices)], axis=0)
        plt.plot(dates[-len(predicted_prices):], predicted_prices, label='Predicted Prices')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    filepath = 'apple_stock_data.csv'
    num_states = 4
    days = 30
    # markov_chain = MarkovChain()
    # mlp = MLP()
    # forest = Forest()
    # svm = SVM()
    # logistic = Logistic()
    # knn = KNN()
    # gaussian = Gaussian()
    exponential = Exponential()
    stock_predictor = StockPredictor(exponential, filepath, num_states, days)
    stock_predictor.preprocess_data()
    stock_predictor.get_states()
    stock_predictor.train()
    stock_predictor.predict()
    stock_predictor.evaluate()
    stock_predictor.plot_prediction()


# Linear Regression
# Gradient Boosting Regressor
# Random Forest Regressor
# Support Vector Regressor
# K-Nearest Neighbors Regressor

# Grid Search
# K-Fold Cross Validation

"""
Add volatility term:
To add volatility term, you can use the volatility of the stock price to adjust the predicted price. For example, you can use the standard deviation of the stock price to estimate the volatility.

Use a more sophisticated clustering algorithm:
To use a more sophisticated clustering algorithm, you can try hierarchical clustering, density-based clustering, or model-based clustering algorithms and evaluate their performance.

Cluster based on the price change (returns):
data['Return'] = np.log(data['Close'] / data['Close'].shift(1)).fillna(np.mean(np.log(data['Close'] / data['Close'].shift(1))))
km = KMeans(n_clusters=2, random_state=0).fit(returns_scaled)
data['Cluster'] = km.labels_
Then the the returns, volatility, and Sharpe ratio of the stocks in each cluster can be calculated as follows:
The "Annualized Return" is the average return of the stocks in each cluster over the period of time in the dataset, scaled up to a one-year period. A positive value means that the stocks in that cluster on average had a positive return, while a negative value means they had a negative return. In this case, cluster 0 had an annualized return of 4.45, while cluster 1 had an annualized return of -3.93.
The "Annualized Volatility" is a measure of the variability of the returns of the stocks in each cluster over the period of time in the dataset, scaled up to a one-year period. A higher value means that the stocks in that cluster had more volatile returns, while a lower value means they had less volatile returns. In this case, cluster 0 had an annualized volatility of 3.99, while cluster 1 had an annualized volatility of 4.19.
The "Sharpe Ratio" is a measure of risk-adjusted performance that takes into account both the return and volatility of the stocks in each cluster. A higher value means that the stocks in that cluster had a better risk-adjusted performance, while a lower value means they had a worse risk-adjusted performance. In this case, cluster 0 had a Sharpe Ratio of -12.39, while cluster 1 had a Sharpe Ratio of -43.58.


Consider more features:
To consider more features, you can collect and preprocess additional data such as volume, news sentiment, technical indicators, and economic indicators. You can then use feature selection methods such as correlation analysis, mutual information, or LASSO regression to select the most relevant features for the model.

Use a different time scale:
To use a different time scale, you can adjust the time interval of the data used for training and testing the model. For example, you can use hourly or weekly data instead of daily data.

Evaluate and fine-tune the model:
To evaluate and fine-tune the model, you can use cross-validation techniques such as k-fold cross-validation or time series cross-validation to estimate the performance of the model. You can also use hyperparameter tuning techniques such as grid search, random search, or Bayesian optimization to optimize the parameters of the model.
"""
