
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
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import (
    ARDRegression,
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    QuantileRegressor,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


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
    def train(self, stock_data, **kwargs):
        pass
    
    def predict(self, stock_data, **kwargs):
        pass


class MarkovChain(Model):
    def __init__(self):
        self.transition_prob_matrix = None

    def train(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        hist = np.histogram2d(states[:-1], states[1:], bins=len(set(states)))[0]
        self.transition_prob_matrix = hist / hist.sum(axis=1, keepdims=True)
        
    def predict(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        current_state_prob = np.array([1/len(set(states))]*len(set(states)))
        future_state_prob = np.dot(current_state_prob, self.transition_prob_matrix)
        future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
        estimated_future_price = np.dot(future_state_prob, future_prices)
        estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
        return estimated_future_price_2d

class MLP(Model):
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, solver='adam', random_state=42)

    def train(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        self.model.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    def predict(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        proba = self.model.predict_proba(stock_data[-1].reshape(-1, 1))
        future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
        estimated_future_price = np.dot(proba[-1], future_prices)
        estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
        return estimated_future_price_2d

class Forest(Model):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    def train(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        self.model.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    def predict(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        proba = self.model.predict_proba(stock_data[-1].reshape(-1, 1))
        future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
        estimated_future_price = np.dot(proba[-1], future_prices)
        estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
        return estimated_future_price_2d

class SVM(Model):
    def __init__(self):
        self.model = SVC(gamma="auto", kernel='rbf', probability=True, random_state=42)

    def train(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        self.model.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    def predict(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        proba = self.model.predict_proba(stock_data[-1].reshape(-1, 1))
        future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
        estimated_future_price = np.dot(proba[-1], future_prices)
        estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
        return estimated_future_price_2d

class Logistic(Model):
    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42)

    def train(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        self.model.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    def predict(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        proba = self.model.predict_proba(stock_data[-1].reshape(-1, 1))
        future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
        estimated_future_price = np.dot(proba[-1], future_prices)
        estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
        return estimated_future_price_2d
    
class SGD(Model):
    def __init__(self):
        self.model = SGDClassifier(loss="log", max_iter=1000, tol=1e-3)

    def train(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        self.model.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    def predict(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        proba = self.model.predict_proba(stock_data[-1].reshape(-1, 1))
        future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
        estimated_future_price = np.dot(proba[-1], future_prices)
        estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
        return estimated_future_price_2d

class KNN(Model):
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)

    def train(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        self.model.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    def predict(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        proba = self.model.predict_proba(stock_data[-1].reshape(-1, 1))
        future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
        estimated_future_price = np.dot(proba[-1], future_prices)
        estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
        return estimated_future_price_2d

class Gradient(Model):
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    def train(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        self.model.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    def predict(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        proba = self.model.predict_proba(stock_data[-1].reshape(-1, 1))
        future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
        estimated_future_price = np.dot(proba[-1], future_prices)
        estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
        return estimated_future_price_2d

class Gaussian(Model):
    def __init__(self):
        self.model = GaussianNB()

    def train(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        self.model.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    def predict(self, stock_data, **kwargs):
        states = kwargs.get('states', None)
        if states is None:
            raise ValueError("states must be provided.")
        proba = self.model.predict_proba(stock_data[-1].reshape(-1, 1))
        future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
        estimated_future_price = np.dot(proba[-1], future_prices)
        estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
        return estimated_future_price_2d



class Exponential(Model):
    def __init__(self):
        self.model = None
        self.days = 0

    def train(self, stock_data, **kwargs):
        self.model = ExponentialSmoothing(stock_data, trend='add', seasonal='add', seasonal_periods=30)
        self.model_fit = self.model.fit()

    def predict(self, stock_data, **kwargs):
        # it will be called iteratively with the last predicted value
        predictions = self.model_fit.predict(start=len(stock_data) + self.days, end=len(stock_data) + self.days)
        self.days += 1
        return np.array([predictions]).reshape(-1, 1)
    
# https://zhuanlan.zhihu.com/p/158548189
# p: Represents the order of the autoregressive (AR) component of the ARIMA model. It indicates the number of past observations that affect the current value. The value of p can be determined by looking at the PACF plot of the time series. If the PACF plot shows a sharp drop-off after the first few lags, it suggests that the AR component has a low order. On the other hand, if the PACF plot shows a slow decay, it indicates a high order of the AR component.
# d: Represents the degree of differencing required to make the time series stationary. It indicates the number of times the data needs to be differenced to remove any trend or seasonality. The value of d can be determined by analyzing the ACF plot of the first difference of the time series. If the ACF plot shows a rapid decay, it suggests that the first difference of the time series is stationary and d=1. If not, you may need to difference the data multiple times until it becomes stationary.
# q: Represents the order of the moving average (MA) component of the ARIMA model. It indicates the number of past errors that affect the current value. The value of q can be determined by looking at the ACF plot of the time series residuals after fitting an AR model. If the ACF plot shows a sharp drop-off after the first few lags, it suggests that the MA component has a low order. On the other hand, if the ACF plot shows a slow decay, it indicates a high order of the MA component.
class arima(Model):
    def __init__(self):
        self.model = None
        self.days = 0

    def train(self, stock_data, **kwargs):
        # self.model = ARIMA(stock_data, order=(1, 1, 1))
        self.model = auto_arima(stock_data, start_p=0, start_q=0, max_p=5, max_q=5, m=12, start_P=0, seasonal=True, d=1, D=1, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
        # self.model_fit = self.model.fit()

    def predict(self, stock_data, **kwargs):
        # predictions = self.model.predict(start=len(stock_data) + self.days, end=len(stock_data) + self.days)
        predictions = self.model.predict(n_periods=self.days + 1)[-1]
        self.days += 1
        return np.array([predictions]).reshape(-1, 1)

class sarimax(Model):
    def __init__(self):
        self.model = None
        self.days = 0

    def train(self, stock_data, **kwargs):
        self.model = SARIMAX(stock_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        self.model = self.model.fit()

    def predict(self, stock_data, **kwargs):
        # it will be called iteratively with the last predicted value
        predictions = self.model.predict(start=len(stock_data) + self.days, end=len(stock_data) + self.days)
        self.days += 1
        return np.array([predictions]).reshape(-1, 1)

# self.model.plot(forecast)
# self.model.plot_components(forecast)
class prophet(Model):
    def __init__(self):
        self.model = Prophet(seasonality_mode='additive', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        self.days = 0

    def train(self, stock_data, **kwargs):
        dates = pd.date_range('2020-01-01', periods=len(stock_data))
        stock_data = stock_data.reshape(-1)
        df = pd.DataFrame({'ds': dates, 'y': stock_data})
        self.model.fit(df)

    def predict(self, stock_data, **kwargs):
        # it will be called iteratively with the last predicted value
        future = self.model.make_future_dataframe(periods=self.days + 1)
        forecast = self.model.predict(future)
        predictions = forecast['yhat'].values[-1]
        self.days += 1
        return np.array([predictions]).reshape(-1, 1)

class Linear(Model):
    def __init__(self):
        self.model = LinearRegression(fit_intercept=False)

    def train(self, stock_data, **kwargs):
        self.model.fit(stock_data[:-1].reshape(-1, 1), stock_data[1:].reshape(-1, ))

    def predict(self, stock_data, **kwargs):
        # it will be called iteratively with the last predicted value
        predictions = self.model.predict(stock_data[-1].reshape(-1, 1))
        return np.array([predictions]).reshape(-1, 1)

class Elastic(Model):
    def __init__(self):
        self.model = ElasticNetCV(l1_ratio=0.5, eps=0.001, n_jobs=-1)

    def train(self, stock_data, **kwargs):
        self.model.fit(stock_data[:-1].reshape(-1, 1), stock_data[1:].reshape(-1, ))

    def predict(self, stock_data, **kwargs):
        # it will be called iteratively with the last predicted value
        predictions = self.model.predict(stock_data[-1].reshape(-1, 1))
        return np.array([predictions]).reshape(-1, 1)

class svr(Model):
    def __init__(self):
        self.model = SVR(kernel='linear', C=10, epsilon=.1, gamma='scale')

    def train(self, stock_data, **kwargs):
        self.model.fit(stock_data[:-1].reshape(-1, 1), stock_data[1:].reshape(-1, ))

    def predict(self, stock_data, **kwargs):
        # it will be called iteratively with the last predicted value
        predictions = self.model.predict(stock_data[-1].reshape(-1, 1))
        return np.array([predictions]).reshape(-1, 1)

class SGDregressor(Model):
    def __init__(self):
        self.model = SGDRegressor(loss='huber', learning_rate='invscaling', penalty=None, alpha=0.001, fit_intercept=False, epsilon=0.05, eta0=0.1, power_t=0.25)

    def train(self, stock_data, **kwargs):
        self.model.fit(stock_data[:-1].reshape(-1, 1), stock_data[1:].reshape(-1, ))

    def predict(self, stock_data, **kwargs):
        # it will be called iteratively with the last predicted value
        predictions = self.model.predict(stock_data[-1].reshape(-1, 1))
        return np.array([predictions]).reshape(-1, 1)

class lasso(Model):
    def __init__(self):
        self.model = LassoCV(cv=5)

    def train(self, stock_data, **kwargs):
        self.model.fit(stock_data[:-1].reshape(-1, 1), stock_data[1:].reshape(-1, ))

    def predict(self, stock_data, **kwargs):
        # it will be called iteratively with the last predicted value
        predictions = self.model.predict(stock_data[-1].reshape(-1, 1))
        return np.array([predictions]).reshape(-1, 1)

class ARDregressor(Model):
    def __init__(self):
        self.model = ARDRegression(compute_score=True)

    def train(self, stock_data, **kwargs):
        self.model.fit(stock_data[:-1].reshape(-1, 1), stock_data[1:].reshape(-1, ))

    def predict(self, stock_data, **kwargs):
        # it will be called iteratively with the last predicted value
        predictions = self.model.predict(stock_data[-1].reshape(-1, 1))
        return np.array([predictions]).reshape(-1, 1)

from hmmlearn import hmm


class GMMHMMModel:
    def __init__(self):
        self.model = hmm.GMMHMM(n_components=1, n_mix=1, covariance_type='diag')

    def train(self, stock_data, **kwargs):
        self.model.fit(stock_data[:-1].reshape(-1, 1))

    def predict(self, stock_data, **kwargs):
        _, state_sequence = self.model.decode(stock_data.reshape(-1, 1))
        state = state_sequence[-1]
        next_mean = self.model.means_[state][0][0]
        return next_mean.reshape(-1, 1)

class gaussianregressor(Model):
    def __init__(self):
        self.model = GaussianProcessRegressor()

    def train(self, stock_data, **kwargs):
        self.model.fit(stock_data[:-1].reshape(-1, 1), stock_data[1:].reshape(-1, ))

    def predict(self, stock_data, **kwargs):
        # it will be called iteratively with the last predicted value
        predictions = self.model.predict(stock_data[-1].reshape(-1, 1))
        return np.array([predictions]).reshape(-1, 1)

class forestregressor(Model):
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=1000, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=1.0, n_jobs=-1)

    def train(self, stock_data, **kwargs):
        self.model.fit(stock_data[:-1].reshape(-1, 1), stock_data[1:].reshape(-1, ))

    def predict(self, stock_data, **kwargs):
        # it will be called iteratively with the last predicted value
        predictions = self.model.predict(stock_data[-1].reshape(-1, 1))
        return np.array([predictions]).reshape(-1, 1)

class knnregressor(Model):
    def __init__(self):
        self.model = KNeighborsRegressor(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=5, p=1, metric='minkowski')

    def train(self, stock_data, **kwargs):
        self.model.fit(stock_data[:-1].reshape(-1, 1), stock_data[1:].reshape(-1, ))

    def predict(self, stock_data, **kwargs):
        # it will be called iteratively with the last predicted value
        predictions = self.model.predict(stock_data[-1].reshape(-1, 1))
        return np.array([predictions]).reshape(-1, 1)

class gradientregressor(Model):
    def __init__(self):
        self.model = GradientBoostingRegressor(loss='huber', learning_rate=0.001, n_estimators=1000, subsample=1.0, min_samples_split=1, max_depth=None, max_features='sqrt')

    def train(self, stock_data, **kwargs):
        self.model.fit(stock_data[:-1].reshape(-1, 1), stock_data[1:].reshape(-1, ))

    def predict(self, stock_data, **kwargs):
        # it will be called iteratively with the last predicted value
        predictions = self.model.predict(stock_data[-1].reshape(-1, 1))
        return np.array([predictions]).reshape(-1, 1)

# AdaBoostRegressor
class adaboostregressor(Model):
    def __init__(self):
        self.model = AdaBoostRegressor(estimator=LinearRegression(), n_estimators=1000, learning_rate=0.1)

    def train(self, stock_data, **kwargs):
        self.model.fit(stock_data[:-1].reshape(-1, 1), stock_data[1:].reshape(-1, ))

    def predict(self, stock_data, **kwargs):
        # it will be called iteratively with the last predicted value
        predictions = self.model.predict(stock_data[-1].reshape(-1, 1))
        return np.array([predictions]).reshape(-1, 1)

from xgboost import XGBRegressor


class xgboostregressor(Model):
    def __init__(self):
        self.model = XGBRegressor(booster='dart', n_estimators=1000, max_depth=None, learning_rate=0.01, n_jobs=-1)

    def train(self, stock_data, **kwargs):
        self.model.fit(stock_data[:-1].reshape(-1, 1), stock_data[1:].reshape(-1, ))

    def predict(self, stock_data, **kwargs):
        # it will be called iteratively with the last predicted value
        predictions = self.model.predict(stock_data[-1].reshape(-1, 1))
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
        self.model.train(self.train_data, states=self.states)

    def predict(self):
        stock_prices = self.train_data
        self.predicted_prices = []
        for _ in range(self.days):
            last_price = self.model.predict(stock_prices, states=self.states)
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
    days = 60
    # markov_chain = MarkovChain() # RMSE: 15.213240208795463 R2: -4.097315833915814
    # mlp = MLP() # RMSE: 9.89007369818134 R2: -1.1542552211196977
    # forest = Forest() # RMSE: 9.675864725503247 R2: -1.0619478379518097
    # svm = SVM() # RMSE: 9.630344715677666 R2: -1.0425926470623281
    # logistic = Logistic() # RMSE: 7.712997064104738 R2: -0.310220621773984
    # sgd = SGD() # RMSE: 9.044707481858211 R2: -0.8017193798435545
    # knn = KNN() # RMSE: 9.675864725503247 R2: -1.0619478379518097
    # gradient = Gradient() # RMSE: 9.659312479842331 R2: -1.0548992328728777
    # gaussian = Gaussian() # RMSE: 9.72262340729911 R2: -1.0819247446449762
    
    # exponential = Exponential() # RMSE: 9.954503639153755 R2: -1.1824148991825063
    # arima = arima() # RMSE: 19.837721685306764 R2: -7.667257201684693
    # sarimax = sarimax() # RMSE: 10.008339334095794 R2: -1.2060844934029231
    # prophet = prophet() # RMSE: 14.943399344486815 R2: -3.9180948889578326
    # linear = Linear() # RMSE: 8.231859770802998 R2: -0.4924301902156174
    # elastic = Elastic() # RMSE: 7.093092026670268 R2: -0.10807533383844303
    # svr = svr() # RMSE: 9.823422832870722 R2: -1.125317285148427
    # sgdregressor = sgdregressor() # RMSE: 7.808880422005978 R2: -0.3429988582907477
    # lasso = lasso() # RMSE: 7.356684300210771 R2: -0.19196179990820061
    # ard = ARDregressor() # RMSE: 7.661828010168722 R2: -0.2928939292250241
    # gmm = GMMHMMModel() # RMSE: 16.07910290674938 R2: -4.694055919642484
    # gaussianregressor = gaussianregressor() # RMSE: 9.586332860267794 R2: -1.0239655113187616
    # forestregressor = forestregressor() # RMSE: 7.205268813051632 R2: -0.14340075478074166
    # knnregressor = knnregressor() # RMSE: 9.185412006195383 R2: -0.8582125148603632
    # gradientregressor = gradientregressor() # RMSE: 11.92090269269456 R2: -2.129798545522773
    # ada = adaboostregressor() # RMSE: 6.260918391178159 R2: 0.13667526425507082
    # xgb = xgboostregressor() # RMSE: 7.417285824083264 R2: -0.21168052625230027
    
    stock_predictor = StockPredictor(gaussianregressor, filepath, num_states, days)
    stock_predictor.preprocess_data()
    stock_predictor.get_states()
    stock_predictor.train()
    stock_predictor.predict()
    stock_predictor.evaluate()
    stock_predictor.plot_prediction()




# GridSearchCV
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
