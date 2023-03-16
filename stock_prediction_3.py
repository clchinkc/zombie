
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


from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


# Step 1: Data Preprocessing
def preprocess_data(filepath):
    # Read data from CSV file
    stock_data = pd.read_csv(filepath)
    # Drop any missing values
    stock_data = stock_data.dropna()
    # stock_data = stock_data[['Close']]
    scaler = MinMaxScaler()
    stock_data = scaler.fit_transform(stock_data[['Close']])
    return stock_data, scaler

# Step 2: Defining the States
def get_states(stock_data, num_states):
    # Use KMeans clustering to group the scaled Close prices into the specified number of clusters (i.e., states)
    kmeans = KMeans(n_clusters=num_states, init='k-means++', n_init=1, max_iter=300).fit(stock_data)
    states = kmeans.labels_
    return states

def markov_chain_prediction(stock_data, states):
    # Step 3: Transition Probability Matrix
    hist = np.histogram2d(states[:-1], states[1:], bins=len(set(states)))[0]
    transition_prob_matrix = hist / hist.sum(axis=1, keepdims=True)

    # Step 4: Future State Probabilities
    current_state_prob = np.array([1/len(set(states))]*len(set(states)))
    future_state_prob = np.dot(current_state_prob, transition_prob_matrix)

    # Step 5: Predicting the Future Prices
    # weighted average of the median prices of all the future states
    future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
    estimated_future_price = np.dot(future_state_prob, future_prices)
    estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
    return estimated_future_price_2d

def mlp_prediction(stock_data, states):
    # Step 4: Train a neural network classifier
    clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, solver='adam', random_state=42, early_stopping=True)
    clf.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    # Step 5: Predict the future states of the last data point
    proba = clf.predict_proba(stock_data[-1].reshape(-1, 1))
    # weighted average of the median prices of all the future states
    future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
    estimated_future_price = np.dot(proba[-1], future_prices)
    estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
    return estimated_future_price_2d

def forest_prediction(stock_data, states):
    # Step 4: Train a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    clf.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    # Step 5: Predict the future states of the last data point
    proba = clf.predict_proba(stock_data[-1].reshape(-1, 1))
    # weighted average of the median prices of all the future states
    future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
    estimated_future_price = np.dot(proba[-1], future_prices)
    estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
    return estimated_future_price_2d

def svm_prediction(stock_data, states):
    # Step 4: Train a SVM classifier
    clf = SVC(gamma='auto', probability=True)
    clf.fit(stock_data[:-1].reshape(-1, 1), states[1:])

    # Step 5: Predict the future states of the last data point
    proba = clf.predict_proba(stock_data[-1].reshape(-1, 1))
    # weighted average of the median prices of all the future states
    future_prices = [np.median(stock_data[states == i]) for i in range(len(set(states)))]
    estimated_future_price = np.dot(proba[-1], future_prices)
    estimated_future_price_2d = np.array([estimated_future_price]).reshape(-1, 1)
    return estimated_future_price_2d

def get_predictions(prediction_method, stock_data, states, days=30):
    # Predict the next 30 days using the Markov chain model
    stock_prices = stock_data
    predictions = []
    for i in range(days):
        # Make a new prediction for the next day
        last_price = prediction_method(stock_prices, states)
        stock_prices = np.append(stock_prices, last_price)[1:]
        predictions.append(last_price)
    return predictions

def plot_predictions(scaler, stock_data, predictions):
    # Plot the predicted prices along with the historical data
    historical_prices = scaler.inverse_transform(stock_data)
    predicted_price = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    dates = pd.read_csv('apple_stock_data.csv')['Date'].values[-len(historical_prices):]
    dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None) for date in dates]
    future_dates = pd.date_range(start=dates[-1], periods=30, freq='D')

    plt.plot(dates, historical_prices, label='Historical Prices')
    plt.plot(future_dates, predicted_price, label='Predicted Prices')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
    
def main():
    stock_data, scaler = preprocess_data('apple_stock_data.csv')
    states = get_states(stock_data, 5)
    
    # Predict the next days using the Markov chain model and the MLP model
    predictions_markov = get_predictions(markov_chain_prediction, stock_data, states, days=30)
    predictions_mlp = get_predictions(mlp_prediction, stock_data, states, days=30)
    predictions_forest = get_predictions(forest_prediction, stock_data, states, days=30)
    predictions_svm = get_predictions(svm_prediction, stock_data, states, days=30)
    
    # get the average of the two predictions
    predictions = (np.array(predictions_markov) + np.array(predictions_mlp) + np.array(predictions_forest) + np.array(predictions_svm)) / 4
    # predictions = predictions_svm
    
    # Plot the predicted prices along with the historical data
    plot_predictions(scaler, stock_data, predictions)

if __name__ == "__main__":
    main()


"""
Add volatility term:
To add volatility term, you can use the volatility of the stock price to adjust the predicted price. For example, you can use the standard deviation of the stock price to estimate the volatility.

Use a more sophisticated clustering algorithm:
To use a more sophisticated clustering algorithm, you can try hierarchical clustering, density-based clustering, or model-based clustering algorithms and evaluate their performance.

Use more advanced machine learning models:
To use more advanced machine learning models, you can try neural networks, support vector machines, decision trees, and random forests and evaluate their performance. You can also use ensemble methods such as stacking or boosting to combine the predictions of multiple models.

Consider more features:
To consider more features, you can collect and preprocess additional data such as volume, news sentiment, technical indicators, and economic indicators. You can then use feature selection methods such as correlation analysis, mutual information, or LASSO regression to select the most relevant features for the model.

Use a different time scale:
To use a different time scale, you can adjust the time interval of the data used for training and testing the model. For example, you can use hourly or weekly data instead of daily data.

Evaluate and fine-tune the model:
To evaluate and fine-tune the model, you can use cross-validation techniques such as k-fold cross-validation or time series cross-validation to estimate the performance of the model. You can also use hyperparameter tuning techniques such as grid search, random search, or Bayesian optimization to optimize the parameters of the model.
"""
