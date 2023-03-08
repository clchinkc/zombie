
"""
import numpy as np
import pandas as pd

# Step 1: Data Preprocessing
stock_data = pd.read_csv('apple_stock_data.csv')
# get close price
prices = stock_data['Close'].values

# Define the transition matrix for the Markov chain
transition_matrix = np.array([[0.7, 0.3],
                            [0.4, 0.6]])

# Define the initial state distribution
initial_state = np.array([0.5, 0.5])

# Define the number of steps to predict
num_steps = 5

# Define a function to predict future prices using the Markov chain
def predict_stock_price(prices, transition_matrix, initial_state, num_steps):
    current_state = initial_state
    predicted_prices = []
    for i in range(num_steps):
        predicted_price = np.dot(current_state, prices)
        predicted_prices.append(predicted_price)
        current_state = np.dot(current_state, transition_matrix)
    return predicted_prices

# Call the function to predict the future stock prices
predicted_prices = predict_stock_price(prices, transition_matrix, initial_state, num_steps)

# Print the predicted prices
print(predicted_prices)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Step 1: Data Preprocessing
stock_data = pd.read_csv('apple_stock_data.csv')
# Drop any missing values
stock_data = stock_data.dropna()
# stock_data = stock_data[['Close']]
scaler = MinMaxScaler()
stock_data = scaler.fit_transform(stock_data[['Close']])

# Step 2: Defining the States
num_states = 5
kmeans = KMeans(n_clusters=num_states, n_init=10, max_iter=1000).fit(stock_data)
states = kmeans.labels_

# Step 3: Transition Probability Matrix
transition_prob_matrix = np.zeros((num_states, num_states))
for i in range(len(states)-1):
    transition_prob_matrix[states[i]][states[i+1]] += 1
transition_prob_matrix /= transition_prob_matrix.sum(axis=1)[:, np.newaxis]

# Step 4: Predicting the Future States
current_state_prob = np.array([1/num_states]*num_states)
future_state_prob = np.dot(current_state_prob, transition_prob_matrix)

# Step 5: Predicting the Future Prices
# weighted average of the median prices of all the future states
future_prices = [np.median(stock_data[states == i]) for i in range(num_states)]
estimated_future_price = np.dot(future_state_prob, future_prices)
# median of prices of the most probable future state
# future_price = np.median(stock_data[states == np.argmax(future_state_prob)])
# estimated_future_price = sum(np.dot(future_state_prob, future_price))
estimated_future_price_2d = np.array([estimated_future_price]).reshape(1, -1)
predicted_price = scaler.inverse_transform(estimated_future_price_2d)
print(predicted_price[0][0])



from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

# Step 5: Predicting the Future Prices
X_train = stock_data[:-1]
y_train = states[:-1]
X_test = stock_data[1:]
y_test = states[1:]

# Train a neural network classifier
clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
clf.fit(X_train, y_train)

# Predict the future states
y_pred = clf.predict(X_test)

# Reverse the scaling of the input data
X_test_scaled = scaler.inverse_transform(X_test)

# Reverse the scaling of the predicted states
y_pred_scaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Generate the predicted stock prices
predictions = [X_test_scaled[0][0]]
for i in range(1, len(X_test_scaled)):
    if y_pred_scaled[i] == 1:
        predictions.append(predictions[i-1] * (1 + volatility))
        raise ValueError('The predicted state is 1')
    elif y_pred_scaled[i] == 2:
        predictions.append(predictions[i-1] * (1 - volatility))
        raise ValueError('The predicted state is 2')
    else:
        predictions.append(np.median(X_test_scaled[y_pred == y_pred[i]]))
        
# Print the predictions for the next day
print(predictions[-1])
        
# add volatility term



"""
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
