
"""
This is a Python code that performs polynomial regression on Apple stock data and visualizes the predicted results. Here are the main steps:

Import the required libraries: matplotlib.pyplot, numpy, pandas, LinearRegression from sklearn.linear_model, and PolynomialFeatures from sklearn.preprocessing.
Read the stock data from a CSV file named apple_stock_data.csv and set the 'Date' column as the index.
Extract the 'Close' column from the data and split it into training and testing sets.
Prepare the training and testing sets for polynomial regression by converting the data into numpy arrays and reshaping them.
Fit a polynomial regression model of degree 3 on the training data using PolynomialFeatures and LinearRegression.
Predict the stock prices for the training and testing sets using the fitted model.
Visualize the predicted results using matplotlib.pyplot. The plot shows the actual stock prices for the training and testing sets as well as the predicted prices for the same periods.
Overall, the code aims to predict the future stock prices of Apple based on historical data using polynomial regression.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('apple_stock_data.csv', index_col='Date', parse_dates=True)

df = data[['Close']]

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train_data, test_data = df[:train_size], df[train_size:]

# Prepare the data for polynomial regression
X_train = np.array(range(train_size)).reshape((-1, 1))
y_train = train_data.values
X_test = np.array(range(train_size, len(df))).reshape((-1, 1))
y_test = test_data.values

# Create a pipeline for polynomial regression
model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                ('linear', LinearRegression())])

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Visualize the results
plt.figure(figsize=(16,8))
plt.title('AAPL stock price prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(df.index[:train_size], y_train, label='Training data')
plt.plot(df.index[:train_size], y_pred_train, label='Training predictions')
plt.plot(df.index[train_size:], y_test, label='Testing data')
plt.plot(df.index[train_size:], y_pred_test, label='Testing predictions')
plt.legend()
plt.show()

