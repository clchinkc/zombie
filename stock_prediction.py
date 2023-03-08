"""
import yfinance as yf

# Get the Apple stock data
apple = yf.Ticker("AAPL")
stock_data = apple.history(start="2022-01-01", end="2022-12-31")

# Save the data to a CSV file
stock_data.to_csv("apple_stock_data.csv")
"""

import matplotlib.pyplot as plt

# Load the stock data into a pandas DataFrame
import pandas as pd
import seaborn as sns

stock_data = pd.read_csv("apple_stock_data.csv")

# Set the index to the date column
stock_data = stock_data.set_index("Date")

# Plot the close column as a line graph
sns.lineplot(x=stock_data.index, y=stock_data["Close"])
plt.show()


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("apple_stock_data.csv")
df = df.set_index("Date")
# Select the "Close" column as the target
y = df["Close"]

# Select the "Open", "High", "Low", and "Volume" columns as the features
X = df[["Open", "High", "Low", "Volume"]]

# Split the data into a training set (80%) and a test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Print the score and mean squared error
print(f"Score: {model.score(X_test, y_test)}")
print(f"Mean squared error: {mean_squared_error(y_test, predictions)}")

# Print the model's coefficients and intercept
# print(f"Coefficients: {model.coef_}")
# print(f"Intercept: {model.intercept_}")

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


# https://neptune.ai/blog/predicting-stock-prices-using-machine-learning
# https://www.youtube.com/watch?v=WcfKaZL4vpA
# stock prediction google search