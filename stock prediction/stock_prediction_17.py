import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load and preprocess data
data = pd.read_csv('apple_stock_data.csv')

# Feature engineering (e.g., moving averages, volatility, etc.)
data['MA_5'] = data['Close'].rolling(window=5).mean()
data['MA_20'] = data['Close'].rolling(window=20).mean()
data['Volatility'] = data['Close'].rolling(window=5).std()
data['High_Low_Spread'] = data['High'] - data['Low']
data['Daily_Return'] = data['Close'].pct_change()

# Dropping rows with NaN values resulted from rolling computations
data = data.dropna()

# Normalize data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data[['Close', 'MA_5', 'MA_20', 'Volatility', 'High_Low_Spread', 'Daily_Return']])

# Separate scaler for 'Close' prices
scaler_close = MinMaxScaler()
prices_normalized = scaler_close.fit_transform(data['Close'].values.reshape(-1, 1)).flatten()

# Prepare data for LSTM
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), :]
        X.append(a)
        Y.append(data[i + look_back, 0])  # Assuming 'Close' price is the target variable
    return np.array(X), np.array(Y)

look_back = 5
X, y = create_dataset(data_normalized, look_back)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Build and train LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluating model performance
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Optionally, plot the loss values for better visualization
# import matplotlib.pyplot as plt
# plt.plot(train_loss, label='Train Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend()
# plt.show()

# Step 3: Use LSTM to predict prices
predicted_prices_normalized = model.predict(X.reshape(X.shape[0], X.shape[1], X.shape[2])).flatten()
predicted_prices = scaler_close.inverse_transform(predicted_prices_normalized.reshape(-1, 1)).flatten()

# Dynamic Programming for maximizing profit
def maxProfit(k, prices):
    if len(prices) == 0:
        return 0, [], []

    n = len(prices)
    if k >= n // 2:
        transactions = [(i, i+1) for i in range(n - 1) if prices[i + 1] > prices[i]]
        return sum(max(prices[i + 1] - prices[i], 0) for i in range(n - 1)), transactions, []

    profits = [[0] * n for _ in range(k + 1)]
    transactions = [[[] for _ in range(n)] for _ in range(k + 1)]

    for j in range(1, k + 1):
        max_profit_so_far = -prices[0]
        max_profit_so_far_index = 0
        for i in range(1, n):
            if prices[i] + max_profit_so_far > profits[j][i - 1]:
                profits[j][i] = prices[i] + max_profit_so_far
                transactions[j][i] = transactions[j - 1][i].copy()
                transactions[j][i].append((max_profit_so_far_index, i))
            else:
                profits[j][i] = profits[j][i - 1]
                transactions[j][i] = transactions[j][i - 1].copy()

            if max_profit_so_far < profits[j - 1][i] - prices[i]:
                max_profit_so_far = profits[j - 1][i] - prices[i]
                max_profit_so_far_index = i

    return profits[k][-1], transactions[k][-1], profits

# Call the function
k = 2
profit, transactions, profit_matrix = maxProfit(k, predicted_prices)
print(f'Maximum profit with {k} transactions: {profit}')

# Profit Breakdown
for idx, (buy, sell) in enumerate(transactions):
    transaction_profit = predicted_prices[sell] - predicted_prices[buy]
    print(f'Transaction {idx + 1}: Buy on day {buy}, Sell on day {sell}, Profit: {transaction_profit}')

# Visualize Intermediate Profit Matrix
sns.heatmap(profit_matrix, annot=True, fmt=".1f")
plt.show()

# Visualize Profit over Time
for j in range(1, k + 1):
    plt.plot(profit_matrix[j], label=f'Transaction {j}')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Profit')
plt.title('Profit over Time per Transaction')
plt.show()
