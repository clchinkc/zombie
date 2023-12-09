import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
data = pd.read_csv('apple_stock_data.csv')

# Feature engineering 
data['MA_5'] = data['Close'].rolling(window=5, min_periods=1).mean()
data['MA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
data['Volatility'] = data['Close'].rolling(window=5, min_periods=1).std()
data['High_Low_Spread'] = data['High'] - data['Low']
data['Daily_Return'] = data['Close'].pct_change()
data = data.dropna()

# Normalize data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data[['Close', 'MA_5', 'MA_20', 'Volatility', 'High_Low_Spread', 'Daily_Return']])
scaler_close = MinMaxScaler()
prices_normalized = scaler_close.fit_transform(data['Close'].values.reshape(-1, 1)).flatten()

# Prepare data for LSTM
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), :]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# Create datasets for two time scales
look_back_short = 5  # Shorter time scale
look_back_long = 20  # Longer time scale

X_short, y_short = create_dataset(data_normalized, look_back_short)
X_long, y_long = create_dataset(data_normalized, look_back_long)

# Determine the minimum length between the two datasets
min_length = min(len(X_short), len(X_long))

# Trim the longer dataset
X_short = X_short[:min_length]
y_short = y_short[:min_length]
X_long = X_long[:min_length]
y_long = y_long[:min_length]

# Now split data into training and testing sets for both time scales
X_short_train, X_short_test, y_short_train, y_short_test = train_test_split(X_short, y_short, test_size=0.2, random_state=42)
X_long_train, X_long_test, y_long_train, y_long_test = train_test_split(X_long, y_long, test_size=0.2, random_state=42)

# Build and train LSTM model
from tensorflow.keras.layers import LSTM, Concatenate, Dense, Input
from tensorflow.keras.models import Model

# Define LSTM layers for both time scales
input_short = Input(shape=(look_back_short, data_normalized.shape[1]))
input_long = Input(shape=(look_back_long, data_normalized.shape[1]))

lstm_short = LSTM(50, return_sequences=True)(input_short)
lstm_short = LSTM(50)(lstm_short)

lstm_long = LSTM(50, return_sequences=True)(input_long)
lstm_long = LSTM(50)(lstm_long)

# Concatenate the outputs from both LSTMs
concat = Concatenate()([lstm_short, lstm_long])

# Dense layers following the concatenated outputs
output = Dense(25, activation='relu')(concat)
output = Dense(1)(output)

# Define the model
model = Model(inputs=[input_short, input_long], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on data from both time scales
history = model.fit([X_short_train, X_long_train], y_short_train, 
                    epochs=10, batch_size=32, 
                    validation_data=([X_short_test, X_long_test], y_short_test))

# Evaluating model performance
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.show()

print(f'Train Loss: {train_loss[-1]}')
print(f'Validation Loss: {val_loss[-1]}')

# Use the Time Fusion LSTM model to predict prices
predicted_prices_normalized_short = model.predict([X_short, X_long]).flatten()
predicted_prices_short = scaler_close.inverse_transform(predicted_prices_normalized_short.reshape(-1, 1)).flatten()

# We use the short time scale predictions for the dynamic programming algorithm
predicted_prices = predicted_prices_short

# Dynamic Programming for maximizing profit
def maxProfit(k, prices):
    if prices.size == 0: return 0, [], []

    n = len(prices)
    if k >= n // 2:
        return sum(max(prices[i + 1] - prices[i], 0) for i in range(n - 1)), [(i, i + 1) for i in range(n - 1) if prices[i + 1] > prices[i]], []

    profits = [[0] * n for _ in range(k + 1)]
    for j in range(1, k + 1):
        max_profit_so_far = -prices[0]
        for i in range(1, n):
            profits[j][i] = max(profits[j][i - 1], prices[i] + max_profit_so_far)
            max_profit_so_far = max(max_profit_so_far, profits[j - 1][i] - prices[i])

    # Reconstruct transactions (can be optimized based on specific needs)
    transactions = []
    for j in range(k, 0, -1):
        for i in range(n - 1, 0, -1):
            if profits[j][i] > profits[j][i - 1]:
                transactions.append((i - 1, i))
                break

    return profits[k][-1], transactions, profits


k = 2
profit, transactions, profit_matrix = maxProfit(k, predicted_prices)
print(f'Maximum profit with {k} transactions: {profit}')

# Profit Breakdown
for idx, (buy, sell) in enumerate(transactions):
    transaction_profit = predicted_prices[sell] - predicted_prices[buy]
    print(f'Transaction {idx + 1}: Buy on day {buy}, Sell on day {sell}, Profit: {transaction_profit}')

# Visualization of Predicted Prices
plt.figure(figsize=(15, 6))
plt.plot(prices_normalized, label='Actual Prices', color='blue')
plt.plot(np.arange(look_back_short, len(predicted_prices) + look_back_short), predicted_prices_normalized_short, label='Predicted Prices', color='red')
plt.legend()
plt.title('Actual vs Predicted Prices')
plt.xlabel('Days')
plt.ylabel('Normalized Prices')
plt.show()

# Visualize Intermediate Profit Matrix
plt.figure(figsize=(15, 6))
sns.heatmap(profit_matrix, annot=True, fmt=".1f")
plt.title('Profit Matrix Heatmap')
plt.show()

# Visualize Profit over Time
plt.figure(figsize=(15, 6))
for j in range(1, k + 1):
    plt.plot(profit_matrix[j], label=f'Transaction {j}')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Profit')
plt.title('Profit over Time per Transaction')
plt.show()

"""
Combining traditional algorithmic paradigms like Divide and Conquer (D&C) or Dynamic Programming (DP) with modern Deep Learning (DL) methods can lead to innovative solutions for complex problems. Here's how they could be potentially combined in the context of stock market analysis:

1. **Feature Extraction and Representation Learning:**
   - Deep learning can be employed to learn feature representations from raw stock market data, which can be used as inputs for D&C or DP algorithms. For instance, a deep learning model could process historical stock price data to extract features that capture underlying market trends, which could then be fed into a D&C or DP algorithm to solve a particular problem like maximizing stock return.

2. **Policy Optimization:**
   - In scenarios where the goal is to devise a trading strategy to maximize returns, Reinforcement Learning (a subset of DL) can be employed. The DP or D&C algorithms can be used to formulate the state transition dynamics which the reinforcement learning agent needs to learn. The agent can learn the optimal policy to maximize returns over time through interaction with the environment (the stock market in this scenario).

3. **Model-based Approaches:**
   - Deep learning models can be used to learn a predictive model of stock prices, which can then be utilized within a D&C or DP framework to solve optimization problems related to stock trading. For example, a DL model could forecast future stock prices, which could then be used by a DP algorithm to determine the optimal times to buy and sell stocks to maximize returns.

4. **Hybrid Algorithms:**
   - Creating hybrid algorithms that integrate the strengths of D&C or DP with DL can also be a novel approach. For example, a D&C or DP algorithm could be used to narrow down the search space or break down the problem into sub-problems, and then deep learning could be employed to solve each sub-problem or optimize within the narrowed search space.

5. **Improving Computational Efficiency:**
   - Deep Learning could be employed to learn the solutions to sub-problems in a D&C or DP framework, potentially speeding up the computation by avoiding the need to solve the same sub-problems repeatedly. This is particularly relevant to DP, where memoization is a key feature.

6. **End-to-End Learning:**
   - In an end-to-end learning setup, a deep learning model could be trained to learn an optimal policy for trading stocks directly from raw data, with the D&C or DP algorithms embedded within the training loop to provide structural guidance or to handle certain aspects of the problem that are well-suited to algorithmic solution.

By creating a synergistic relationship between traditional algorithmic paradigms and modern deep learning techniques, it's possible to tackle complex, real-world problems in the stock market domain with a higher level of sophistication and effectiveness.
"""
