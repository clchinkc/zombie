
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# Download stock data
data = pd.read_csv('apple_stock_data.csv', parse_dates=['Date'], index_col='Date')
prices = data['Close'].to_numpy()

# Calculate the price difference
price_diff = np.diff(prices)

# Scale the price difference to have zero mean and unit variance
scaler = StandardScaler()
price_diff_scaled = scaler.fit_transform(price_diff.reshape(-1, 1))

# Create and fit the Hidden Markov Model
num_hidden_states = 3
model = hmm.GaussianHMM(n_components=num_hidden_states, covariance_type='diag', n_iter=1000)
model.fit(price_diff_scaled)

# Find the state sequence with the highest likelihood
state_sequence = model.predict(price_diff_scaled)

# Predict the next day's price difference
next_day_state = np.argmax(model.transmat_[state_sequence[-1], :])
next_day_diff = model.means_[next_day_state][0]

# De-scale the predicted price difference
next_day_diff = scaler.inverse_transform(next_day_diff.reshape(-1, 1))

# Predict the next day's stock price
next_day_price = prices[-1] + next_day_diff

print(f"Next day's predicted stock price: {next_day_price[0]}")
