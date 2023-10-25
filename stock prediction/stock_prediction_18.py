import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Fetch data from yfinance
tickers_list = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA', 'META', 'NVDA', 'NFLX']
data_frames = []

for ticker in tickers_list:
    stock = yf.Ticker(ticker)
    df = stock.history(period="60d")
    df['Ticker'] = ticker
    data_frames.append(df)

combined_data = pd.concat(data_frames)

# Selecting features
selected_features = ['Close', 'Open', 'Volume', 'High', 'Low', 'Dividends', 'Stock Splits']

# Add Ticker column for reference and filter out later
df = combined_data[['Ticker'] + selected_features]

# Data Cleanup
df = df.dropna(subset=selected_features)

# Data Normalization
autoscaler = StandardScaler()
df_normalized = df.copy()
df_normalized[selected_features] = autoscaler.fit_transform(df_normalized[selected_features])

# Clustering
kmeans = KMeans(n_clusters=3, n_init="auto")
df_normalized['Cluster'] = kmeans.fit_predict(df_normalized[selected_features])

print(df_normalized['Cluster'].value_counts())

# LSTM Integration
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

SEQ_LENGTH = 10
X, y = create_sequences(df_normalized[selected_features + ['Cluster']].values, SEQ_LENGTH)

# Building LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.Dense(len(selected_features) + 1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, shuffle=False)

# Predicting the next day's price
last_sequence = df_normalized[selected_features + ['Cluster']].values[-SEQ_LENGTH:]
predicted_price = model.predict(last_sequence.reshape(1, SEQ_LENGTH, len(selected_features) + 1))
predicted_price_transformed = autoscaler.inverse_transform(predicted_price[:, :7])
print("Predicted price:", predicted_price_transformed)

# Predicting the next day's price for each cluster
for cluster in range(3):
    last_sequence = df_normalized[df_normalized['Cluster'] == cluster][selected_features + ['Cluster']].values[-SEQ_LENGTH:]
    if last_sequence.shape[0] != SEQ_LENGTH:
        print("Cluster", cluster, "doesn't have enough data for prediction.")
        continue
    predicted_price = model.predict(last_sequence.reshape(1, SEQ_LENGTH, len(selected_features) + 1))
    predicted_price_transformed = autoscaler.inverse_transform(predicted_price[:, :7])
    print("Cluster", cluster, "predicted price:", predicted_price_transformed)

# Predicting the next day's price for each stock
for ticker in tickers_list:
    last_sequence = df_normalized[df_normalized['Ticker'] == ticker][selected_features + ['Cluster']].values[-SEQ_LENGTH:]
    if last_sequence.shape[0] != SEQ_LENGTH:
        print("Ticker", ticker, "doesn't have enough data for prediction.")
        continue
    predicted_price = model.predict(last_sequence.reshape(1, SEQ_LENGTH, len(selected_features) + 1))
    predicted_price_transformed = autoscaler.inverse_transform(predicted_price[:, :7])
    print("Ticker", ticker, "predicted price:", predicted_price_transformed)
