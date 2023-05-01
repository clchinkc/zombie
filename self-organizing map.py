
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minisom import MiniSom

# Load data
data = pd.read_csv('apple_stock_data.csv', usecols=['Open', 'High', 'Low', 'Close', 'Volume']).values

# Normalize data
data = (data - data.mean(axis=0)) / data.std(axis=0)

# Set SOM parameters
map_size = (5, 5)
sigma = 1.0
learning_rate = 0.5
num_epochs = 100

# Create SOM
som = MiniSom(*map_size, data.shape[1], sigma=sigma, learning_rate=learning_rate)
som.random_weights_init(data)
som.train_random(data, num_epochs)

# Plot SOM
plt.figure(figsize=(map_size[0], map_size[1]))
plt.pcolor(som.distance_map().T, cmap='bone_r') # distance map as background
plt.colorbar()

# Add data points as markers
marker_size = 5
marker_color = 'black'
for i, x in enumerate(data):
    w = som.winner(x)
    plt.plot(w[0]+0.5, w[1]+0.5, marker='o', markerfacecolor='None',
            markersize=marker_size, markeredgecolor=marker_color,
            markeredgewidth=0.5)

plt.show()

# Identify clusters and patterns in the data
mapped_data = np.array([som.winner(x) for x in data])
cluster_labels = np.unique(mapped_data, axis=0)

print("Cluster labels: ", cluster_labels)

# Extract features from SOM weights
features = som.get_weights().reshape(map_size[0]*map_size[1], data.shape[1])
feature_labels = np.arange(features.shape[0])

print("Features: ", features)
print("Feature labels: ", feature_labels)