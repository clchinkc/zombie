import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.cluster import KMeans


# Function to load and normalize data
def load_and_normalize_data(file_path):
    try:
        data = pd.read_csv(file_path, usecols=['Open', 'High', 'Low', 'Close', 'Volume'])
        # Normalizing data
        normalized_data = (data - data.mean()) / data.std()
        return normalized_data
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None

# Function to create and train the SOM
def create_and_train_som(data, map_size=(5, 5), sigma=1.0, learning_rate=0.5, num_epochs=100):
    som = MiniSom(map_size[0], map_size[1], data.shape[1], sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(data.values)
    som.train_random(data.values, num_epochs)
    return som

# Function to plot the SOM
def plot_som(som, data, labels=None):
    plt.figure(figsize=(10, 10))
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.colorbar()
    for i, x in enumerate(data.values):
        w = som.winner(x)
        plt.plot(w[0]+0.5, w[1]+0.5, 'o', markerfacecolor='None', markeredgecolor='red', markersize=6)
        if labels is not None:
            plt.text(w[0]+0.5, w[1]+0.5, labels[i], color='red', ha='center', va='center')
    plt.title('SOM Distance Map')
    plt.xlabel('SOM X-Dimension')
    plt.ylabel('SOM Y-Dimension')
    plt.show()

# Function to plot heatmaps for each feature
def plot_feature_heatmaps(som, data):
    n_features = data.shape[1]
    fig, axs = plt.subplots(nrows=1, ncols=n_features, figsize=(20, 5))
    for i, ax in enumerate(axs):
        weights = som.get_weights()[:,:,i]
        ax.pcolor(weights, cmap='coolwarm')
        ax.set_title(data.columns[i])
        ax.set_xlabel('SOM X-Dimension')
        ax.set_ylabel('SOM Y-Dimension')
    plt.tight_layout()
    plt.show()

# Function to analyze clusters
def analyze_clusters(som, data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(som.get_weights().reshape(-1, data.shape[1]))
    cluster_labels = kmeans.labels_
    return cluster_labels.reshape(som.get_weights().shape[:2])

# Function to print cluster explanation
def print_cluster_explanation(cluster_labels):
    unique_clusters = np.unique(cluster_labels)
    print("Cluster Analysis:")
    for cluster in unique_clusters:
        count = np.count_nonzero(cluster_labels == cluster)
        print(f"Cluster {cluster}: contains {count} nodes, representing similar stock trading patterns.")

# Main Function
def main():
    data = load_and_normalize_data('apple_stock_data.csv')
    if data is not None:
        som = create_and_train_som(data)
        plot_som(som, data)
        plot_feature_heatmaps(som, data)
        cluster_labels = analyze_clusters(som, data)
        print_cluster_explanation(cluster_labels)

if __name__ == "__main__":
    main()
