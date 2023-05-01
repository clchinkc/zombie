
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load the data into a pandas dataframe
df = pd.read_csv('apple_stock_data.csv')

# Extract the relevant columns
X = df[['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']]
y = df['Close']

# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_std)

np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

# Print the explained variance ratio for each principal component
print(pca.explained_variance_ratio_)

# Select the first 4 principal components
X_pca = X_pca[:, :4]

# Train a linear regression model using the principal components
lr = LinearRegression()
lr.fit(X_pca, y)

# Make predictions on new data
new_X = [[123.45, 125.67, 122.34, 1000000, 0.25, 0]]
new_X_pca = pca.transform(new_X)
new_X_pca = new_X_pca[:, :4]
y_pred = lr.predict(new_X_pca)
print(y_pred)

# Back-transform the coefficients using the loadings matrix of the PCA
loadings = pca.components_[:4, :]
coefficients = lr.coef_
back_transformed = coefficients.dot(loadings)

# Print the back-transformed coefficients
print(back_transformed)