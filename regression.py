import numpy as np

# Assume we have a dataset with population size (y) and various input variables (X)
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])
y = np.array([1, 2, 3, 4, 5])

# Initialize the model parameters
w = np.zeros(3)
b = 0

# Set the learning rate
alpha = 0.001

# Define the mean squared error loss function
def mse(y, y_pred):
    return ((y - y_pred) ** 2).mean()

# Define the gradient descent update rule
def update_params(w, b, X, y, y_pred, alpha):
    w_grad = -2 * X.T.dot(y - y_pred) / len(y)
    b_grad = -2 * (y - y_pred).mean()
    w -= alpha * w_grad
    b -= alpha * b_grad
    return w, b

# Define the prediction function
def predict(w, b, X):
    return X.dot(w) + b

# Fit the model to the data
for i in range(10000):
    y_pred = predict(w, b, X)
    w, b = update_params(w, b, X, y, y_pred, alpha)
    if i % 10 == 0:
        print(f"Iteration {i}: MSE = {mse(y, y_pred)}")

# Use the model to make predictions on new data
X_new = np.array([[6, 7, 8]])
y_new_pred = predict(w, b, X_new)
print(f"Prediction for population size at time 6: {y_new_pred}")














