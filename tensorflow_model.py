
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# Assume we have a dataset with population size (y) and various input variables (X)
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])
y = np.array([1, 2, 3, 4, 5])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001)

# Normalize the data
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)
y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_shape=(3,), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='relu')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

# Fit the model to the training data
model.fit(X_train, y_train, epochs=1000, batch_size=1024, validation_data=(X_test, y_test))

# Use the model to make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = tf.keras.losses.mean_squared_error(y_test, y_pred).numpy()
print(f"MSE: {mse}")

# Assume we have new data with various input variables (X_new)
X_new = np.array([[6, 7, 8]])

# Normalize the new data using the mean and standard deviation from the training data
X_new = (X_new - X_mean) / X_std

# Use the model to make predictions on the new data
y_new_pred = model.predict(X_new)

y_new_pred = y_new_pred * y_std + y_mean

print(f"Prediction for population size at time 6: {y_new_pred}")