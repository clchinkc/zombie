
import numpy as np
import pandas as pd


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def denormalize(data, reference):
    return data * (np.max(reference) - np.min(reference)) + np.min(reference)

def grey_relational_coefficient(reference, comparison):
    abs_diff = np.abs(reference - comparison)
    max_diff = np.max(abs_diff)
    min_diff = np.min(abs_diff)
    return (min_diff + 0.5 * max_diff) / (abs_diff + 0.5 * max_diff + 1e-10)

def grey_relational_analysis(reference_sequence, comparison_sequences):
    normalized_reference = normalize(reference_sequence)
    normalized_comparisons = np.array([normalize(sequence) for sequence in comparison_sequences])

    coefficients = np.array([grey_relational_coefficient(normalized_reference, sequence) for sequence in normalized_comparisons])

    weights = coefficients.sum(axis=0) / coefficients.sum()

    aggregated_sequence = np.average(normalized_comparisons, axis=1, weights=weights)
    
    denormalized_aggregated_sequence = denormalize(aggregated_sequence, reference_sequence)

    print("Weights:", weights)
    print("Coefficients:", coefficients)
    print("Aggregated sequence:", aggregated_sequence)

    return denormalized_aggregated_sequence


# Load data from CSV file
data = pd.read_csv('apple_stock_data.csv')

# Extract the 'Close' column as the reference sequence
reference_sequence = data['Close'].values

# Extract other columns or time periods as comparison sequences
comparison_sequences = [
    data['Close'].shift(1, fill_value=np.mean(reference_sequence)).dropna().values,  # Previous day's closing price
    data['Close'].rolling(window=5, min_periods=1).mean().dropna().values,  # 5-day moving average
]

# Perform grey relational analysis
prediction = grey_relational_analysis(reference_sequence, comparison_sequences)

print("Predicted sequence:", prediction)




"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv('apple_stock_data.csv', index_col='Date', parse_dates=True)


# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df['Close'].values.reshape(-1,1))


def grey_relational_grading(x0, xi, rho=0.5):
    # Absolute difference sequence
    abs_diff_sequence = np.abs(x0 - xi)

    # Maximum and minimum
    max_val = np.max(abs_diff_sequence)
    min_val = np.min(abs_diff_sequence)

    # Grey Relational Coefficient
    grey_relational_coefficient = (min_val + (rho * max_val)) / (abs_diff_sequence + (rho * max_val))

    # Grey Relational Grade
    grey_relational_grade = np.mean(grey_relational_coefficient)

    return grey_relational_grade

# The first row of df_scaled is used as x0
x0 = df_scaled[0]

# Initialize a list to hold the grey relational grades
grey_relational_grades = []

# Calculate the grey relational grade for each sequence
for i in range(1, len(df_scaled)):
    xi = df_scaled[i]
    grey_relational_grades.append(grey_relational_grading(x0, xi))

from sklearn.linear_model import LinearRegression

# Create a linear regression model
model = LinearRegression()

# Reshape the grey relational grades to a 2D array
X = np.array(grey_relational_grades[:-1]).reshape(-1,1)

# The target variable is the next day's grey relational grade
y = grey_relational_grades[1:]

# Fit the model
model.fit(X, y)

# Use the model to predict the grey relational grade for the next day
next_day_grey_relational_grade = model.predict(X[-1].reshape(-1,1))

# Denormalize the prediction to get the actual stock price
predicted_price = scaler.inverse_transform(next_day_grey_relational_grade.reshape(-1,1))

print('Predicted price: %.2f' % predicted_price)

# Calculate the error
error = np.abs(df['Close'][-1] - predicted_price)
print('Error: %.2f' % error)
"""

"""
import numpy as np
import pandas as pd
from cv2 import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('apple_stock_data.csv', index_col='Date', parse_dates=True)

# Normalize the data
def normalization(data):
    # Normalize data to (0,1)
    min_data = np.min(data)
    max_data = np.max(data)
    normalized_data = (data - min_data) / (max_data - min_data)
    return normalized_data, min_data, max_data

df_scaled, min_data, max_data = normalization(df['Close'].values.reshape(-1,1))

def greyRelationalAnalysis(data, primary_data):
    max_val = np.max(data)
    min_val = np.min(data)
    primary_data = np.array(primary_data)
    data = np.array(data)
    grey_relational_coefficient = (min_val + 0.5 * max_val) / (data + 0.5 * max_val)
    grey_relational_coefficient = np.min(grey_relational_coefficient) / grey_relational_coefficient
    return grey_relational_coefficient

# The first row of df_scaled is used as primary_data
primary_data = df_scaled[0]

# Calculate grey relational coefficients
gra_coefficients = greyRelationalAnalysis(df_scaled, primary_data)

# Split the data into training and testing sets
train_size = int(len(gra_coefficients) * 0.7)
train, test = gra_coefficients[0:train_size], gra_coefficients[train_size:len(gra_coefficients)]

# Reshape the data to fit the model
train = train.reshape(-1,1)
test = test.reshape(-1,1)

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(train[:-1], train[1:])

# Make predictions
predictions = model.predict(test[:-1])

# Calculate the error
error = mean_squared_error(test[1:], predictions)
print('Test MSE: %.3f' % error)

# Print the actual prices and predictions
print('Last actual coefficient: ', test[-1])
print('Last predicted coefficient: ', predictions[-1])

# Inverse normalization
predicted_price = (predictions[-1] * (max_data - min_data)) + min_data

# Print the actual prices and predictions
print('Last actual price: ', df['Close'][-1])
print('Predicted price: %.2f' % predicted_price)
"""

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalization(data):
    # Normalize data to (0,1)
    min_data = np.min(data)
    max_data = np.max(data)
    return (data - min_data) / (max_data - min_data)

def grey_relational_coefficient(data, primary_data, rho=0.5):
    # Calculate grey relational coefficient
    data = np.abs(data - primary_data)
    max_data = np.max(data)
    min_data = np.min(data)
    return (min_data + rho * max_data) / (data + rho * max_data)

# Load the stock data
df = pd.read_csv('apple_stock_data.csv')

# Assume that we are using the previous day's open, high, low, and close prices
# as influencing factors for the next day's close price
df['PrevOpen'] = df['Open'].shift(1)
df['PrevHigh'] = df['High'].shift(1)
df['PrevLow'] = df['Low'].shift(1)
df['PrevClose'] = df['Close'].shift(1)

# Drop missing values
df = df.dropna()

# Normalize the data
for col in ['PrevOpen', 'PrevHigh', 'PrevLow', 'PrevClose']:
    df[col] = normalization(df[col])

# Calculate the grey relational coefficients
for col in ['PrevOpen', 'PrevHigh', 'PrevLow', 'PrevClose']:
    df[col + 'GreyRelationalCoefficient'] = grey_relational_coefficient(df[col], df['Close'])

# Determine the weights of the influencing factors
weights = df[['PrevOpenGreyRelationalCoefficient', 'PrevHighGreyRelationalCoefficient', 'PrevLowGreyRelationalCoefficient', 'PrevCloseGreyRelationalCoefficient']].mean()

# For simplicity, we'll predict the next day's close price as a weighted average of the previous day's open, high, low, and close prices
df['PredictedClose'] = df['PrevOpen'] * weights[0] + df['PrevHigh'] * weights[1] + df['PrevLow'] * weights[2] + df['PrevClose'] * weights[3]

# Denormalize the predicted close price
df['PredictedClose'] = df['PredictedClose'] * (df['Close'].max() - df['Close'].min()) + df['Close'].min()

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(df['Close'], label='Actual Close')
plt.plot(df['PredictedClose'], label='Predicted Close')
plt.legend()
plt.show()
"""