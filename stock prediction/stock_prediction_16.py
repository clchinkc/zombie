
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def denormalize(data, reference):
    return data * (np.max(reference) - np.min(reference)) + np.min(reference)

def grey_relational_coefficient(reference, comparison, rho=0.5):
    abs_diff = np.abs(reference - comparison)
    max_diff = np.max(abs_diff)
    min_diff = np.min(abs_diff)
    return (min_diff + rho * max_diff) / (abs_diff + rho * max_diff + 1e-10)

def grey_relational_analysis(reference_sequence, comparison_sequences):
    normalized_reference = normalize(reference_sequence)
    normalized_comparisons = np.array([normalize(sequence) for sequence in comparison_sequences])

    coefficients = np.array([grey_relational_coefficient(normalized_reference, sequence) for sequence in normalized_comparisons])

    weights = coefficients.sum(axis=0) / coefficients.sum()

    aggregated_sequence = np.average(normalized_comparisons, axis=1, weights=weights)
    
    denormalized_aggregated_sequence = denormalize(aggregated_sequence, reference_sequence)
    
    print("Weights:", weights)
    print("Coefficients:", coefficients)
    print("Denormalized aggregated sequence:", denormalized_aggregated_sequence)

    return denormalized_aggregated_sequence

# Load data
data = pd.read_csv('apple_stock_data.csv')

# Create shifted columns and other sequences
data['PrevOpen'] = data['Open'].shift(1, fill_value=np.mean(data['Open']))
data['PrevHigh'] = data['High'].shift(1, fill_value=np.mean(data['High']))
data['PrevLow'] = data['Low'].shift(1, fill_value=np.mean(data['Low']))
data['PrevClose'] = data['Close'].shift(1, fill_value=np.mean(data['Close']))
data['5DayMovingAverage'] = data['Close'].rolling(window=5, min_periods=1).mean()

# Handle missing values
data.dropna(inplace=True)

# Extract the 'Close' column as the reference sequence
reference_sequence = data['Close'].values

# Extract other columns as comparison sequences
comparison_sequences = [
    data['PrevOpen'].values, 
    data['PrevHigh'].values, 
    data['PrevLow'].values,
    data['PrevClose'].values,
    data['5DayMovingAverage'].values
]

# Perform grey relational analysis
predicted_sequence = grey_relational_analysis(reference_sequence, comparison_sequences)

# Visualization
next_date = datetime.datetime.strptime(data['Date'].values[-1], '%Y-%m-%d %H:%M:%S%z') + datetime.timedelta(days=1)
next_date_str = next_date.strftime('%Y-%m-%d %H:%M:%S%z')

plt.figure(figsize=(10,5))
plt.plot(data['Date'].values, data['Close'].values, label='Actual Close')
plt.plot(next_date_str, predicted_sequence[0], label='Predicted Close', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

print(data['Close'].values[-1])
print(predicted_sequence[0])


