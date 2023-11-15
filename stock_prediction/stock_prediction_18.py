import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.fft import fft, ifft


def load_stock_data(symbol, start_date, end_date):
    """
    Load stock data from Yahoo Finance.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def apply_fft(data):
    """
    Apply FFT to the closing prices of the stock data.
    """
    close_prices = data['Close'].values
    fft_result = fft(close_prices)
    return fft_result

def reconstruct_signal(fft_result, threshold=0.1):
    """
    Reconstruct the signal using only significant frequencies.
    """
    fft_copy = np.copy(fft_result)
    magnitudes = np.abs(fft_result)

    # Zero out frequencies below the threshold
    fft_copy[magnitudes < threshold * np.max(magnitudes)] = 0

    # Inverse FFT to get the reconstructed signal
    reconstructed_signal = ifft(fft_copy)
    return reconstructed_signal.real

def plot_signals(original, reconstructed, title='Stock Price Analysis'):
    """
    Plot the original and reconstructed signals.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(original, label='Original Signal')
    plt.plot(reconstructed, label='Reconstructed Signal', linestyle='--')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Load data for a specific stock
symbol = 'AAPL'  # Example with Apple Inc.
start_date = '2020-01-01'
end_date = '2023-01-01'
stock_data = load_stock_data(symbol, start_date, end_date)

# Apply FFT
fft_result = apply_fft(stock_data)

# Reconstruct the signal using significant frequencies
reconstructed_signal = reconstruct_signal(fft_result)

# Plot the original and reconstructed signals
plot_signals(stock_data['Close'].values, reconstructed_signal)