import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0

    def update(self, setpoint, pv):
        """
        Update the PID controller.
        :param setpoint: The desired set point.
        :param pv: Present value - the current value.
        :return: The control output.
        """
        error = setpoint - pv
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative

# Example usage with stock price data from a CSV file
def predict_stock_price(csv_file, kp, ki, kd):
    # Load historical stock data from a CSV file
    data = pd.read_csv(csv_file)
    close_prices = data['Close']

    pid = PIDController(kp, ki, kd)
    setpoint = close_prices.mean()  # Example setpoint as the mean price
    predictions = []

    for price in close_prices:
        control = pid.update(setpoint, price)
        predicted_price = price + control  # Adjust the price based on PID control
        predictions.append(predicted_price)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], close_prices, label='Actual Price')
    plt.plot(data['Date'], predictions, label='Predicted Price', linestyle='--')
    plt.title("Stock Price Prediction")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

# Example parameters
csv_file = 'apple_stock_data.csv'  # Replace with your CSV file path
kp, ki, kd = 0.1, 0.01, 0.05

predict_stock_price(csv_file, kp, ki, kd)
