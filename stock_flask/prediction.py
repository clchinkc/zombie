
import base64
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

matplotlib.use('agg')

def predict_stock_price(stock, period, algorithm):
    # Retrieve historical price data for the stock
    # stock_data = yf.download(stock, period=period)
    stock_data = pd.read_csv('apple_stock_data.csv', index_col=0, parse_dates=True)

    # Preprocess the data for machine learning
    X = np.array([i for i in range(len(stock_data['Close']))]).reshape(-1, 1)
    y = stock_data['Close'].values.reshape(-1)

    # Train a machine learning model
    if algorithm == 'linear':
        model = LinearRegression()
    elif algorithm == 'random_forest':
        model = RandomForestRegressor()
    else:
        model = MLPRegressor()

    model.fit(X, y)

    # Generate predictions
    future_periods = 30 # predict 30 days into the future
    future_X = np.array([i for i in range(len(stock_data), len(stock_data)+future_periods)]).reshape(-1, 1)
    future_y = model.predict(future_X)

    # Visualize the historical data and predicted future prices
    fig, ax = plt.subplots()
    ax.plot(X, y)
    ax.plot(future_X, future_y)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'{stock} Historical and Future Prices')
    ax.legend(['Historical Prices', 'Predicted Future Prices'])

    # Convert the plot to a base64-encoded string and render the template
    buffer = BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buffer)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return stock, future_y, plot_data