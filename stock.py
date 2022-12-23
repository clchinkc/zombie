# https://github.com/ranaroussi/yfinance
# https://algotrading101.com/learn/yahoo-finance-api-guide/

import pandas as pd
import numpy as np

# get stock data from a csv file
def get_stock_data_from_csv(stock_name):
    stock_data = pd.read_csv(stock_name + '.csv', index_col='Date', parse_dates=True)
    return stock_data

def import_scaler():
    # import the scaler
    from sklearn.preprocessing import MinMaxScaler
    # create the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # return the scaler
    return scaler

def train_model():
    # import the yfinance library
    import yfinance as yf

    # Get the stock data for Apple Inc. (ticker symbol "AAPL")
    aapl = yf.Ticker("AAPL")
    data = aapl.history(period="max")
    
    # Alternatively, you can use yf.download to get the data
    # data = yf.download(tickers="AAPL", period="max")
    # get the number of rows and columns in the data set
    data.shape
    # visualize the closing price history
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 8))
    plt.title('Close Price History')
    plt.plot(data['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.show()
    # create a new dataframe with only the 'Close' column
    data = data.filter(['Close'])
    # convert the dataframe to a numpy array
    dataset = data.values
    # get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset) * .8))
    training_data_len
    # scale the all of the data to be values between 0 and 1
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # create the scaled training data set
    train_data = scaled_data[0:training_data_len, :]
    # split the data into x_train and y_train data sets
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
    # convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # import the LSTM model
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    # build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    # compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    # create the testing data set
    # create a new array containing scaled values from index 1543 to 2002
    test_data = scaled_data[training_data_len - 60:, :]
    # create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])
    # convert the data to a numpy array
    x_test = np.array(x_test)
    # reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    # get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    print(rmse)
    # plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
    # show the valid and predicted prices
    print(valid, predictions)
    # Get the stock data for Apple Inc. (ticker symbol "AAPL")
    aapl = yf.Ticker("AAPL")
    data = aapl.history(period="max")
    # create a new dataframe
    new_df = data.filter(['Close'])
    # get the last 60 day closing price values and convert the dataframe to an array
    last_60_days = new_df[-60:].values
    # scale the data to be values between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)
    # create an empty list
    X_test = []
    # append the past 60 days
    X_test.append(last_60_days_scaled)
    # convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    # reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # get the predicted scaled price
    pred_price = model.predict(X_test)
    # undo the scaling
    pred_price = scaler.inverse_transform(pred_price)
    print(pred_price)
    
# use stock data for prediction
def predict_stock_data(stock_data, model):
    # get the last row of data
    last_row = stock_data.iloc[-1]
    # get the last row of data as a list
    last_row_list = last_row.tolist()
    # get only the last 60 days of data
    last_60_days = stock_data[-60:]
    # get the last 60 days of data as a list
    last_60_days_list = last_60_days.values.tolist()
    # scale the data
    last_60_days_scaled = scaler.transform(last_60_days_list)
    # create an empty list
    X_test = []
    # append the last 60 days of data to the empty list
    X_test.append(last_60_days_scaled)
    # convert the X_test list to a numpy array
    X_test = np.array(X_test)
    # reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # get the predicted scaled price
    pred_price = model.predict(X_test)
    # undo the scaling
    pred_price = scaler.inverse_transform(pred_price)
    # get the predicted price
    pred_price = pred_price[0][0]
    # get the actual price
    actual_price = last_row_list[4]
    # return the predicted price and the actual price
    return pred_price, actual_price


