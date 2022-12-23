# https://facebook.github.io/prophet/docs/quick_start.html#python-api

# pip install fbprophet

import pandas as pd
from fbprophet import Prophet

# Load the data into a Pandas dataframe
df = pd.read_csv("hang_seng_index_data.csv")

# Rename the columns to match the format expected by Prophet
df = df.rename(columns={"date": "ds", "close": "y"})

# Fit the Prophet model to the data
m = Prophet()
m.fit(df)

# Make predictions for the next 30 days
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

# View the predicted values
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
