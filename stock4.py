
import pandas as pd
import yfinance as yf
from datetime import datetime
from datetime import timedelta
import plotly.graph_objects as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '${:,.2f}'.format

today = datetime.today().strftime('%Y-%m-%d')
start_date = '2016-01-01'
eth_df = yf.download('ETH-USD',start_date, today)
eth_df.tail()
# We see that our data has date, open, high, low, close, adjusted close price, and volume

# Now we’ll do a bit of analysis on our data running info()
eth_df.info()

# As well as checking for null values just in case.
eth_df.isnull().sum()

# We need a date column for our prophet model, but it’s not listed as one of the columns. Let’s figure out why that’s the case.
eth_df.columns

# We’ll reset the index, and we can have our Date as a column.
eth_df.reset_index(inplace=True)
eth_df.columns

# The prophet library requires us to have only two columns in our data frame — “ds” and “y”, which is the dateandopen columns respectively.
# So let’s grab the necessary columns and put it into a new data frame. Then we use the rename function to change the column names.
df = eth_df[["Date", "Open"]]
new_names = {
    "Date": "ds", 
    "Open": "y",
}
df.rename(columns=new_names, inplace=True)

# Running the tail function again, we see our data is ready for Prophet.
df.tail()

# plot the open price
x = df["ds"]
y = df["y"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y))
# Set title
fig.update_layout(
    title_text="Time series plot of Ethereum Open Price",
)

# First we define our model, and tune it according to your purpose, then you can fit it to your data frame.
# Note this is a very simple model and there can be more tuning done to it to improve its accuracy.
m = Prophet(
    seasonality_mode="multiplicative" 
)
m.fit(df)

# Now we create an entire years worth of date data for our prophet model to make predictions
future = m.make_future_dataframe(periods = 365)
future.tail()
# We see the date is one year from today’s date.

# Then, running the predictions is as easy as calling the predict function
# Then we grab the essential columns we need.
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# We can also get the price prediction for the next day just for fun.
next_day = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
forecast[forecast['ds'] == next_day]['yhat'].item()

# Prophet also has built-in plotly functions that can help us easily visualize our forecast.
plot_plotly(m, forecast)

# Our forecasting model also includes growth curve trend, weekly seasonal, and yearly seasonal components which can be visualized like this.
plot_components_plotly(m, forecast)

# You could even experiment with alternatives like the ARIMA model or Deep learning (LSTM Models) to perform forecasting, and then compare their performance using diagnostics like R-squared or RMSE.