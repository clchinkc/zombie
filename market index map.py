
"""
import plotly.express as px
import yfinance as yf

# List of stock market indices from various countries
indices = {
    'US': '^GSPC', # S&P 500
    'Japan': '^N225', # Nikkei 225
    'China': '000001.SS', # Shanghai Composite
    'India': '^BSESN', # BSE Sensex
    'Germany': '^GDAXI', # DAX
    'UK': '^FTSE', # FTSE 100
    'France': '^FCHI', # CAC 40
    'Canada': '^GSPTSE', # S&P/TSX Composite
    'Brazil': '^BVSP', # Bovespa
    'Australia': '^AXJO', # S&P/ASX 200
    'Hong Kong': '^HSI', # Hang Seng
    
}

# Fetch data from yfinance
data = yf.download(list(indices.values()), period='1d', interval='1d')['Adj Close']

# Rename columns with country names
data.columns = list(indices.keys())

# Create a Scatter map using Plotly Express
# Create a choropleth map using Plotly Express
fig = px.choropleth(data_frame=data.mean(), locations=data.mean().index, 
                    locationmode='country names', color=data.mean(),
                    title='Stock Market Indices by Country')
fig.show()
"""

import pandas as pd
import plotly.express as px
import yfinance as yf

# Define a list of stock market indexes and their respective countries with coordinates
index_list = [
    {'ticker': '^GSPC', 'name': 'S&P 500', 'country': 'USA', 'latitude': 37.0902, 'longitude': -95.7129},
    {'ticker': '^FTSE', 'name': 'FTSE 100', 'country': 'UK', 'latitude': 55.3781, 'longitude': -3.4360},
    {'ticker': '^N225', 'name': 'Nikkei 225', 'country': 'Japan', 'latitude': 36.2048, 'longitude': 138.2529},
    {'ticker': '^GDAXI', 'name': 'DAX', 'country': 'Germany', 'latitude': 51.1657, 'longitude': 10.4515},
    {'ticker': '^HSI', 'name': 'Hang Seng', 'country': 'Hong Kong', 'latitude': 22.3193, 'longitude': 114.1694},
    {'ticker': '^BSESN', 'name': 'BSE SENSEX', 'country': 'India', 'latitude': 20.5937, 'longitude': 78.9629},
    {'ticker': '^AXJO', 'name': 'S&P/ASX 200', 'country': 'Australia', 'latitude': -25.2744, 'longitude': 133.7751},
    {'ticker': '000001.SS', 'name': 'SSE Composite Index', 'country': 'China', 'latitude': 35.8617, 'longitude': 104.1954},
    {'ticker': '^KS11', 'name': 'KOSPI', 'country': 'South Korea', 'latitude': 35.9078, 'longitude': 127.7669},
    {'ticker': '^TWII', 'name': 'TSEC weighted index', 'country': 'Taiwan', 'latitude': 23.6978, 'longitude': 120.9605},
]

# Fetch stock data and store it in the index list
for index_info in index_list:
    stock = yf.Ticker(index_info['ticker'])
    data = stock.history(period='1d')
    index_info['price'] = data.iloc[-1]['Close']
    index_info['change'] = data.iloc[-1]['Close'] - data.iloc[-1]['Open']
    index_info['change_percent'] = index_info['change'] / data.iloc[-1]['Open'] * 100

# Create a DataFrame to store the index data
df = pd.DataFrame(index_list)

# Create a scatter map using Plotly Express
fig = px.scatter_geo(
    df,
    lat='latitude',
    lon='longitude',
    hover_name='country',
    hover_data=['name', 'price', 'change', 'change_percent'],
    color='change_percent',
    size='price',
    projection="natural earth",
    color_continuous_scale=px.colors.diverging.RdYlGn,
    title="Stock Market Indices by Country"
)
fig.show()

