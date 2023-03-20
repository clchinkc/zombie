

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

# Load some sample data
data = pd.read_csv('apple_stock_data.csv', index_col=0, parse_dates=True)
data.columns = [col.lower() for col in data.columns]
klines = data.set_index(pd.to_datetime(data.index, utc=True))

# Financial indicators
def MA(df, n=20):
    df['MA'] = df['close'].rolling(n, min_periods=1).mean()
    return df

def RSI(df, n=14):
    df['delta'] = df['close'] - df['close'].shift(1)
    df['gain'] = np.where(df['delta'] >= 0, df['delta'], 0)
    df['loss'] = np.where(df['delta'] < 0, abs(df['delta']), 0)
    avg_gain = []
    avg_loss = []
    gain = df['gain'].tolist()
    loss = df['loss'].tolist()
    for i in range(len(df)):
        if i < n:
            avg_gain.append(np.NaN)
            avg_loss.append(np.NaN)
        elif i == n:
            avg_gain.append(df['gain'].rolling(n).mean().tolist()[n])
            avg_loss.append(df['loss'].rolling(n).mean().tolist()[n])
        elif i > n:
            avg_gain.append(((n - 1) * avg_gain[i - 1] + gain[i]) / n)
            avg_loss.append(((n - 1) * avg_loss[i - 1] + loss[i]) / n)
    df['avg_gain'] = np.array(avg_gain)
    df['avg_loss'] = np.array(avg_loss)
    df['RS'] = df['avg_gain'] / df['avg_loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))
    return df

def MACD(df, n_fast=12, n_slow=26):
    EMAfast = df.close.ewm(span=n_fast, min_periods=n_slow - 1).mean()
    EMAslow = df.close.ewm(span=n_slow, min_periods=n_slow - 1).mean()
    df['MACD'] = EMAfast - EMAslow
    df['MACDsignal'] = df.MACD.ewm(span=9, min_periods=8).mean()
    return df

def BollingerBand(df, n=20):
    df['Average'] = df['close'].rolling(n).mean()
    df['BB_up'] = df['Average'] + 2 * df['close'].rolling(n).std()
    df['BB_dn'] = df['Average'] - 2 * df['close'].rolling(n).std()
    df['BB_width'] = df['BB_up'] - df['BB_dn']
    return df

# Create a new style for the chart with the moving average
mc = mpf.make_marketcolors(up='green', down='red')
my_style = mpf.make_mpf_style(base_mpl_style='seaborn', marketcolors=mc, mavcolors=['black'])

# Plot the candlestick chart
ma20_data = mpf.make_addplot(MA(klines, 20)['MA'], panel = 0, color='red', ylabel='MA')
ma50_data = mpf.make_addplot(MA(klines, 50)['MA'], panel = 0, color='blue', ylabel='MA')
ma100_data = mpf.make_addplot(MA(klines, 100)['MA'], panel = 0, color='green', ylabel='MA')
ma200_data = mpf.make_addplot(MA(klines, 200)['MA'], panel = 0, color='yellow', ylabel='MA')
rsi_data = mpf.make_addplot(RSI(klines, 14)['RSI'], panel = 2, color = 'b', ylabel = 'RSI')
macd_data = mpf.make_addplot(MACD(klines, 12, 26)['MACD'], panel = 3, color = 'r', ylabel = 'MACD')
macdsignal_data = mpf.make_addplot(MACD(klines, 12, 26)['MACDsignal'], panel = 3, color = 'g', ylabel = 'MACD')
average_data = mpf.make_addplot(BollingerBand(klines, 20)['Average'], panel = 4, color = 'g', ylabel = 'BB')
bbup_data = mpf.make_addplot(BollingerBand(klines, 20)['BB_up'], panel = 4, color = 'y', ylabel = 'BBW')
bbdn_data = mpf.make_addplot(BollingerBand(klines, 20)['BB_dn'], panel = 4, color = 'b', ylabel = 'BBW')
mpf.plot(klines, type = 'candle', style = 'yahoo', title = 'APPL Chart', block = False, volume = True, addplot = [ma20_data, ma50_data, ma100_data, ma200_data, rsi_data, macd_data, macdsignal_data, average_data, bbup_data, bbdn_data])
plt.show()



# https://blog.csdn.net/Shepherdppz/article/details/117575286
# https://blog.csdn.net/Shepherdppz/article/details/108205721?spm=1001.2014.3001.5501
# https://github.com/matplotlib/mplfinance/blob/master/examples/price-movement_plots.ipynb
# https://github.com/matplotlib/mplfinance/blob/master/examples/using_lines.ipynb
# https://github.com/matplotlib/mplfinance/blob/master/examples/addplot.ipynb

"""
import time
from datetime import datetime

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import requests

# Get Bitcoin data
# # Global Variables Setting
symbol = 'BTCUSDT'
url = 'https://api.binance.com/'

# # Get Market Data
def GetKline(url, symbol, interval, startTime = None, endTime = None):
    try:
        data = requests.get(url + 'api/v3/klines', params={'symbol': symbol, 'interval': interval, 'startTime': startTime, 'limit': 1000}).json()
    except Exception as e:
        print ('Error! problem is {}'.format(e.args[0]))
    tmp  = []
    pair = []
    for base in data:
        tmp  = []
        for i in range(0,6):
            if i == 0:
                base[i] = datetime.fromtimestamp(base[i]/1000)
            tmp.append(base[i])
        pair.append(tmp)
    df = pd.DataFrame(pair, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df.date = pd.to_datetime(df.date)
    df.set_index("date", inplace=True)
    df = df.astype(float)
    return df

def GetHistoricalKline(url, symbol, interval, startTime):
    # init
    klines = GetKline(url, symbol, interval)
    tmptime = ToMs(klines.iloc[0].name)
    
    # Send request until tmptime > startTime
    while tmptime > startTime:
        tmptime -= PeriodToMs(interval) * 1000 # tmp minus period ms plus 1000 (1000 K)
        if tmptime < startTime:
            tmptime = startTime
        tmpdata = GetKline(url, symbol, interval, tmptime)
        klines  = pd.concat([tmpdata, klines])

    return klines.drop_duplicates(keep='first', inplace=False)

# Math Tools
def ToMs(date):
    return int(time.mktime(time.strptime(str(date), "%Y-%m-%d %H:%M:%S")) * 1000) # Binance timestamp format is 13 digits

def PeriodToMs(period):
    Ms = None
    ToSeconds = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60
    }
    unit = period[-1]

    if unit in ToSeconds:
        try:
            Ms = int(period[:-1]) * ToSeconds[unit] * 1000
        except ValueError:
            pass
    return Ms

klines = GetHistoricalKline(url, symbol, '1d', ToMs('2023-01-01 12:00:00'))
"""

