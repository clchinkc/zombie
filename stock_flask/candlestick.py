
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html
from plotly.subplots import make_subplots


def create_candlestick_figure(stock, period):
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

    # Create a Plotly chart
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, row_heights=[0.3, 0.15, 0.15, 0.15, 0.15], vertical_spacing=0.035)

    # Add the candlestick chart
    fig.add_trace(go.Candlestick(x=klines.index, open=klines['open'], high=klines['high'], low=klines['low'], close=klines['close'], name="Candlestick"), row=1, col=1)

    # Add moving averages
    fig.add_trace(go.Scatter(x=MA(klines, 20).index, y=MA(klines, 20)['MA'], mode='lines', line=dict(color='red', width=1), name="MA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=MA(klines, 50).index, y=MA(klines, 50)['MA'], mode='lines', line=dict(color='blue', width=1), name="MA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=MA(klines, 100).index, y=MA(klines, 100)['MA'], mode='lines', line=dict(color='green', width=1), name="MA 100"), row=1, col=1)
    fig.add_trace(go.Scatter(x=MA(klines, 200).index, y=MA(klines, 200)['MA'], mode='lines', line=dict(color='yellow', width=1), name="MA 200"), row=1, col=1)

    # Add RSI
    fig.add_trace(go.Scatter(x=RSI(klines, 14).index, y=RSI(klines, 14)['RSI'], mode='lines', line=dict(color='blue', width=1), name="RSI"), row=3, col=1)

    # Add MACD and MACD signal
    fig.add_trace(go.Scatter(x=MACD(klines, 12, 26).index, y=MACD(klines, 12, 26)['MACD'], mode='lines', line=dict(color='red', width=1), name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=MACD(klines, 12, 26).index, y=MACD(klines, 12, 26)['MACDsignal'], mode='lines', line=dict(color='green', width=1), name="MACD Signal"), row=4, col=1)

    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=BollingerBand(klines, 20).index, y=BollingerBand(klines, 20)['Average'], mode='lines', line=dict(color='green', width=1), name="BB Average"), row=5, col=1)
    fig.add_trace(go.Scatter(x=BollingerBand(klines, 20).index, y=BollingerBand(klines, 20)['BB_up'], mode='lines', line=dict(color='yellow', width=1), name="BB Upper"), row=5, col=1)
    fig.add_trace(go.Scatter(x=BollingerBand(klines, 20).index, y=BollingerBand(klines, 20)['BB_dn'], mode='lines', line=dict(color='blue', width=1), name="BB Lower"), row=5, col=1)

    # Set chart properties
    fig.update_layout(title="APPL Chart with Technical Indicators", xaxis_rangeslider_visible=True, height=1000)
    fig.update_xaxes(type='category')
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    fig.update_yaxes(title_text="Bollinger Bands", row=5, col=1)
    
    return fig



def create_dash_layout(fig):
    # Define the layout
    layout = html.Div([
        html.H1("Candlestick Chart with Technical Indicators"),
        dcc.Graph(figure=fig, id='candlestick'),
        dbc.Row([
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Card 1", className="card-title"),
                        html.P("This is the first card."),
                    ])
                ),
            ], md=6),
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Card 2", className="card-title"),
                        html.P("This is the second card."),
                    ])
                ),
            ], md=6),
        ]),
    ])

    return layout

