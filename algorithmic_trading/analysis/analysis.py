import datetime
import pandas as pd
import numpy as np
import requests
import dash
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output

# Convert human-readable date to milliseconds (UTC)
def get_start_time(year, month, day, hour=0, minute=0, second=0):
    dt = datetime.datetime(year, month, day, hour, minute, second)  # Create datetime object
    timestamp_ms = int(dt.timestamp() * 1000)  # Convert to milliseconds
    return timestamp_ms

def get_btc_data(interval, limit, start_time=None):
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit={limit}"
    if start_time:
        url += f"&startTime={start_time}"
    response = requests.get(url).json()
    
    new_data = pd.DataFrame(response, columns=["timestamp", "open", "high", "low", "close", "volume", 
                                               "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"])
    
    # Convert timestamp and necessary columns
    new_data["timestamp"] = pd.to_datetime(new_data["timestamp"], unit="ms", utc=True).dt.tz_convert("Etc/GMT-3")
    new_data[["open", "high", "low", "close", "volume"]] = new_data[["open", "high", "low", "close", "volume"]].astype(float)
    
    return new_data[["timestamp", "open", "high", "low", "close", "volume"]]

# Function to calculate Fibonacci retracement levels
def calculate_fibonacci_levels(df, lookback=300):
    df = df.tail(lookback)
    max_price = df["high"].max()
    min_price = df["low"].min()

    levels = {
        "0%": max_price,
        "23.6%": max_price - 0.236 * (max_price - min_price),
        "38.2%": max_price - 0.382 * (max_price - min_price),
        "50%": max_price - 0.5 * (max_price - min_price),
        "61.8%": max_price - 0.618 * (max_price - min_price),
        "78.6%": max_price - 0.786 * (max_price - min_price),
        "100%": min_price
    }
    
    return levels

# Function to calculate RSI
def calculate_rsi(df, period=14):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    return df

# Function to calculate MACD
def calculate_macd(df, short=12, long=26, signal=9):
    df["EMA12"] = df["close"].ewm(span=short, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=long, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]  # Histogram
    
    return df

# Function to calculate Moving Averages
def calculate_moving_averages(df):
    df["EMA9"] = df["close"].ewm(span=9, adjust=False).mean()  # 20-period EMA
    df["EMA21"] = df["close"].ewm(span=21, adjust=False).mean()  # 20-period EMA
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()  # 50-period EMA
    df["EMA100"] = df["close"].ewm(span=100, adjust=False).mean()  # 100-period EMA
    return df

  
def calculate_vwap(df, period_candles=None, name="VWAP"):
    """
    Calculate VWAP (Volume Weighted Average Price) for different periods.
    :param df: DataFrame with 'high', 'low', 'close', and 'volume'.
    :param period_candles: Number of candles to use for VWAP (e.g., 10, 50, 200).
                           If None, calculates cumulative VWAP for all data.
    :return: DataFrame with VWAP column.
    """
    df["Typical Price"] = (df["high"] + df["low"] + df["close"]) / 3

    if period_candles:
        df[name] = (
            df["Typical Price"].rolling(window=period_candles).apply(lambda x: (x * df["volume"]).sum()) /
            df["volume"].rolling(window=period_candles).sum()
        )
    else:
        df["Cumulative TP x Volume"] = (df["Typical Price"] * df["volume"]).cumsum()
        df["Cumulative Volume"] = df["volume"].cumsum()
        df[name] = df["Cumulative TP x Volume"] / df["Cumulative Volume"]

    return df


# Initialize Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Live Bitcoin (BTC/USDT) Price with Moving Averages, Fibonacci, RSI & MACD", style={'text-align': 'center'}),
    dcc.Graph(id='live-graph'),
    dcc.Interval(id='interval-component', interval=60000, n_intervals=0)  # Refresh every 5 seconds
])

# Callback function to update the graph with real-time data
@app.callback(
    Output('live-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n_intervals):
    
    # Fetch the latest 4-hour BTC data
    limit = 1200
    days = limit/6 # total_candlesticks/n_candle_stick_per_day
    start_time = datetime.date.today() - datetime.timedelta(days=days)
    data = get_btc_data(interval="4h",
                        limit=limit)
    
    # Compute Fibonacci levels for two weeks
    fib_levels = calculate_fibonacci_levels(data, lookback=150)

    # Compute RSI, MACD, and Moving Averages
    data = calculate_rsi(data)
    data = calculate_macd(data)
    data = calculate_moving_averages(data)
    data = calculate_vwap(data, period_candles=1, name="4-Hourly VWAP") # Daily VWAP
    data = calculate_vwap(data, period_candles=6, name="Daily VWAP") # Weekly VWAP

    # Create a live chart with 3 subplots: Price, RSI, MACD
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, shared_yaxes=True,vertical_spacing=0.1,
                        subplot_titles=("BTC/USDT Price with Moving Averages & Fibonacci Levels",
                                        "RSI (Relative Strength Index)", 
                                        "MACD Indicator"))

    # --- Price Chart with Fibonacci Levels & Moving Averages ---
    fig.add_trace(go.Candlestick(
        x=data["timestamp"],  # Time
        open=data["open"],    # Open Price
        high=data["high"],    # High Price
        low=data["low"],      # Low Price
        close=data["close"],  # Close Price
        name="BTC/USDT"
    ))

    for level, price in fib_levels.items():
        fig.add_trace(go.Scatter(x=[data["timestamp"].iloc[0], data["timestamp"].iloc[-1]], 
                                 y=[price, price], mode="lines", name=f"Fib {level}", line=dict(dash="dash")), row=1, col=1)

    # Add Moving Averages
    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["EMA50"], mode="lines",
                             name="50-EMA", line=dict(color="orange")), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["EMA100"], mode="lines",
                             name="100-EMA", line=dict(color="red")), row=1, col=1)

    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["EMA21"], mode="lines",
                             name="21-EMA", line=dict(color="green")), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["EMA9"], mode="lines",
                             name="9-EMA", line=dict(color="brown")), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["4-Hourly VWAP"], mode="lines",
                         name="4-Hourly VWAP", line=dict(color="black", dash="dot")), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["Daily VWAP"], mode="lines",
                         name="Daily VWAP", line=dict(color="purple", dash="dot")), row=1, col=1)


    # --- RSI Chart ---
    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["RSI"], mode="lines",
                             name="RSI", line=dict(color="purple")), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="red", dash="dash"), row=2, col=1, annotation_text="Overbought")
    fig.add_hline(y=30, line=dict(color="green", dash="dash"), row=2, col=1, annotation_text="Oversold")

    # --- MACD Chart ---
    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["MACD"], mode="lines",
                             name="MACD", line=dict(color="blue")), row=3, col=1)
    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["Signal"], mode="lines",
                             name="Signal", line=dict(color="orange")), row=3, col=1)
    fig.add_trace(go.Bar(x=data["timestamp"], y=data["MACD_Hist"],
                         name="MACD Histogram", marker_color="gray"), row=3, col=1)

    # Update layout
    fig.update_layout(title="Real-Time BTC/USDT Price Chart with Moving Averages, Fibonacci, RSI & MACD",
                      xaxis_title="Time",
                      height=8000,
                      showlegend=True)

    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
