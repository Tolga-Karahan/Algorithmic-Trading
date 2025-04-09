import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import dash
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output


# Function to fetch US stock data using yfinance
def get_us_stock_data(symbol, interval, period):
    data = yf.download(symbol, period=f"{period}d", interval=interval)
    data.reset_index(inplace=True)
    data.rename(columns={"Date": "timestamp"}, inplace=True)
    return data


# Technical indicator functions (same as before)
def calculate_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df, short=12, long=26, signal=9):
    df["EMA12"] = df["Close"].ewm(span=short, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=long, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]
    return df


def calculate_moving_averages(df):
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA100"] = df["Close"].ewm(span=100, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=100, adjust=False).mean()
    return df


# Dash App
app = dash.Dash(__name__)

# List of US stock symbols and interval options
stock_symbols = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "QQQ",
    "PSQ",
    "SQQQ",
    "TQQQ",
    "VOO",
    "SCHD",
    "SCHY",
    "SCHG",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "LB",
    "EUAD",
    "ASML",
    "AMD",
    "RKLB",
    "PLTR",
    "FIX",
]
intervals = [
    "1h",
    "4h",
    "1d",
    "5d",
    "1wk",
    "1mo",
]

app.layout = html.Div(
    [
        html.H1("US Stock Technical Dashboard", style={"text-align": "center"}),
        html.Div(
            [
                html.Label("Select Stock Symbol:"),
                dcc.Dropdown(
                    id="symbol-dropdown",
                    options=[{"label": sym, "value": sym} for sym in stock_symbols],
                    value="AAPL",
                ),
                html.Label("Select Interval:"),
                dcc.Dropdown(
                    id="interval-dropdown",
                    options=[{"label": i, "value": i} for i in intervals],
                    value="1d",
                ),
                html.Label("Enter the Period in Days:"),
                dcc.Input(
                    id="period-input",
                    type="text",
                    value="30",  # default value
                    debounce=True,  # triggers only after Enter or focus out
                    placeholder="Enter a number",
                    style={"width": "100%", "padding": "8px"},
                ),
            ],
            style={"width": "48%", "margin": "auto"},
        ),
        dcc.Graph(id="stock-chart"),
    ]
)


@app.callback(
    Output("stock-chart", "figure"),
    Input("symbol-dropdown", "value"),
    Input("interval-dropdown", "value"),
    Input("period-input", "value"),
)
def update_graph(symbol, interval, period):
    df = get_us_stock_data(symbol, interval, period)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_moving_averages(df)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Price", "RSI", "MACD"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlestick",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["EMA9"], name="EMA9", line=dict(color="blue")
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["EMA21"], name="EMA21", line=dict(color="green")
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["EMA50"], name="EMA50", line=dict(color="green")
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["EMA100"], name="EMA100", line=dict(color="green")
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["EMA200"], name="EMA200", line=dict(color="green")
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["RSI"], name="RSI", line=dict(color="purple")
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line=dict(color="red", dash="dash"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="green", dash="dash"), row=2, col=1)

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["MACD"], name="MACD", line=dict(color="blue")
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["Signal"], name="Signal", line=dict(color="orange")
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df["timestamp"], y=df["MACD_Hist"], name="MACD Hist", marker_color="gray"
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=900,
        showlegend=True,
        title=f"Technical Indicators for {symbol} ({interval})",
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
