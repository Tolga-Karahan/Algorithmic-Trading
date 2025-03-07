import os
import datetime
import pandas as pd
import numpy as np
from tqdm.contrib.concurrent import process_map  # Parallel tqdm
from multiprocessing import cpu_count, set_start_method
from isyatirimhisse import StockData

# Initialize StockData
stock_data = StockData()


def get_closest_weekday_before(date=None):
    """Find the closest weekday before the given date (defaults to today)."""
    if date is None:
        date = datetime.date.today()

    # If today is Saturday (5), subtract 1 day (Friday)
    # If today is Sunday (6), subtract 2 days (Friday)
    if date.weekday() == 5:  # Saturday
        return (date - datetime.timedelta(days=1)).strftime("%d-%m-%Y")
    elif date.weekday() == 6:  # Sunday
        return (date - datetime.timedelta(days=2)).strftime("%d-%m-%Y")
    else:
        return date.strftime("%d-%m-%Y")

    # Otherwise, return the same date (it's already a weekday)
    return


def get_turkish_stock_symbols(n=-1):
    """Fetch Turkish stock symbols dynamically or read from a local file."""
    if "symbols.txt" not in os.listdir():
        print("Fetching symbols from İş Yatırım. Saving locally for reuse.")

        # Fetch all BIST stocks from İş Yatırım
        bist_stocks = stock_data.get_stock_list()
        stocks = bist_stocks["Kod"].tolist()  # Extract stock tickers

        if n != -1:
            stocks = stocks[:n]

        # Save symbols to a file
        with open("symbols.txt", "w") as f:
            for stock in stocks:
                f.write(f"{stock}\n")

        return stocks
    else:
        print("Symbol file found, returning the symbols!")
        with open("symbols.txt", "r") as f:
            return f.read().split("\n")[:-1]


def get_stock_price(symbol, date):
    """Fetch stock price for a given symbol and date using İş Yatırım."""
    df = stock_data.get_data(symbols=symbol, start_date=date, end_date=date)

    if df is None or df.empty or "CLOSING_TL" not in df.columns:
        print(f"No data available for {symbol} on {date}")
        return None

    return df.iloc[-1]["CLOSING_TL"]  # Closing price


def get_stock_price_series(symbol, start_date, end_date):
    df = stock_data.get_data(symbols=symbol, start_date=start_date, end_date=end_date)

    if df is None or df.empty or "CLOSING_TL" not in df.columns:
        print(f"No data available for {symbol}!")
        return pd.DataFrame()

    return df.sort_index().reset_index()


def calculate_change_in_percent(first_price, last_price):
    if first_price and last_price:
        try:
            return ((last_price - first_price) / first_price) * 100
        except TypeError:
            return
    else:
        return


def calculate_momentum_indicators(df):
    """Calculate momentum indicators for a stock."""

    # Relative Strength Index (RSI)
    delta = df["CLOSING_TL"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Moving Averages (Short-term vs Long-term)
    df["SMA_9"] = df["CLOSING_TL"].rolling(window=9).mean()
    df["SMA_21"] = df["CLOSING_TL"].rolling(window=21).mean()

    # MACD (12-day EMA - 26-day EMA)
    df["EMA_12"] = df["CLOSING_TL"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["CLOSING_TL"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    return df.iloc[-1]


def process_stock(stock):
    """Process stock data in parallel."""
    try:
        today = datetime.date.today()
        one_week_ago = today - datetime.timedelta(days=7)
        two_weeks_ago = today - datetime.timedelta(days=14)
        three_weeks_ago = today - datetime.timedelta(days=21)
        one_month_ago = today - datetime.timedelta(days=30)

        one_week_ago_price = get_stock_price(stock, get_closest_weekday_before(one_week_ago))
        two_weeks_ago_price = get_stock_price(stock, get_closest_weekday_before(two_weeks_ago))
        three_weeks_ago_price = get_stock_price(stock, get_closest_weekday_before(three_weeks_ago))
        one_month_ago_price = get_stock_price(stock, get_closest_weekday_before(one_month_ago))
        latest_price = get_stock_price(stock, get_closest_weekday_before())

        one_month_full_data = get_stock_price_series(
            stock,
            get_closest_weekday_before(one_month_ago),
            get_closest_weekday_before(),
        )

        change_percent_one_week = calculate_change_in_percent(one_week_ago_price, latest_price)
        change_percent_two_weeks = calculate_change_in_percent(two_weeks_ago_price, latest_price)
        change_percent_three_weeks = calculate_change_in_percent(three_weeks_ago_price, latest_price)
        change_percent_one_month = calculate_change_in_percent(one_month_ago_price, latest_price)

        momentum_data = calculate_momentum_indicators(one_month_full_data)

        if (change_percent_one_week > 0
            and change_percent_one_week < change_percent_two_weeks
            and change_percent_two_weeks > 0
            and change_percent_two_weeks < change_percent_three_weeks
            and change_percent_three_weeks > 0
            and momentum_data is not None and momentum_data["RSI"] < 70
        ):
            return {
                "Stock": stock,
                "1M Ago Price": one_month_ago_price,
                "Current Price": latest_price,
                "Change Last Week %": round(change_percent_one_week, 2) if change_percent_one_week else np.nan,
                "Change Last Two Weeks %": round(change_percent_two_weeks, 2) if change_percent_two_weeks else np.nan,
                "Change Last Three Weeks %": round(change_percent_three_weeks, 2) if change_percent_three_weeks else np.nan,
                "Change Last Month %": round(change_percent_one_month, 2) if change_percent_one_month else np.nan,
                "SMA_9": momentum_data["SMA_9"],
                "SMA_21": momentum_data["SMA_21"],
                "RSI": momentum_data["RSI"],
                "MACD": momentum_data["MACD"],
            }

    except Exception as e:
        print(f"Error processing {stock}: {e}")

    return None


def find_top_performing_stocks():
    """Find high-momentum Turkish stocks."""

    turkish_stocks = get_turkish_stock_symbols()
    print(f"Found {len(turkish_stocks)} Turkish stocks.")

    num_workers = max(cpu_count() - 1, 1)  # Use all CPU cores except 1
    results = process_map(process_stock, turkish_stocks, max_workers=num_workers)
    
    # Remove None values (stocks that did not meet criteria)
    results = [res for res in results if res is not None]

    # Convert results to DataFrame
    prices_df = pd.DataFrame(
        results,
        columns=[
            "Stock",
            "1M Ago Price",
            "Current Price",
            "Change Last Week %",
            "Change Last Two Weeks %",
            "Change Last Three Weeks %",
             "Change Last Month %",
            "SMA_9",
            "SMA_21",
            "RSI",
            "MACD",
        ],
    ).sort_values(by="RSI", ascending=True)

    if prices_df.empty:
        print("No stocks are detected!")
    else:
        print("\nTop Turkish Stocks with high momentum:")
        print(prices_df.to_string(index=False))


if __name__ == "__main__":
    set_start_method("fork")  # Fix multiprocessing for MacOS
    find_top_performing_stocks()
