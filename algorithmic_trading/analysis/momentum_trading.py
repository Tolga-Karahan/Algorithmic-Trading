import yfinance as yf
import pandas as pd
import os

from tqdm.contrib.concurrent import process_map  # Parallel tqdm
from multiprocessing import cpu_count, set_start_method

BATCH_SIZE = 300


def get_us_tickers(n=-1):
    """Fetch Turkish stock symbols dynamically or read from a local file."""
    if "us_stocks.txt" not in os.listdir():
        return [
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
    else:
        print("Symbol file found, returning the symbols!")
        with open("us_stocks.txt", "r") as f:
            return f.read().split("\n")[:-1]
        
def create_tickers_batch(tickers):
    for i in range(len(tickers)//BATCH_SIZE+1):
        yield tickers[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
    yield tickers[i*BATCH_SIZE:]
        
def process_stock(ticker_batch):
    try:
        data = yf.download(ticker_batch, period="1d", interval="1m", group_by='ticker', progress=False)
    except Exception as e:
        print(f"Batch failed: {e}")
        return []
    
    matched = []
    for ticker in ticker_batch:
        try:
            df = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
            if df.empty or 'Open' not in df.columns or 'Close' not in df.columns:
                continue

            open_price = df.iloc[0]['Open']
            last_price = df.iloc[-1]['Close']
            percent_change = ((last_price - open_price) / open_price) * 100

            if 10 <= percent_change <= 15:
                matched.append({
                    "Ticker": ticker,
                    "Open": round(open_price, 2),
                    "Last": round(last_price, 2),
                    "% Change": round(percent_change, 2)
                })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    print(f"Matched: {matched}")
    return matched


def find_stocks():
    tickers = get_us_tickers()
    tickers = [batch for batch in create_tickers_batch(tickers)]

    num_workers = max(cpu_count() - 1, 1)  # Use all CPU cores except 1
    results = process_map(process_stock, tickers, max_workers=num_workers)

    # Remove None values (stocks that did not meet criteria)
    results = [res for res in results if res is not None]

    if results:
        return pd.DataFrame(
            results, 
            columns=["Ticker", "Last", "% Change"]
        ).sort_values(by="% Change", ascending=True)
    else:
        print("Empty dataframe!")


if __name__ == "__main__":
    set_start_method("fork")  # Fix multiprocessing for MacOS
    print(find_stocks())