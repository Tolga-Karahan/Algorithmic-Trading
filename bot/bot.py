import os
import requests
import time

import ccxt
import pandas as pd

class TradingBot:
    def __init__(self):
        # ✅ Initialize Binance API
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_SECRET_KEY")
        self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'options': {'defaultType': 'future'},  # Use futures if trading leverage
        })


    def get_btc_data(self, interval, limit, start_time=None):
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


    def calculate_vwap(self, df, period_candles=None, name="VWAP"):
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

    def create_order(self):
        # ✅ Trading Parameters
        symbol = "BTC/USDT"
        risk_percentage = 0.5  # 0.5% risk per trade
        rrr = 2  # Risk-to-Reward Ratio (1:2)

        # ✅ Fetch Latest Price
        ticker = self.exchange.fetch_ticker(symbol)
        entry_price = ticker["last"]  # Current market price

        # ✅ Calculate Stop Loss & Take Profit
        stop_loss = entry_price * (1 - (risk_percentage / 100))  # 0.5% below entry
        take_profit = entry_price * (1 + (risk_percentage * rrr / 100))  # 1% above entry

        # ✅ Check Account Balance
        balance = self.exchange.fetch_balance()
        usdt_balance = balance["total"]["USDT"]
        trade_size = (usdt_balance * 0.1) / entry_price  # Risk 10% of balance

        try:
            # ✅ Place Limit Buy Order and Verify Response
            print(f"📌 Placing Limit Order at {entry_price} for {trade_size} {symbol}...")
            try:
                limit_order = self.exchange.create_limit_buy_order(symbol, trade_size, entry_price)
            except Exception as e:
                print("Error while creating limit order!")
                print(e)

            # Ensure order contains an ID
            if not limit_order or "id" not in limit_order:
                raise Exception("❌ Order placement failed, no valid order ID returned.")

            order_id = limit_order["id"]  # Extract order ID
            print(f"✅ Limit Order Placed: {order_id}")

            # ✅ Wait for Order Execution
            order_filled = False
            while not order_filled:
                open_orders = self.exchange.fetch_open_orders(symbol)
                
                # If order is still open, wait
                if any(order["id"] == order_id for order in open_orders):
                    print("⌛ Order still open, waiting for execution...")
                else:
                    # Check if the order is closed
                    closed_orders = self.exchange.fetch_closed_orders(symbol)
                    if any(order["id"] == order_id for order in closed_orders):
                        print("✅ Order Executed!")
                        order_filled = True
                    else:
                        print("⚠ Order not in closed orders yet, waiting...")

            # ✅ Set Stop Loss & Take Profit (OCO Order - One Cancels Other)
            print(f"📌 Placing OCO Order: TP {take_profit}, SL {stop_loss}...")
            oco_order = self.exchange.create_order(
                symbol=symbol,
                type="OCO",  # One Cancels Other
                side="sell",
                amount=trade_size,
                price=take_profit,  # Take Profit Price
                params={"stopPrice": stop_loss}  # Stop Loss Price
            )

            print(f"✅ Stop Loss Set at {stop_loss} | Take Profit Set at {take_profit}")

        except Exception as e:
            print(f"❌ Error: {e}")


    # ✅ Function to Check if Open Orders Exist
    def has_open_order(self, symbol="BTC/USDT"):
        open_orders = self.exchange.fetch_open_orders(symbol)
        return len(open_orders) > 0


    # ✅ Function to Continuously Fetch Data & Execute Orders
    def run_trading_bot(self):
        while True:
            try:
                if self.has_open_order():
                    print("Open Order Exists. Skipping Trade.")
                else:
                    print("🔄 Fetching latest BTC/USDT data...")

                    # ✅ Fetch the latest 4-hour interval BTC data
                    limit = 6  # Fetch last 6 candles (24 hours)
                    data = self.get_btc_data(interval="4h", limit=limit)

                    # ✅ Calculate VWAPs dynamically
                    data = self.calculate_vwap(data, period_candles=1, name="4-Hourly VWAP")
                    data = self.calculate_vwap(data, period_candles=6, name="Daily VWAP")

                    # ✅ Check for VWAP Crossover (Bullish & Bearish)
                    prev_4h_vwap = data["4-Hourly VWAP"].iloc[-2]
                    curr_4h_vwap = data["4-Hourly VWAP"].iloc[-1]
                    prev_daily_vwap = data["Daily VWAP"].iloc[-2]
                    curr_daily_vwap = data["Daily VWAP"].iloc[-1]

                    # 🟢 Bullish Crossover (4H VWAP crosses above Daily VWAP)
                    if prev_4h_vwap < prev_daily_vwap and curr_4h_vwap > curr_daily_vwap:
                            print("✅ Bullish VWAP Crossover Detected! Placing Buy Order...")
                            create_order()

                    # ✅ Wait before fetching new data (e.g., every 10 seconds)
                    time.sleep(10)

            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(10)  # Wait before retrying
            

if __name__ == "__main__":
    bot = TradingBot()
    # ✅ Run the Trading Bot
    bot.run_trading_bot()
       