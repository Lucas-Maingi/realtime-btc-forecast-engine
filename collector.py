import time
import requests
import pandas as pd

BASE_URL = "https://api.binance.com/api/v3/klines"


def fetch_historical_data(symbol="BTCUSDT", interval="1m", total_candles=10000):
    """
    Downloads historical klines from Binance using pagination.
    Loops backwards in time to fetch more than the 1000-candle limit.
    """
    all_candles = []
    end_time = None
    batch_size = 1000
    batches_needed = total_candles // batch_size

    print(f"Fetching {total_candles} candles in {batches_needed} batches...")

    for i in range(batches_needed):
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": batch_size
        }
        if end_time:
            params["endTime"] = end_time

        response = requests.get(BASE_URL, params=params)
        raw_data = response.json()

        if not raw_data:
            print("No more data returned from Binance.")
            break

        # Extract OHLCV
        for row in raw_data:
            all_candles.append({
                "timestamp": row[0],
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5])
            })

        # Set end_time to 1 millisecond before the oldest candle in this batch
        oldest_timestamp = raw_data[0][0]
        end_time = oldest_timestamp - 1

        print(f"  Batch {i + 1}/{batches_needed} downloaded (oldest timestamp: {oldest_timestamp})")
        time.sleep(0.1)  # small pause so Binance doesn't rate-limit us

    df = pd.DataFrame(all_candles)

    # Sort chronologically (oldest first, newest last) and drop any duplicates
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Add a human-readable datetime column
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df


if __name__ == "__main__":
    df = fetch_historical_data(total_candles=10000)
    df.to_csv("btc_historical.csv", index=False)
    print(f"\nSuccessfully saved {len(df)} rows to btc_historical.csv")
    print(df[["datetime", "open", "high", "low", "close", "volume"]].head())