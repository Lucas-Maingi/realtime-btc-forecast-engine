import json
import time
from datetime import datetime, timezone, timedelta
import requests
import websocket
import pandas as pd
import numpy as np

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"
API_URL = "http://127.0.0.1:8000/predict"

# Timezone definition for UTC-4
UTC_MINUS_4 = timezone(timedelta(hours=-4))

# Buffer to hold candles
candles_buffer = []


def warm_up_buffer(symbol="BTCUSDT", limit=15):
    """Fetches the last 15 closed candles via REST API so we can predict immediately!"""
    global candles_buffer
    print(f"Pre-warming buffer with last {limit} historical candles from Binance...")
    
    response = requests.get(BINANCE_REST_URL, params={"symbol": symbol, "interval": "1m", "limit": limit})
    raw = response.json()
    
    candles_buffer = []
    for row in raw:
        candles_buffer.append({
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5])
        })
    print(f"Buffer warmed up! ({len(candles_buffer)}/{limit} candles ready). Live predictions start on NEXT candle!\n")


def on_message(ws, message):
    global candles_buffer

    data = json.loads(message)
    kline = data.get("k", {})

    is_closed = kline.get("x", False)
    current_close = float(kline.get("c", 0.0))

    if is_closed:
        # Extract candle close timestamp and convert to UTC-4
        close_time_ms = kline.get("T", int(time.time() * 1000))
        dt = datetime.fromtimestamp(close_time_ms / 1000, tz=UTC_MINUS_4)
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S [UTC-4]")

        print(f"\n[{time_str}] [1M CANDLE CLOSED] BTC Price: ${current_close:,.2f}")

        candles_buffer.append({
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": current_close,
            "volume": float(kline["v"])
        })

        if len(candles_buffer) > 20:
            candles_buffer.pop(0)

        df = pd.DataFrame(candles_buffer)

        # Compute features
        return_1m = (df["close"].iloc[-1] - df["open"].iloc[-1]) / df["open"].iloc[-1]
        sma_5 = df["close"].tail(5).mean()
        sma_15 = df["close"].tail(15).mean()
        close_to_sma_5 = current_close / sma_5
        close_to_sma_15 = current_close / sma_15

        returns = (df["close"] - df["open"]) / df["open"]
        volatility_15 = returns.tail(15).std()
        if np.isnan(volatility_15):
            volatility_15 = 0.0005

        volume_sma_15 = df["volume"].tail(15).mean()
        volume_ratio = df["volume"].iloc[-1] / (volume_sma_15 + 1e-8)

        payload = {
            "return_1m": float(return_1m),
            "close_to_sma_5": float(close_to_sma_5),
            "close_to_sma_15": float(close_to_sma_15),
            "volatility_15": float(volatility_15),
            "volume_ratio": float(volume_ratio)
        }

        start_time = time.time()
        try:
            response = requests.post(API_URL, json=payload)
            latency_ms = (time.time() - start_time) * 1000
            result = response.json()

            print(f"  [AI PREDICTION] Direction: {result['direction']} ({result['up_probability']*100:.2f}% UP)")
            print(f"  [SIGNAL]        {result['signal']} (Confidence: {result['confidence_percentage']}%)")
            print(f"  [LATENCY]       {latency_ms:.2f} ms")
        except Exception as e:
            print(f"  [Error contacting API]: {e}")


def on_open(ws):
    print("[Connected to Binance Live WebSocket Stream]")
    print("Listening for live BTC/USDT trades...")


if __name__ == "__main__":
    warm_up_buffer()
    ws = websocket.WebSocketApp(
        BINANCE_WS_URL,
        on_open=on_open,
        on_message=on_message
    )
    ws.run_forever()