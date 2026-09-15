import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw OHLCV DataFrame and engineers technical features for ML training.
    """
    df = df.copy()

    # 1. 1-minute return (percentage change in this minute)
    df["return_1m"] = (df["close"] - df["open"]) / df["open"]

    # 2. Moving averages (5-period and 15-period)
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_15"] = df["close"].rolling(window=15).mean()

    # Normalize price against moving averages (ratio of current close to SMA)
    df["close_to_sma_5"] = df["close"] / df["sma_5"]
    df["close_to_sma_15"] = df["close"] / df["sma_15"]

    # 3. Volatility (Standard deviation of returns over 15 minutes)
    df["volatility_15"] = df["return_1m"].rolling(window=15).std()

    # 4. Volume surge (Current volume vs 15-minute average volume)
    df["volume_sma_15"] = df["volume"].rolling(window=15).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma_15"] + 1e-8)   # 1e-8 avoids division by zero

    # 5. Target label: 1 if NEXT minute close is higher than CURRENT close, else 0
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # 6. Cleaning: Rolling windows create NaN for the first 14 rows, and shift(-1) creates NaN for the last row
    df = df.dropna().reset_index(drop=True)

    return df

# List of columns our PyTorch model will actually use as input features
FEATURE_COLUMNS = [
    "return_1m",
    "close_to_sma_5",
    "close_to_sma_15",
    "volatility_15",
    "volume_ratio",
]

if __name__ == "__main__":
    print("Loading raw data from btc_historical.csv...")
    raw_df = pd.read_csv("btc_historical.csv")

    engineered_df = engineer_features(raw_df)

    # Save the clean features for training
    engineered_df.to_csv("btc_features.csv", index=False)

    print(f"Engineered dataset shape: {engineered_df.shape}")
    print(f"Target distribution (1=UP, 0=DOWN):\n{engineered_df['target'].value_counts(normalize=True)}")
    print("\nFirst 3 rows of features:")
    print(engineered_df[FEATURE_COLUMNS + ["target"]].head(3))