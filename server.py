import joblib
import numpy as np
from pydantic import BaseModel, Field
import torch
from fastapi import FastAPI, HTTPException

from features import FEATURE_COLUMNS
from model import BTCPriceClassifier

# 1. Initialize FastAPI app
app =FastAPI(
    title="Real-Time BTC Forecast Engine",
    description="Production inference API serving directional Bitcoin price predictions.",
    version="1.0.0"
)

# 2. Global variables to hold model and scaler in memory
model = None
scaler = None


@app.on_event("startup")
def load_artifacts():
    """Loads the PyTorch model and the feature scaler into RAM when server starts."""
    global model, scaler

    print("Loading scaler.joblib...")
    scaler = joblib.load("scaler.joblib")

    print("Loading model.pth...")
    model = BTCPriceClassifier(input_dim=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()   # Put model in evaluation mode (turns off dropout)

    print("All artifacts successfully loaded into memory.")

# 3. Pydantic Schema for incoming trade features
# 3. Pydantic Schema for incoming trade features
class MarketFeatures(BaseModel):
    return_1m: float = 0.00068
    close_to_sma_5: float = 1.00078
    close_to_sma_15: float = 1.00063
    volatility_15: float = 0.00052
    volume_ratio: float = 0.6577
    
# 4. Health check endpoint
@app.get("/health")
def health_check():
    return{"status": "healthy", "model_loaded": model is not None}

# 5. Prediction endpoint
@app.post("/predict")
def predict_direction(features: MarketFeatures):
    if model is None or scaler is None:
        raise HTTPExecption(status_code=503, detail="Model is not ready.")

    # Convert incoming JSON features to numpy array in the exact feature column order
    raw_features = np.array([[
        features.return_1m,
        features.close_to_sma_5,
        features.close_to_sma_15,
        features.volatility_15,
        features.volume_ratio
    ]])

    # Scale the features using the saved scaler (transform, NOT fit_transform!)
    scaled_features = scaler.transform(raw_features)

    # Convert to PyTorch tensor and run inference
    tensor_input = torch.tensor(scaled_features, dtype=torch.float32)

    with torch.no_grad():
        probability = model(tensor_input).item()

    # Determine trading signal
    if probability >= 0.55:
        signal = "BUY"
        direction = "UP"

    elif probability <= 0.45:
        signal = "SELL"
        direction = "DOWN"

    else:
        signal = "HOLD"
        direction = "NEUTRAL"

    return{
        "direction": direction,
        "up_probability": round(probability, 4),
        "confidence_percentage": round(abs(probability - 0.5) * 200, 2),
        "signal": signal
    }
