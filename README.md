markdown


# Real-Time BTC Forecast Engine
A production-grade machine learning pipeline that connects to Binance's live WebSocket stream, engineers features from raw Bitcoin trade data in real-time, and serves directional price predictions (UP/DOWN) through a FastAPI endpoint — all packaged in Docker.
## Architecture
Binance WebSocket (Live Trades) │ ▼ collector.py ──► Captures raw trade stream via WebSocket │ ▼ features.py ──► Engineers rolling averages, volume & volatility │ ▼ model.py ──► PyTorch binary classifier (price direction) │ ▼ train.py ──► Trains on historical Binance kline data │ ▼ server.py ──► FastAPI serves real-time predictions │ ▼ Dockerfile ──► Containerized for cloud deployment



## Tech Stack
| Layer | Technology |
| :--- | :--- |
| Data Ingestion | Binance WebSocket API |
| Data Engineering | Pandas |
| Model | PyTorch |
| API | FastAPI + Pydantic |
| Containerization | Docker |
| Language | Python 3.9+ |
## Features
- **Live streaming data** — Connects directly to Binance's public WebSocket
- **Feature engineering pipeline** — Rolling averages, trade volume, price volatility
- **Binary classification** — Predicts short-term BTC price direction (UP/DOWN)
- **Production-ready API** — JSON in, JSON out via FastAPI
- **Containerized** — Single `docker build` deploys the full stack
## Getting Started
```bash
git clone https://github.com/Lucas-Maingi/realtime-btc-forecast-engine.git
cd realtime-btc-forecast-engine
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate
pip install -r requirements.txt
License
MIT



