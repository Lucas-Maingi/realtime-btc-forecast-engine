# 1. Official lightweight Python runtime
FROM python:3.11-slim

# 2. Prevent Python from buffering outputs (real-time logs)
ENV PYTHONUNBUFFERED=1

# 3. Set working directory inside container
WORKDIR /app

# 4. Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the code and trained model weights into container
COPY features.py .
COPY model.py .
COPY server.py .
COPY model.pth .
COPY scaler.joblib .

# 6. Expose port 8000 for the API
EXPOSE 8000

# 7. Start FastAPI server listening on 0.0.0.0
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]