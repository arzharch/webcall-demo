# Use Python 3.11 (Cloud Run officially supports 3.11, 3.13 may have issues)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install production infrastructure packages
RUN pip install --no-cache-dir \
    redis>=4.5.0 \
    tiktoken>=0.5.0 \
    opentelemetry-api>=1.20.0 \
    opentelemetry-sdk>=1.20.0 \
    opentelemetry-semantic-conventions>=0.41b0 \
    opentelemetry-exporter-gcp-trace>=1.6.0

# Copy application code
COPY . .

# Create a simple HTTP server wrapper for Cloud Run
# Cloud Run requires an HTTP server on port 8080
COPY <<'EOF' /app/main.py
"""
HTTP server wrapper for Cloud Run deployment.
Provides health endpoints and keeps the service alive.
"""
import os
import asyncio
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Import health checker
from infra import get_health_checker, health_endpoint_handler, liveness_endpoint_handler, readiness_endpoint_handler

app = FastAPI(title="Bella Voice AI")
logger = logging.getLogger(__name__)

@app.get("/")
async def root():
    return {"service": "Bella Voice AI", "status": "running", "version": "2.0"}

@app.get("/health")
async def health():
    result, status_code = health_endpoint_handler()
    return JSONResponse(content=result, status_code=status_code)

@app.get("/healthz")
async def liveness():
    result, status_code = liveness_endpoint_handler()
    return JSONResponse(content=result, status_code=status_code)

@app.get("/readyz")
async def readiness():
    result, status_code = readiness_endpoint_handler()
    return JSONResponse(content=result, status_code=status_code)

@app.get("/stats")
async def stats():
    """Session and infrastructure stats."""
    from infra import get_session_manager, get_telemetry
    
    session_manager = get_session_manager()
    telemetry = get_telemetry()
    
    return {
        "sessions": session_manager.get_session_stats(),
        "telemetry": telemetry.get_health(),
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    logger.info(f"Starting Bella Voice AI HTTP server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
EOF

# Expose port 8080 (Cloud Run requirement)
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run the HTTP server
CMD ["python", "main.py"]
