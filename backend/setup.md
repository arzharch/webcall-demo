# Production Setup Guide for Bella Voice AI

This comprehensive guide covers setting up the production infrastructure for the Bella Cucina voice AI reservation system, including Google Cloud Platform (GCP) deployment.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Google Cloud Platform Setup](#google-cloud-platform-setup)
4. [Redis Setup](#redis-setup)
5. [Environment Configuration](#environment-configuration)
6. [Deployment](#deployment)
7. [Monitoring & Observability](#monitoring--observability)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Accounts
- Google Cloud Platform account with billing enabled
- Deepgram account (for STT)
- OpenAI account (for LLM)

### Required Tools
- Python 3.9+
- Google Cloud CLI (`gcloud`)
- Docker (for containerized deployment)
- Redis (local or cloud)

---

## Local Development Setup

### 1. Install Python Dependencies

```bash
# Core application packages
pip install -r requirements.txt

# Production infrastructure packages
pip install redis>=4.5.0 tiktoken>=0.5.0

# OpenTelemetry packages (recommended for observability)
pip install opentelemetry-api>=1.20.0 opentelemetry-sdk>=1.20.0 opentelemetry-semantic-conventions>=0.41b0

# For GCP Cloud Trace integration
pip install opentelemetry-exporter-gcp-trace>=1.6.0
```

### 2. Set Up Local Redis

**Windows (WSL2):**
```bash
wsl --install
# In WSL:
sudo apt update && sudo apt install redis-server
sudo service redis-server start
redis-cli ping  # Should return PONG
```

**macOS:**
```bash
brew install redis
brew services start redis
redis-cli ping  # Should return PONG
```

**Docker:**
```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
docker exec -it redis redis-cli ping  # Should return PONG
```

### 3. Configure Environment

Create `.env` file in `backend/`:

```env
# === API Keys ===
GOOGLE_APPLICATION_CREDENTIALS=./synthion-demo-call-80b547a5bbbd.json
DEEPGRAM_API_KEY=your_deepgram_api_key
OPENAI_API_KEY=your_openai_api_key

# === Redis ===
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# === Environment ===
ENV=development
LOG_LEVEL=INFO
JSON_LOGS=false

# === Telemetry (Optional) ===
OTEL_ENABLED=true
OTEL_SERVICE_NAME=bella-voice-ai
OTEL_TRACE_EXPORTER=console
```

### 4. Run the Application

```bash
cd backend
python tinker_voice_v2.py
```

---

## Google Cloud Platform Setup

### Step 1: Create GCP Project

```bash
# Set project ID (replace with your project)
export PROJECT_ID="bella-voice-ai-prod"

# Create project
gcloud projects create $PROJECT_ID --name="Bella Voice AI"

# Set as default
gcloud config set project $PROJECT_ID

# Enable billing (required for most services)
# Do this in the GCP Console: https://console.cloud.google.com/billing
```

### Step 2: Enable Required APIs

```bash
# Enable all required APIs
gcloud services enable \
    texttospeech.googleapis.com \
    speech.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    redis.googleapis.com \
    secretmanager.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com \
    cloudtrace.googleapis.com \
    vpcaccess.googleapis.com
```

### Step 3: Create Service Account

```bash
# Create service account for the application
gcloud iam service-accounts create bella-voice-sa \
    --display-name="Bella Voice AI Service Account"

# Grant required roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:bella-voice-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/texttospeech.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:bella-voice-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/redis.editor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:bella-voice-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:bella-voice-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/cloudtrace.agent"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:bella-voice-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/logging.logWriter"

# Download key for local development (optional)
gcloud iam service-accounts keys create ./gcp-credentials.json \
    --iam-account=bella-voice-sa@$PROJECT_ID.iam.gserviceaccount.com
```

### Step 4: Set Up Secret Manager

Store sensitive keys in Secret Manager instead of environment variables:

```bash
# Store API keys
echo -n "your_deepgram_api_key" | gcloud secrets create deepgram-api-key \
    --data-file=-

echo -n "your_openai_api_key" | gcloud secrets create openai-api-key \
    --data-file=-

# Verify
gcloud secrets list
```

### Step 5: Create VPC Network (for Memorystore)

```bash
# Create VPC connector for Cloud Run to access Memorystore
# Using asia-south1 (Mumbai) for lowest latency in India
gcloud compute networks vpc-access connectors create redis-connector \
    --region=asia-south1 \
    --network=default \
    --range=10.8.0.0/28 \
    --min-instances=2 \
    --max-instances=10
```

### Step 6: Create Memorystore (Redis)

```bash
# Create Redis instance in Mumbai region
gcloud redis instances create bella-redis \
    --size=1 \
    --region=asia-south1 \
    --redis-version=redis_7_0 \
    --tier=basic

# Get the Redis IP (save this for later)
gcloud redis instances describe bella-redis --region=asia-south1 \
    --format="value(host)"
```

**Note:** Basic tier is ~$35/month. For production, consider Standard tier for HA.

### Step 7: Create Cloud Run Service

#### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
    redis>=4.5.0 \
    tiktoken>=0.5.0 \
    opentelemetry-api>=1.20.0 \
    opentelemetry-sdk>=1.20.0 \
    opentelemetry-exporter-gcp-trace>=1.6.0

# Copy application
COPY . .

# Health check endpoint (for Cloud Run)
EXPOSE 8080

# Use the HTTP server version for Cloud Run
CMD ["python", "main.py"]
```

#### Build and Deploy

```bash
# Build container
gcloud builds submit --tag gcr.io/$PROJECT_ID/bella-voice-ai

# Get Redis IP
REDIS_IP=$(gcloud redis instances describe bella-redis --region=asia-south1 --format="value(host)")

# Deploy to Cloud Run in Mumbai region
gcloud run deploy bella-voice-ai \
    --image=gcr.io/$PROJECT_ID/bella-voice-ai \
    --platform=managed \
    --region=asia-south1 \
    --service-account=synthionai@synthion-demo-call.iam.gserviceaccount.com \
    --vpc-connector=redis-connector \
    --set-env-vars="REDIS_HOST=$REDIS_IP,ENV=production,JSON_LOGS=true,OTEL_ENABLED=true" \
    --set-secrets="DEEPGRAM_API_KEY=deepgram-api-key:latest,OPENAI_API_KEY=openai-api-key:latest" \
    --min-instances=1 \
    --max-instances=10 \
    --memory=2Gi \
    --cpu=2 \
    --timeout=300 \
    --concurrency=80
```

---

## Redis Setup

### Option A: Local Redis (Development)

See [Local Development Setup](#2-set-up-local-redis)

### Option B: GCP Memorystore (Production)

See [Step 6: Create Memorystore](#step-6-create-memorystore-redis)

### Option C: Redis Cloud (Alternative)

1. Go to [Redis Cloud](https://redis.com/try-free/)
2. Create a free database
3. Copy the connection string
4. Set in `.env`:
   ```env
   REDIS_HOST=your-redis-cloud-host.redis-cloud.com
   REDIS_PORT=16379
   REDIS_PASSWORD=your-redis-password
   ```

### Redis Data Structure

The application uses Redis for:

| Key Pattern | Purpose | TTL |
|-------------|---------|-----|
| `tts:hash:*` | TTS audio cache | 24 hours |
| `llm:hash:*` | LLM response cache | 5 minutes |
| `session:*` | Call session state | 30 minutes |
| `active_calls` | Set of active session IDs | None |
| `transfer_queue` | Transfer ticket queue | 1 hour |
| `archive:YYYY-MM-DD` | Daily session archives | 7 days |

---

## Environment Configuration

### Development (.env)

```env
# API Keys
GOOGLE_APPLICATION_CREDENTIALS=./synthion-demo-call-80b547a5bbbd.json
DEEPGRAM_API_KEY=your_key
OPENAI_API_KEY=your_key

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# App Config
ENV=development
LOG_LEVEL=DEBUG
JSON_LOGS=false

# Telemetry
OTEL_ENABLED=true
OTEL_TRACE_EXPORTER=console
```

### Production (Cloud Run)

```bash
# Set via gcloud or Console
REDIS_HOST=10.0.0.3              # Memorystore IP
ENV=production
LOG_LEVEL=INFO
JSON_LOGS=true
OTEL_ENABLED=true
OTEL_TRACE_EXPORTER=gcp          # Sends to Cloud Trace
```

---

## Monitoring & Observability

### Cloud Logging

Logs are automatically sent to Cloud Logging when `JSON_LOGS=true`.

**View logs:**
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=bella-voice-ai" --limit=50
```

**Filter by session:**
```bash
gcloud logging read "jsonPayload.session_id=abc123" --limit=20
```

### Cloud Trace - Detailed Guide

Cloud Trace shows you **where time is spent** in each voice AI turn. It breaks down every operation (LLM calls, TTS synthesis, Redis cache lookups) into visual timelines.

#### What You'll See

Each "trace" represents **one complete turn** (user speaks → AI responds):

```
User: "Book a table for 2"
  ├─ STT (Deepgram)          [100ms]
  ├─ Intent Classification   [300ms]
  │  └─ LLM Call (OpenAI)    [280ms]
  ├─ Response Generation     [400ms]
  │  └─ LLM Call (OpenAI)    [380ms]
  └─ TTS Synthesis           [200ms]
     ├─ Redis Cache Check    [5ms]   ← Cache miss
     └─ Google TTS API       [195ms]
Total: 1000ms (1 second)
```

#### How to Access

**Method 1: GCP Console (Easiest)**

1. Go to: https://console.cloud.google.com/traces/list
2. Select project: **synthion-demo-call**
3. You'll see a timeline of all requests

**Method 2: Direct Link**

```
https://console.cloud.google.com/traces/list?project=synthion-demo-call
```

#### Understanding the Interface

**Main View - Trace List:**
- **Latency Graph**: Shows distribution of request times (p50, p95, p99)
- **Trace Table**: List of individual requests with duration
- **Timeline**: When requests occurred

**What to Look For:**

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Total Turn Time | < 1.5s | 1.5-3s | > 3s |
| LLM Call | < 500ms | 500ms-1s | > 1s |
| TTS (cached) | < 100ms | 100-200ms | > 200ms |
| TTS (fresh) | < 400ms | 400-800ms | > 800ms |

#### Analyzing a Trace (Step-by-Step)

1. **Click on any trace** in the list
2. You'll see a waterfall diagram with colored bars:
   - **Blue bars** = Your application code
   - **Green bars** = Successful external calls
   - **Red bars** = Errors
   - **Yellow bars** = Warnings

3. **Expand spans** to see details:
   ```
   Turn (1.2s total)
   ├─ llm_agent_call (800ms)
   │  ├─ intent_classification (300ms)
   │  │  └─ openai.chat.completions (280ms) ← Click for details
   │  └─ response_generation (500ms)
   └─ tts_synthesis (200ms)
      └─ google.cloud.texttospeech (195ms)
   ```

4. **Click on any span** to see:
   - Duration
   - Attributes (model name, input length, etc.)
   - Errors (if any)

#### Common Issues You'll Spot

**Slow LLM Calls (> 1s):**
```
Fix: Check if OpenAI is experiencing issues
Check: https://status.openai.com
```

**Slow TTS (> 500ms on first call):**
```
Normal: First synthesis is always slower
Fix: Pre-warm cache with common phrases
```

**Redis Timeouts:**
```
Symptom: Missing tts_synthesis spans
Fix: Check Redis connectivity
Command: gcloud redis instances describe bella-redis --region=asia-south1
```

#### Filtering Traces

**Find slow requests:**
```
Click "Filter" → Enter: @type:RootSpan AND duration > 2s
```

**Find errors:**
```
Click "Filter" → Enter: @type:RootSpan AND status:ERROR
```

**Find specific session:**
```
Click "Filter" → Enter: session_id:abc123
```

#### Example Analysis

**Good Trace (Fast Turn):**
```
✓ Total: 850ms
  ├─ LLM: 400ms (cached intent)
  ├─ TTS: 50ms (Redis cache hit!)
  └─ Audio: 400ms (playback)
```

**Bad Trace (Slow Turn):**
```
✗ Total: 3.2s
  ├─ LLM: 1.8s (timeout retry)
  ├─ TTS: 600ms (no cache)
  └─ Redis: TIMEOUT (circuit breaker opened)
  
Action: Check Redis connectivity
```

#### Setting Up Alerts

1. From a trace, click **"Create Alert"**
2. Set condition:
   ```
   Latency > 2000ms for 5 minutes
   ```
3. Add notification channel (email/SMS)

#### Command Line Access

```bash
# List recent traces
gcloud trace traces list --limit=10

# Get specific trace
gcloud trace traces describe TRACE_ID

# Export traces to BigQuery for analysis
gcloud trace sinks create trace-export \
    bigquery.googleapis.com/projects/synthion-demo-call/datasets/traces
```

#### Quick Health Check

**Good indicators:**
- Most traces < 1.5s
- TTS cache hit rate > 60%
- No red (error) spans
- Consistent latency (not spiky)

**Bad indicators:**
- Many traces > 3s
- Frequent timeouts
- Redis connection errors
- Circuit breakers opening

### Cloud Monitoring Dashboards

Create a custom dashboard:

1. Go to Cloud Monitoring
2. Create Dashboard
3. Add widgets:
   - **Latency**: Cloud Run Request Latencies
   - **Error Rate**: Cloud Run Error Count
   - **Active Sessions**: Custom metric (if exported)
   - **Redis Memory**: Memorystore metrics

### Alerting

Set up alerts for:

```bash
# High latency alert
gcloud alpha monitoring policies create \
    --notification-channels=YOUR_CHANNEL_ID \
    --display-name="High Latency Alert" \
    --condition-filter='resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_latencies"' \
    --condition-threshold-value=5000 \
    --condition-threshold-comparison=COMPARISON_GT
```

---

## Troubleshooting

### Redis Connection Errors

```bash
# Test connectivity from Cloud Run
gcloud run services describe bella-voice-ai --region=asia-south1 \
    --format="value(status.url)"

# Check VPC connector
gcloud compute networks vpc-access connectors describe redis-connector \
    --region=asia-south1

# Verify Memorystore IP
gcloud redis instances describe bella-redis --region=asia-south1
```

### Circuit Breaker Stays Open

1. Check logs for underlying errors:
   ```bash
   gcloud logging read "jsonPayload.message:\"circuit breaker\"" --limit=20
   ```

2. Check external service status:
   - OpenAI: https://status.openai.com
   - Google Cloud: https://status.cloud.google.com

### High Latency

1. Check Redis cache hit rate in logs
2. Review Cloud Trace for slow spans
3. Consider:
   - Upgrading Redis tier
   - Increasing Cloud Run memory/CPU
   - Pre-warming TTS cache

### Session Not Persisting

1. Verify Redis connection:
   ```python
   from infra import get_redis_client
   client = get_redis_client()
   print(client.ping())  # Should return True
   ```

2. Check session TTL isn't too short

---

## Cost Estimation

| Service | Estimated Monthly Cost |
|---------|----------------------|
| Cloud Run (min 1 instance) | $15-50 |
| Memorystore Basic (1GB) | $35 |
| Text-to-Speech | $4 per 1M chars |
| OpenAI GPT-3.5 | ~$0.002 per turn |
| Cloud Logging/Trace | Free tier usually covers |

**Total for light usage:** ~$60-100/month
**Total for production:** ~$150-300/month

---

## Next Steps

After Phase 2 setup:

1. **Phase 3**: A/B testing framework for prompts
2. **Phase 4**: Multi-region deployment
3. **Phase 5**: Telephony integration (Twilio/Vonage)
4. **Phase 6**: Analytics dashboard
