# Cloud Run Deployment Guide

## Prerequisites

✅ You should have completed:
- Created VPC connector
- Created Redis (Memorystore)
- Stored secrets in Secret Manager

## Step 1: Build and Push Container

```bash
# Set your project
export PROJECT_ID="synthion-demo-call"
gcloud config set project $PROJECT_ID

# Build and push to Google Container Registry
cd backend
gcloud builds submit --tag gcr.io/$PROJECT_ID/bella-voice-ai

# This takes ~5-10 minutes for first build
```

## Step 2: Get Redis IP

```bash
# Get Redis host IP
REDIS_IP=$(gcloud redis instances describe bella-redis \
    --region=asia-south1 \
    --format="value(host)")

echo "Redis IP: $REDIS_IP"
# Save this IP for next step
```

## Step 3: Deploy to Cloud Run

```bash
# Deploy with all configurations
gcloud run deploy bella-voice-ai \
    --image=gcr.io/$PROJECT_ID/bella-voice-ai \
    --platform=managed \
    --region=asia-south1 \
    --service-account=synthionai@synthion-demo-call.iam.gserviceaccount.com \
    --vpc-connector=redis-connector \
    --set-env-vars="REDIS_HOST=$REDIS_IP,REDIS_PORT=6379,ENV=production,JSON_LOGS=true,OTEL_ENABLED=true,OTEL_TRACE_EXPORTER=gcp" \
    --set-secrets="DEEPGRAM_API_KEY=deepgram-api-key:latest,OPENAI_API_KEY=openai-api-key:latest,GOOGLE_APPLICATION_CREDENTIALS=gcp-credentials:latest" \
    --min-instances=1 \
    --max-instances=10 \
    --memory=2Gi \
    --cpu=2 \
    --timeout=300 \
    --concurrency=80 \
    --allow-unauthenticated

# Takes ~2-3 minutes
```

### Explanation of Flags

| Flag | Purpose |
|------|---------|
| `--region=asia-south1` | Mumbai region (low latency) |
| `--vpc-connector` | Access to Redis in VPC |
| `--min-instances=1` | Keep 1 warm instance (no cold starts) |
| `--memory=2Gi` | Enough for models + caching |
| `--cpu=2` | 2 vCPUs for fast processing |
| `--timeout=300` | 5 min timeout for long calls |
| `--concurrency=80` | Max 80 requests per instance |
| `--allow-unauthenticated` | Public access (remove for private) |

## Step 4: Test Deployment

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe bella-voice-ai \
    --region=asia-south1 \
    --format="value(status.url)")

echo "Service URL: $SERVICE_URL"

# Test health endpoint
curl $SERVICE_URL/health

# Test readiness
curl $SERVICE_URL/readyz

# View stats
curl $SERVICE_URL/stats
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": 1234567890.123,
  "components": {
    "redis": {"status": "healthy", "latency_ms": 2.5},
    "circuit_breakers": {"status": "healthy"},
    "telemetry": {"status": "healthy"}
  }
}
```

## Step 5: View Logs

```bash
# Stream logs
gcloud run services logs tail bella-voice-ai --region=asia-south1

# View last 50 logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=bella-voice-ai" --limit=50
```

## Step 6: Monitor Performance

### Cloud Trace
```
https://console.cloud.google.com/traces/list?project=synthion-demo-call
```

### Cloud Logging
```
https://console.cloud.google.com/logs/query?project=synthion-demo-call
```

### Cloud Run Metrics
```
https://console.cloud.google.com/run/detail/asia-south1/bella-voice-ai?project=synthion-demo-call
```

## Updating the Service

### Option 1: Quick Update (same image)
```bash
gcloud run services update bella-voice-ai \
    --region=asia-south1 \
    --set-env-vars="NEW_VAR=value"
```

### Option 2: Full Rebuild
```bash
# Rebuild image
gcloud builds submit --tag gcr.io/$PROJECT_ID/bella-voice-ai

# Deploy new version
gcloud run deploy bella-voice-ai \
    --image=gcr.io/$PROJECT_ID/bella-voice-ai \
    --region=asia-south1
```

## Troubleshooting

### Container Won't Start
```bash
# Check build logs
gcloud builds list --limit=5

# Check service status
gcloud run services describe bella-voice-ai --region=asia-south1
```

### Can't Connect to Redis
```bash
# Verify VPC connector
gcloud compute networks vpc-access connectors describe redis-connector \
    --region=asia-south1

# Check Redis IP
gcloud redis instances describe bella-redis --region=asia-south1
```

### High Latency
```bash
# Check if warm instance is running
gcloud run services describe bella-voice-ai \
    --region=asia-south1 \
    --format="value(spec.template.spec.containers[0].resources.limits.memory)"

# Increase resources if needed
gcloud run services update bella-voice-ai \
    --region=asia-south1 \
    --memory=4Gi \
    --cpu=4
```

## Cost Optimization

### Reduce Costs (Development)
```bash
# Use min-instances=0 (cold starts OK)
gcloud run services update bella-voice-ai \
    --region=asia-south1 \
    --min-instances=0 \
    --memory=1Gi \
    --cpu=1
```

### Increase Performance (Production)
```bash
# Keep warm, more resources
gcloud run services update bella-voice-ai \
    --region=asia-south1 \
    --min-instances=2 \
    --memory=4Gi \
    --cpu=4 \
    --max-instances=20
```

## Rollback

```bash
# List revisions
gcloud run revisions list --service=bella-voice-ai --region=asia-south1

# Rollback to previous
gcloud run services update-traffic bella-voice-ai \
    --region=asia-south1 \
    --to-revisions=REVISION_NAME=100
```

## Clean Up

```bash
# Delete service
gcloud run services delete bella-voice-ai --region=asia-south1

# Delete container images
gcloud container images delete gcr.io/$PROJECT_ID/bella-voice-ai
```
