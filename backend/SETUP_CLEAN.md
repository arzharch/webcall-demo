# Bella Cucina Voice Bot - Backend Setup

## Overview
Clean, modular backend for AI restaurant voice assistant with STT, LLM, and TTS.

**Flow:** Audio → Deepgram STT → Gemini LLM → Google TTS → Audio + Transcripts + Tickets

---

## Prerequisites
- Python 3.10+
- API Keys (all have free tiers):
  - Gemini API (already have)
  - Deepgram API
  - Google Cloud TTS

---

## Setup Instructions

### 1. Get API Keys

#### Deepgram (STT)
1. Visit: https://deepgram.com
2. Sign up for free account
3. Get API key from dashboard
4. **Free tier:** 45 minutes/month

#### Google Cloud TTS
1. Visit: https://console.cloud.google.com
2. Create new project
3. Enable "Cloud Text-to-Speech API"
4. Create service account:
   - Go to IAM & Admin → Service Accounts
   - Create Service Account
   - Grant role: "Cloud Text-to-Speech User"
   - Create JSON key
   - Download the JSON file
5. **Free tier:** 1 million characters/month

### 2. Configure Environment

Create/update `.env` file:
```bash
GEMINI_API_KEY="your_existing_gemini_key"
DEEPGRAM_API_KEY="paste_deepgram_key_here"
GOOGLE_APPLICATION_CREDENTIALS="path/to/google-credentials.json"
```

**Important:** Place the Google Cloud JSON key file in the backend folder and update the path in `.env`.

### 3. Install Dependencies

```bash
pip install -r requirements_clean.txt
```

### 4. Run Server

```bash
python main_clean.py
```

Server starts at: `http://localhost:8000`

---

## API Endpoints

### REST Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /session/start` - Create new call session
- `GET /transcripts` - List all call transcripts
- `GET /transcripts/{call_id}` - Get specific transcript
- `GET /tickets` - List all booking tickets
- `GET /tickets/{ticket_id}` - Get specific ticket

### WebSocket
- `WS /ws/audio/{call_id}` - Real-time voice conversation
  - Send: Audio bytes (PCM 16kHz)
  - Receive: Audio bytes + transcript JSON

---

## Project Structure

```
backend/
├── main_clean.py              # FastAPI app + WebSocket
├── config_clean.py            # Settings
├── models_clean.py            # Data models
├── prompts_clean.py           # System prompt
├── stt_service_clean.py       # Deepgram STT
├── tts_service_clean.py       # Google Cloud TTS
├── llm_service_clean.py       # Gemini LLM
├── storage_service_clean.py   # JSON storage
├── requirements_clean.txt     # Dependencies
├── .env                       # API keys (not in git)
└── data/
    ├── transcripts/           # Call transcripts
    └── tickets/               # Booking tickets
```

---

## Features

### Call Transcription
- Every call saved with timestamp
- Full message history (user + assistant)
- Automatic summary generation
- Call quality rating (1-5)

### Booking Detection
- Gemini analyzes conversation for booking intent
- Extracts: name, date, time, party size, phone
- Auto-creates ticket when booking confirmed
- Stores in `data/tickets/`

### Rating System
Based on:
- Booking successful? (+1)
- Conversation length/engagement (+1)
- Base score: 3/5

---

## Usage Example

### Frontend Connection
```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/audio/${callId}`);

// Send audio
ws.send(audioBytes);

// Receive responses
ws.onmessage = (event) => {
  if (event.data instanceof Blob) {
    // Audio response - play it
  } else {
    // Transcript JSON
    const data = JSON.parse(event.data);
    console.log(data.role, data.content);
  }
};
```

---

## Testing

### Quick Test
```bash
curl http://localhost:8000/health
```

### Create Session
```bash
curl -X POST http://localhost:8000/session/start
```

### List Transcripts
```bash
curl http://localhost:8000/transcripts
```

---

## Cost Estimates (Free Tiers)

- **Deepgram:** 45 min/month = ~90 calls (30 sec each)
- **Google TTS:** 1M chars = ~5000 responses
- **Gemini:** Generous free tier

**Total:** 100% free for testing/POC

---

## Troubleshooting

### Google Cloud TTS Not Working
- Verify JSON key file path in `.env`
- Ensure "Cloud Text-to-Speech API" is enabled
- Check service account has correct role

### Deepgram Errors
- Verify API key is correct
- Check free tier quota (45 min/month)

### WebSocket Connection Issues
- Check CORS settings in `config_clean.py`
- Verify frontend URL in `CORS_ORIGINS`

---

## Next Steps

1. Test with Postman/curl
2. Connect frontend
3. Make test calls
4. Review transcripts in `data/transcripts/`
5. Check tickets in `data/tickets/`

---

## Notes

- All data stored in JSON files (simple, portable)
- No database needed for POC
- No VAD (Voice Activity Detection) complexity
- Clean, modular architecture
- Easy to extend/modify
