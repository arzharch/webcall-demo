# ✅ CLEAN BACKEND - READY TO USE

## What Was Done
Completely rebuilt backend from scratch with modular, clean architecture.

---

## Files to Use (All with `_clean` suffix)

### Core Files
1. **main_clean.py** - FastAPI application (START HERE)
2. **config_clean.py** - Configuration & settings
3. **models_clean.py** - Data models (Pydantic)
4. **prompts_clean.py** - System prompt for restaurant assistant

### Services (All in root directory)
5. **stt_service_clean.py** - Speech-to-Text (Deepgram)
6. **tts_service_clean.py** - Text-to-Speech (Google Cloud)
7. **llm_service_clean.py** - LLM (Gemini + booking detection)
8. **storage_service_clean.py** - JSON file storage

### Dependencies & Setup
9. **requirements_clean.txt** - Python packages to install
10. **SETUP_CLEAN.md** - Complete setup instructions
11. **.env** - Updated with new API key requirements

---

## Quick Start

### 1. Get API Keys
- **Deepgram:** https://deepgram.com (STT - 45 min free/month)
- **Google Cloud:** https://console.cloud.google.com (TTS - 1M chars free/month)
- **Gemini:** Already have it ✅

### 2. Setup .env
Already configured, just add:
- Deepgram API key
- Google Cloud credentials path

### 3. Install
```bash
pip install -r requirements_clean.txt
```

### 4. Run
```bash
python main_clean.py
```

Server: http://localhost:8000

---

## Architecture

```
Audio → Deepgram STT → Gemini LLM → Google TTS → Audio
                                ↓
                        Store Transcript + Ticket
```

---

## Features

✅ Real-time voice conversation
✅ Automatic transcription (saved with timestamps)
✅ Booking detection (auto-creates tickets)
✅ Call summaries (AI-generated)
✅ Call ratings (1-5 scale)
✅ JSON storage (no database needed)

---

## Next Steps

1. Read `SETUP_CLEAN.md` for detailed instructions
2. Get Deepgram and Google Cloud API keys
3. Install dependencies
4. Run the server
5. Test with frontend
