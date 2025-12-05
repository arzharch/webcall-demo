# AI Restaurant Reservation Assistant - "Bella Cucina"

A production-ready voice-driven AI assistant that handles restaurant reservations, answers menu questions, and manages bookings through natural conversation using LangGraph orchestration.

## 🎯 Demo Overview

This project demonstrates:
- **Real-time voice conversations** via WebSocket
- **LangGraph-powered agent** with tool calling and state management
- **RAG (Retrieval-Augmented Generation)** for menu/restaurant info
- **Streaming TTS/STT** for natural conversation flow
- **CRM integration** for booking management

## 📋 Prerequisites

### System Requirements
- **Python 3.10+** (Python 3.11 recommended)
- **4GB RAM minimum** (8GB recommended for smooth model loading)
- **2GB disk space** (for downloaded models)

### Required API Keys

#### 1. Google Gemini API Key (REQUIRED)
The backend uses Google's Gemini AI for conversation management.

**Get your API key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key (format: `AIza...`)

**Cost:** Free tier includes 60 requests/minute

---

## 🚀 Complete Setup Guide

This guide covers the setup for the backend server.

#### Step 1: Navigate to the Backend Directory
From the project root, move into the backend folder.
```bash
cd backend
```

#### Step 2: Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

**Create the environment:**
```bash
python -m venv venv
```

**Activate the environment:**
- On Windows:
  ```bash
  .\venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

#### Step 3: Install Dependencies
Install all required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
*(Note: The first installation may take several minutes as it will download AI models for TTS and STT).*

#### Step 4: Configure Environment Variables
The application requires the Gemini API key to be set as an environment variable.

- Create a new file named `.env` inside this `backend` directory.
- Add your API key to this file in the following format:
  ```
  GEMINI_API_KEY="your_gemini_api_key_here"
  ```

#### Step 5: Run the Server
Once the setup is complete, you can start the FastAPI server.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
The `--reload` flag can be added for development to automatically restart the server on code changes (`uvicorn main:app --reload`).

The server is now running and accessible at `http://localhost:8000`.
