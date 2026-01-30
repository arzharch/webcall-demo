# AI Restaurant Reservation Assistant - "Bella Cucina"

This project is a voice-driven AI assistant that handles restaurant reservations, answers menu questions, and manages bookings through natural conversation.

## Architecture Overview

The system is designed with a two-phased architecture:

1.  **Phase 1 (Core MVP):** A functional, modular voice bot built with FastAPI, using separate services for STT, TTS, LLM (Gemini), and RAG.
2.  **Phase 2 (Advanced Streaming):** An evolution of the MVP into a low-latency, interruptible agent using LangChain as an orchestrator (the "Streaming Brain") for a more fluid conversational experience.

For a detailed breakdown of the architecture, components, and key concepts, please refer to the `workflow.md` file.

## Project Setup

### 1. Prerequisites
- Python 3.11+
- An environment management tool like `venv` or `conda`.
- Node.js and `npm` (for the frontend, if you choose to build it).

### 2. Backend Setup

Navigate to the project root directory (`voice-bot-mvp`).

**a. Create a Virtual Environment:**
```bash
python -m venv venv
```

**b. Activate the Environment:**
- On Windows:
  ```bash
  .\venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

**c. Install Dependencies:**
All required Python packages are listed in `backend/requirements.txt`. Install them using pip:
```bash
pip install -r backend/requirements.txt
```

**d. Configure Environment Variables:**
The backend requires an API key for the Gemini LLM.

- Create a file named `.env` inside the `backend` directory.
- Add your API key to this file in the following format:
  ```
  GOOGLE_API_KEY="your_GOOGLE_API_KEY_here"
  ```

### 3. Frontend Setup

This project includes a basic React frontend to interact with the voice bot.

- **Navigate to the frontend directory:**
  ```bash
  cd frontend
  ```

- **Install Dependencies:**
  Install the required Node.js packages using `npm`.
  ```bash
  npm install
  ```

### 4. Running the Application

**a. Backend**
First, ensure your backend server is running.

- **Navigate to the backend directory:**
  ```bash
  cd backend
  ```
- **Run the FastAPI server:**
  ```bash
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
  ```
The server will be accessible at `http://localhost:8000`.

**b. Frontend**
With the backend running, open a *new* terminal for the frontend.

- **Navigate to the frontend directory:**
  ```bash
  cd frontend
  ```
- **Start the React development server:**
  ```bash
  npm start
  ```
This will open the application in your web browser, usually at `http://localhost:3000`. You can then use the UI to start a voice call with the 

---
*This project is being built by an AI agent. The setup instructions will be updated as the project progresses.*
