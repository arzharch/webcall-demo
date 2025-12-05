# API Documentation - Bella Cucina Voice Bot

This document outlines the API endpoints provided by the backend server.

## REST API

The REST API provides endpoints for health checks and basic CRM functionality.

### Health Check

- **Endpoint:** `GET /`
- **Description:** A simple health check endpoint to verify that the server is running and accessible.
- **Success Response (200 OK):**
  ```json
  {
    "status": "ok",
    "message": "Welcome to the Voice Bot Streaming API!"
  }
  ```

### List CRM Tickets

- **Endpoint:** `GET /tickets`
- **Description:** Retrieves a list of all reservation tickets that have been created and stored in the CRM system. This is useful for debugging and verifying that reservations are being created correctly.
- **Success Response (200 OK):**
  A JSON array of `Ticket` objects.
  ```json
  [
    {
      "id": "ticket_abc123de",
      "call_id": "call_1678886400",
      "timestamp": "2025-03-15T12:00:00.000Z",
      "customer_name": "John Smith",
      "customer_phone": null,
      "intent": "reservation",
      "details": {
        "date": "2025-03-22",
        "time": "19:00",
        "party_size": 4,
        "customer_name": "John Smith"
      },
      "status": "confirmed",
      "transcript": [],
      "summary": "Reservation for John Smith"
    }
  ]
  ```

---

## WebSocket API

The WebSocket API is the primary interface for the real-time voice conversation.

### Voice Conversation Endpoint

- **Endpoint:** `WS /ws/audio/{call_id}`
- **Description:** Handles the main voice conversation flow. It establishes a persistent, bidirectional connection for streaming audio data between the client and the server.
- **URL Parameter:**
  - `{call_id}`: A unique identifier for the call session, provided by the client upon connection.

### Communication Flow

1.  **Connection:** The client initiates a WebSocket connection to the endpoint.
2.  **Greeting:** Upon successful connection, the server immediately sends an initial audio stream containing a welcome message (e.g., "Welcome to Bella Cucina...").
3.  **Client-to-Server (User Speech):**
    - The client continuously streams raw audio chunks from the user's microphone to the server.
    - **Format:** The server expects audio in a format that can be decoded to raw PCM (16kHz, 16-bit, mono). The reference frontend uses `audio/webm;codecs=opus`.
4.  **Server-to-Client (Bot Speech):**
    - After the server's Voice Activity Detection (VAD) detects that the user has finished speaking, it transcribes the audio and processes the query through the LangGraph agent.
    - The agent's text response is synthesized into audio and streamed back to the client in real-time.
    - **Format:** The server sends raw WAV audio chunks as binary messages.
5.  **Interruption:** If the user starts speaking while the bot is sending audio, the server detects this, stops the bot's audio stream, and immediately begins processing the new user input.
