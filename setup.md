# External Setup and Model Management Guide

This document explains the external dependencies, API keys, and automatic model management handled by the Bella Cucina Voice Bot application.

## 1. API Keys

### Google Gemini API Key

- **Requirement:** This is a **mandatory** key for the AI's core reasoning and conversational abilities.
- **Setup:**
    1.  Obtain an API key from [Google AI Studio](https://makersuite.google.com/app/apikey).
    2.  Create a file named `.env` in the `backend/` directory.
    3.  Add the key to the file in this format: `GEMINI_API_KEY="your_api_key_here"`

## 2. Automatic Model Caching

The application is designed to be as self-contained as possible. The AI models required for Speech-to-Text (STT), Text-to-Speech (TTS), and Retrieval-Augmented Generation (RAG) are downloaded and cached automatically on the first run. **You do not need to manually download these models.**

### Speech-to-Text (STT)

- **Library:** `faster-whisper`
- **Process:** The first time the server starts, this library will download the Whisper model specified in `config.py` (default: `"base"` model).
- **Cache Location:** The model is cached in the user's home directory under a path similar to `~/.cache/huggingface/hub/models--Systran--faster-whisper-...`.

### Text-to-Speech (TTS)

- **Library:** `TTS` (from Coqui)
- **Process:** On first server startup, the TTS library will download the voice model specified in `config.py` (default: `"tts_models/en/ljspeech/tacotron2-DDC"`).
- **Cache Location:** The model is cached in the user's home directory under `~/.local/share/tts/`.

### RAG & Embeddings

- **Library:** `sentence-transformers`
- **Process:** The first time the RAG service is initialized, this library will download the embedding model specified in `config.py` (default: `"all-MiniLM-L6-v2"`).
- **Cache Location:** The model is cached in the user's home directory under `~/.cache/torch/sentence_transformers/`.

**Note:** The initial server startup will be longer to account for these downloads. Subsequent startups will be much faster as the models will be loaded from the local cache.

## 3. Automatic Vector Database Generation

### FAISS Index

- **Library:** `faiss-cpu`
- **Process:** The application uses a FAISS vector store for its RAG capabilities.
    1.  On the first server startup, the `RAGService` checks for the existence of an index file in the `backend/data/` directory.
    2.  If no index is found, it will automatically:
        -   Read the `restaurant_kb.json` file.
        -   Generate embeddings for all the menu items and FAQs.
        -   Build a FAISS index from these embeddings.
        -   Save the index and its corresponding documents to `backend/data/faiss_index.index` and `backend/data/faiss_index.pkl`.
    3.  On subsequent startups, the service will load the pre-built index directly from these files, making initialization much faster.

No manual steps are required for this process. To rebuild the index, simply delete the `faiss_index.index` and `faiss_index.pkl` files from the `backend/data/` directory and restart the server.
