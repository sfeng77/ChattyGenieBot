# Chatty Genie Bot

Telegram assistant backed by a local Ollama model with retrieval-augmented generation over documents on disk.

## Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) running locally
- Telegram bot token from @BotFather

Pull the models you plan to use:
```powershell
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (or export env vars):
```text
TELEGRAM_BOT_TOKEN=123456:ABCDEF
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1
EMBED_MODEL=nomic-embed-text
```

Place reference documents under `data/docs`. Supported formats: `.txt`, `.md`, `.markdown`, `.rst`.

Build the retrieval index:
```powershell
python -m app.rag.indexer
```

## Run the bot
```powershell
python main.py
```

The bot runs in long-polling mode. Use `/reset` to clear chat memory and `/reindex` to rebuild the index (owner only).

## Notes
- Responses stream into Telegram via message edits.
- The RAG context is derived from cosine similarity over Ollama embeddings.
- Guardrails: per-message length cap, retrieval failure handling, configurable reindex permissions.
