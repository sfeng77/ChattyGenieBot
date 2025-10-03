# Chatty Genie Bot

Telegram assistant built on the OpenAI Agents SDK. The bot simply transports messages between Telegram and an agent that holds per-chat memory via SQLite sessions. Optional web search hooks into the Ollama Web Search API so the agent can cite fresh results.

## Prerequisites
- Python 3.11+
- An OpenAI API key with access to the configured model (or Ollama running with the OpenAI-compatible server)
- Telegram bot token from @BotFather
- Ollama API key (for the web search endpoint)

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration
Create a `.env` file (or export environment variables):
```text
TELEGRAM_BOT_TOKEN=123456:ABCDEF
OPENAI_API_KEY=ollama
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_MODEL=gpt-oss:20b
OPENAI_TEMPERATURE=0.2
OPENAI_TRACING_ENABLED=false
WEB_SEARCH_ENABLED=false
WEB_SEARCH_BASE_URL=https://ollama.com
WEB_SEARCH_ENDPOINT=/api/web_search
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_TIMEOUT=15
OLLAMA_API_KEY=your-ollama-api-key
LOG_LEVEL=DEBUG
MAX_INPUT_CHARS=4000
```
`OPENAI_API_BASE` defaults to the local Ollama OpenAI-compatible endpoint. Set `WEB_SEARCH_ENABLED=true` to expose the `web_search` tool; the Ollama API key is required and is sent as `Authorization: Bearer <OLLAMA_API_KEY>`.
`SESSIONS_DB_PATH` may be set if you need a custom storage location. By default, session state lives at `data/sessions/sessions.db`.

## Run the bot
```powershell
python main.py
```

Commands:
- `/start` - welcome message
- `/help` - command reference
- `/reset` - clear the conversation memory for the current chat

## Notes
- Conversation memory is managed by OpenAI Agents sessions; we keep one SQLite session per chat ID.
- When the web search tool is enabled, the agent will call it for time-sensitive questions and respond with short citations.
- Refer to the OpenAI Agents Python quickstart (https://openai.github.io/openai-agents-python/quickstart/) for extending the agent with tools or different models.
