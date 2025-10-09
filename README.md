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
PROGRESS_EDIT_THROTTLE_MS=800
PROGRESS_KEEP_TIMELINE=false
PROGRESS_TOOL_RESULT_MAX_CHARS=160
WHITELISTED_USER_IDS=123456789,987654321
HISTORY_PRUNE_ENABLED=true
HISTORY_PRUNE_THRESHOLD_ITEMS=60
HISTORY_KEEP_LAST_ITEMS=12
HISTORY_SUMMARY_MAX_CHARS=800
FINANCE_ENABLED=false
FINANCE_PROVIDER=alpha_vantage
FINANCE_API_KEY=
FINANCE_TIMEOUT=10
FINANCE_WINDOW_DAYS=7
FINANCE_CACHE_TTL_MINUTES=10
VISION_ENABLED=false
VISION_MODEL=gemma3:12b
VISION_TEMPERATURE=0.2
VISION_TIMEOUT=20
VISION_MAX_EDGE=1280
VISION_SYSTEM_PROMPT=You are a concise vision assistant. Focus on factual observations and answer user questions about the image.
```

`OPENAI_API_BASE` defaults to the local Ollama OpenAI-compatible endpoint. Set `WEB_SEARCH_ENABLED=true` to expose the `web_search` tool; the Ollama API key is required and is sent as `Authorization: Bearer <OLLAMA_API_KEY>`.
`SESSIONS_DB_PATH` may be set if you need a custom storage location. By default, session state lives at `data/sessions/sessions.db`.
Enable `FINANCE_ENABLED=true` with an Alpha Vantage API key to expose the `stock_trend` tool for 7-day price trends.
Enable `VISION_ENABLED=true` to let the agent analyze Telegram photos with the configured vision model.
Use `PROGRESS_EDIT_THROTTLE_MS`, `PROGRESS_KEEP_TIMELINE`, and `PROGRESS_TOOL_RESULT_MAX_CHARS` to tune how often Telegram messages are updated and how much tool output is surfaced. Define `WHITELISTED_USER_IDS` as a comma-separated list of Telegram user IDs to restrict access (leave empty to allow everyone). Use `HISTORY_PRUNE_ENABLED`, `HISTORY_PRUNE_THRESHOLD_ITEMS`, `HISTORY_KEEP_LAST_ITEMS`, and `HISTORY_SUMMARY_MAX_CHARS` to control automatic summarization of long conversations.

## Run the bot
```powershell
python main.py
```

Commands:
- `/start` - welcome message
- `/help` - command reference
- `/reset` - clear the conversation memory for the current chat
- `/progress` - toggle live progress updates

## Notes
- Conversation memory is managed by OpenAI Agents sessions; we keep one SQLite session per chat ID.
- Use /progress to toggle live tool/model status updates while a turn is running.
- When the web search tool is enabled, the agent will call it for time-sensitive questions and respond with short citations.
- Refer to the OpenAI Agents Python quickstart (https://openai.github.io/openai-agents-python/quickstart/) for extending the agent with tools or different models.
