# ChattyGenieBot — Personal Assistant Improvement Plan

## What the Bot Already Does (TL;DR)

Before adding anything, it's worth being precise about what you have:

- **Telegram bot** (python-telegram-bot v21) routing messages to the **OpenAI Agents SDK**, which works against either real OpenAI or a local Ollama endpoint
- **Per-chat SQLite session memory** with auto-prune/summarization when sessions grow long
- **Long-term chat history** in a separate FTS5-backed SQLite DB — supports keyword search, recap, Q&A grounded in past turns
- **Tools** (all optional/feature-flagged): web search (Ollama), stock trends (Alpha Vantage), image analysis (Ollama vision), voice transcription (faster-whisper local)
- **Style learning** (`/learn`) — analyzes a user's messages and stores a behavior/tone profile
- **RAG infrastructure** in `app/rag/` — Ollama embeddings + numpy vector store for indexing markdown/text docs
- Commands: `/start`, `/help`, `/reset`, `/progress`, `/learn`, `/recap`

The architecture is clean and extensible. Every new capability follows the same pattern:
1. Create a tool factory in `app/tools/<name>.py`
2. Register it in `app/tools/__init__.py`  
3. Add config fields to `app/config.py`
4. Wire into `AgentRuntime.__init__()` in `agent_runtime.py`
5. Add a prompt suffix in `prompt.py`
6. Optionally add a Telegram command in `bot.py`

---

## Capability 1: Email Checking

### What to add
A read-only email tool so you can ask things like "any urgent emails?" or "summarize what I missed today."

### Recommended approach: IMAP + app password
Gmail OAuth is powerful but involves a consent screen and token refresh headaches for a personal bot. The simpler path is an **app password** (Gmail → Security → 2FA → App passwords) + standard IMAP. This works for Gmail, Outlook, iCloud, or any IMAP provider.

### Libraries
```
imapclient>=3.0    # nicer IMAP wrapper than stdlib imaplib
```
The stdlib `email` module handles message parsing — no extra dependency needed.

### New files

**`app/email_client.py`** — thin async wrapper around `imapclient`:
```python
import asyncio
import imapclient
import email
from email.header import decode_header

class IMAPEmailClient:
    def __init__(self, host, port, username, password, use_ssl=True):
        ...
    
    async def fetch_messages(self, folder="INBOX", max_count=5, unread_only=True):
        """Returns list of {uid, from, subject, date, snippet, read}"""
        return await asyncio.to_thread(self._fetch_sync, ...)
    
    async def search(self, query: str, max_count=10):
        """IMAP SEARCH by subject/from/body keywords"""
        ...
```

**`app/tools/email.py`** — two tools:
```python
@function_tool(name_override="check_email")
async def check_email(folder: str = "INBOX", max: int = 5, unread_only: bool = True):
    """Fetch recent emails. Returns sender, subject, date, and a short snippet."""

@function_tool(name_override="search_email")  
async def search_email(query: str, max: int = 10):
    """Search emails by keyword (checks subject and sender)."""
```

### Config additions (`app/config.py`)
```python
email_enabled: bool = Field(False, alias="EMAIL_ENABLED")
email_imap_host: str | None = Field(None, alias="EMAIL_IMAP_HOST")
email_imap_port: int = Field(993, alias="EMAIL_IMAP_PORT")
email_username: str | None = Field(None, alias="EMAIL_USERNAME")
email_password: str | None = Field(None, alias="EMAIL_PASSWORD")  # use app password
email_default_folder: str = Field("INBOX", alias="EMAIL_DEFAULT_FOLDER")
email_max_results: int = Field(5, alias="EMAIL_MAX_RESULTS")
```

### `.env` additions
```
EMAIL_ENABLED=true
EMAIL_IMAP_HOST=imap.gmail.com
EMAIL_USERNAME=you@gmail.com
EMAIL_PASSWORD=your-16-char-app-password
```

### Wiring into `agent_runtime.py`
Follow the exact same pattern as `finance_tool`: check `settings.email_enabled`, call `_build_email_tool()`, fall back to `create_disabled_email_tool()`.

### Prompt suffix for `prompt.py`
```
Use check_email to fetch recent messages when the user asks about their inbox.
Never send email without explicit confirmation. Summarize message subjects and senders concisely.
```

### One gotcha
IMAP connections have timeouts and state — wrap all calls in `asyncio.to_thread()` to keep the bot non-blocking, and reconnect on idle timeout. `imapclient` handles this reasonably if you call `client.noop()` periodically or just reconnect per-call for a personal bot with low traffic.

---

## Capability 2: Notes / Personal Knowledge Base

### What to add
The ability to save arbitrary notes and recall them later — "remember that the meeting code is 4829", "what did I save about the vacation plan?", "show all notes tagged work."

### Recommended approach: SQLite table + FTS5 (reuse existing infrastructure)
You already have a well-structured SQLite DB with FTS5 triggers. Just add a `notes` table to the same DB and a `NoteStore` class that mirrors `ChatStore`'s pattern.

### New file: `app/storage/note_store.py`
```python
class NoteStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._ensure_schema()
    
    def save_note(self, *, chat_id: int, title: str, content: str, tags: list[str] = None) -> int:
        """Insert or update a note. Returns note id."""
    
    def search_notes(self, chat_id: int, query: str, limit: int = 10) -> list[dict]:
        """FTS5 keyword search across title + content."""
    
    def list_notes(self, chat_id: int, tag: str = None, limit: int = 20) -> list[dict]:
        """List recent notes, optionally filtered by tag."""
    
    def get_note(self, chat_id: int, note_id: int) -> dict | None: ...
    def delete_note(self, chat_id: int, note_id: int) -> bool: ...
```

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT,          -- JSON array of strings
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    title, content,
    content='notes', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
-- plus INSERT/DELETE/UPDATE triggers like messages_fts
```

Initialize `NoteStore` in `AgentRuntime.__init__()` by passing `self._chat_store.get_connection()` — it reuses the same SQLite connection already opened.

### New file: `app/tools/notes.py`
```python
@function_tool(name_override="save_note")
async def save_note(title: str, content: str, tags: list[str] | None = None) -> dict:
    """Save a note to the personal knowledge base."""

@function_tool(name_override="search_notes")
async def search_notes(query: str, limit: int = 5) -> dict:
    """Search saved notes by keyword."""

@function_tool(name_override="list_notes")
async def list_notes(tag: str | None = None) -> dict:
    """List recent notes, optionally filtered by tag."""
```

The tricky part with Agents SDK tools is that they're stateless functions, so you need to inject the `NoteStore` via closure (same pattern `create_ollama_web_search_tool` uses with `WebSearchClient`). Make a `create_notes_tools(note_store, chat_id_resolver)` factory.

### Telegram command
Add `/note <text>` as a quick shortcut that saves a note without needing to converse with the agent:
```python
@require_authorized
async def note_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = " ".join(context.args or [])
    # save directly, reply with confirmation + id
```
Add `/notes` to list recent notes inline.

### Semantic search later (optional upgrade)
The existing `app/rag/` infrastructure already does Ollama embeddings → numpy arrays. You could extend it to embed notes at save-time and do cosine similarity search, giving you "find notes about X" even when keywords don't match. Start with FTS5 (already built) and add this later.

---

## Capability 3: Todo Tracking

### What to add
A simple task list you can manage conversationally: "add a todo to call the dentist tomorrow", "what's on my list?", "mark the dentist call done."

### Recommended approach: SQLite table (same DB, same pattern as notes)

### New file: `app/storage/todo_store.py`
```python
class TodoStore:
    def add_todo(self, *, chat_id: int, title: str, due: str = None, priority: int = 0, tags: list[str] = None) -> int: ...
    def list_todos(self, chat_id: int, status: str = "pending", limit: int = 20) -> list[dict]: ...
    def complete_todo(self, chat_id: int, todo_id: int) -> bool: ...
    def update_todo(self, chat_id: int, todo_id: int, **fields) -> bool: ...
    def delete_todo(self, chat_id: int, todo_id: int) -> bool: ...
    def search_todos(self, chat_id: int, query: str) -> list[dict]: ...
```

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | done | cancelled
    priority INTEGER NOT NULL DEFAULT 0,       -- 0=normal, 1=high, 2=urgent
    due_date TEXT,                             -- ISO date string, nullable
    tags TEXT,                                 -- JSON array
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS todos_chat_status ON todos(chat_id, status);
```

### New file: `app/tools/todos.py`
```python
@function_tool(name_override="add_todo")
async def add_todo(title: str, due: str | None = None, priority: str = "normal") -> dict:
    """Add a task to the todo list. due can be 'tomorrow', '2025-05-01', etc."""

@function_tool(name_override="list_todos")
async def list_todos(status: str = "pending") -> dict:
    """List todos. status can be 'pending', 'done', or 'all'."""

@function_tool(name_override="complete_todo")
async def complete_todo(todo_id: int) -> dict:
    """Mark a todo as done by its ID."""

@function_tool(name_override="update_todo")
async def update_todo(todo_id: int, title: str | None = None, due: str | None = None, priority: str | None = None) -> dict:
    """Update a todo's title, due date, or priority."""
```

### Telegram command
`/todos` — shows pending tasks formatted as a list. No bot conversation needed for a quick glance.

### Natural language due dates
Add `python-dateparser` to requirements.txt:
```
python-dateparser>=1.2
```
Parse "next Monday", "tomorrow", "end of week" → ISO date before storing. The agent can pass natural strings and the tool resolves them.

---

## Capability 4: Reminders

### What to add
"Remind me in 2 hours to check the oven" → bot sends you a Telegram message at the right time.

### Recommended approach: python-telegram-bot's built-in JobQueue
`python-telegram-bot` ships with `JobQueue` (backed by `APScheduler`). It's already wired into the `Application` object — you just need to use it. No additional library needed.

### New SQLite table: `reminders`
Persist reminders so they survive bot restarts:
```sql
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    message TEXT NOT NULL,
    fire_at TEXT NOT NULL,     -- ISO datetime UTC
    fired INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
```

On bot startup, reload unfired reminders from DB and re-schedule them into `job_queue`.

### New tool: `app/tools/reminders.py`
```python
@function_tool(name_override="set_reminder")
async def set_reminder(message: str, when: str) -> dict:
    """Schedule a reminder. 'when' can be a duration ('2 hours', '30 minutes')
    or a time ('tomorrow at 9am', '2025-05-01 14:00')."""
```

The tool stores in DB and calls `context.job_queue.run_once(callback, delay, data={...})`. The callback just sends a Telegram message to the chat.

### Prompt suffix
```
Use set_reminder when the user asks to be reminded of something.
Parse relative times like '2 hours' or 'tomorrow at 9am' and confirm the scheduled time back to the user.
```

### Tricky part: passing `job_queue` to the tool
The tool needs access to the Telegram `Application`'s `job_queue`, which isn't available in `AgentRuntime`. Two options:
- Store the `job_queue` reference in `AgentRuntime` (pass it in `build_application`)
- Make the reminder tool a lightweight stub that returns a "scheduled" payload, and handle the actual scheduling in the Telegram message handler after the agent returns

Option 2 is simpler and keeps `AgentRuntime` decoupled from Telegram.

---

## Capability 5: Fetch a URL / Web Scraping

### What to add
"Summarize this article: [URL]" — the agent fetches the page and summarizes it, without needing a full search.

### Libraries
`httpx` is already a dependency. Just add:
```
beautifulsoup4>=4.12
```

### New tool: `app/tools/fetch_url.py`
```python
@function_tool(name_override="fetch_url")
async def fetch_url(url: str) -> dict:
    """Fetch a URL and return its main text content (stripped of HTML)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    # remove nav, script, style, footer
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return {"url": url, "content": text[:8000]}  # cap at 8k chars for context window
```

Add a `FETCH_URL_ENABLED` config flag. Low risk, high utility.

---

## Capability 6: Smarter Agent Memory (Quick Win)

### Problem
Right now the agent's only memory of past context is the active session (auto-pruned with a summary) plus the FTS chat history. But the agent can't *proactively* search its own history during a turn — it just sees whatever is in the current session window.

### What to add
Give the agent a `search_history` tool so it can look up past conversations on demand:

```python
@function_tool(name_override="search_memory")
async def search_memory(query: str, limit: int = 5) -> dict:
    """Search past conversations for relevant context."""
```

This is essentially wrapping the existing `ChatStore.search_messages()` method as an agent tool. It's maybe 20 lines of code and immediately makes the bot much more useful as a personal assistant ("what did we discuss about X last month?").

---

## Implementation Roadmap

Here's a practical order — each step is independent and shippable:

**Week 1 — Todos & Notes** (pure SQLite, no external deps)
- Add `TodoStore` + `NoteStore` in `app/storage/`
- Add `todos.py` + `notes.py` in `app/tools/`
- Wire into `AgentRuntime`, add `/todo` and `/note` commands to `bot.py`
- Add `search_memory` tool (it's just a wrapper — 20 lines)

**Week 2 — Reminders** (uses existing JobQueue)
- Add reminders schema + persistence
- Add `set_reminder` tool
- Handle restart recovery in `build_application()`
- Add `python-dateparser` to requirements.txt

**Week 3 — URL Fetching** (httpx already installed)
- Add `fetch_url` tool
- Add `beautifulsoup4` to requirements.txt
- Test with a few article URLs

**Week 4 — Email** (external dependency, needs app password setup)
- Add `imapclient` to requirements.txt
- Write `IMAPEmailClient` + email tools
- Test with Gmail app password
- Add config + docs

---

## requirements.txt additions (all four capabilities)

```
# existing
httpx>=0.27,<1
openai-agents>=0.3.3,<0.4
pydantic>=2.6,<3
pydantic-settings>=2.1,<3
python-telegram-bot>=21,<22
Pillow>=10,<11
faster-whisper>=1.0,<2

# new
imapclient>=3.0,<4          # email (IMAP)
beautifulsoup4>=4.12,<5     # URL fetch
python-dateparser>=1.2,<2   # natural language due dates / reminder times
```

---

## Notes on Architecture Fit

A few things worth keeping in mind as you build:

**Tool injection pattern** — all existing tools use factory functions (e.g., `create_ollama_web_search_tool(client, ...)`) that close over dependencies. Follow the same pattern for notes/todos/email so the stateless tool functions can reach your store objects.

**`chat_id` scoping** — notes and todos should be scoped to `chat_id` so they're isolated per conversation. The `AgentRuntime` already passes `chat_id` into everything; the tool factories can capture it via a resolver callable, or you can store the current `chat_id` in a `contextvars.ContextVar` set at the start of each `_run()` call.

**The RAG module is underused** — `app/rag/indexer.py` builds numpy embeddings for docs in a directory. Once you have notes stored in SQLite, you could auto-index them into the RAG store for semantic search. This gives you "find notes about my Japan trip" even if you didn't tag them well. It's an optional upgrade but the plumbing is already there.

**Don't add a new DB** — everything new should go into the existing `chat_history.db` via shared tables. One SQLite file, one connection, one WAL journal. The `ChatStore.get_connection()` method already exposes the connection for exactly this reason (see `StyleProfileStore`).
