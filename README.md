# 🔎 Tool-Using Research Agent

A conversational research agent built with **LangGraph** that answers questions by
calling tools — web search (Tavily) and live weather (Open-Meteo) — and cites its
sources. Includes a **Streamlit** chat UI with login/signup, persistent per-user
conversation history, and streaming responses.

## Features

- **Tool-using agent** — routes each question to the right tool (or no tool at all)
  using an LLM (via Groq) with LangGraph's `ToolNode` / `tools_condition`
- **Web search** — Tavily search tool for general research questions
- **Weather lookup** — custom tool using Open-Meteo (geocoding + forecast), no API key required
- **Source citations** — system prompt instructs the model to cite sources inline
  and list URLs at the end of its answer
- **Persistent memory** — conversation state saved to sqlite (`chat_history.db`)
  via LangGraph's `SqliteSaver` checkpointer, so chats survive app restarts
- **Multi-user auth** — signup/login with bcrypt-hashed passwords and JWT session
  tokens; each user only sees their own saved conversations
- **Resume past conversations** — sidebar shows previous threads labeled with a
  short title derived from the first message, not a raw ID
- **Streaming responses** — assistant replies stream token-by-token in the UI

## Project structure

```
.
├── agent.py          # LangGraph graph: state, tools, model, nodes, checkpointer
├── app.py             # Streamlit UI: auth gate, chat interface, streaming
├── auth.py            # User signup/login, password hashing, JWT issue/verify
├── requirements.txt   # Python dependencies
├── .env.example        # Template for required environment variables
└── chat_history.db    # sqlite DB (created automatically on first run)
```

## Setup

1. **Clone / copy the project**, then create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables** — copy the example file and fill in your keys:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env`:

   ```
   GROQ_API_KEY=your-groq-key-here
   TAVILY_API_KEY=tvly-your-key-here
   JWT_SECRET=replace-this-with-a-long-random-string
   ```

   - Get a Groq API key at [console.groq.com](https://console.groq.com)
   - Get a Tavily API key at [tavily.com](https://tavily.com) (free tier: 1,000 searches/month)
   - `JWT_SECRET` can be any long random string — used to sign session tokens.
     Generate one quickly with:
     ```bash
     python -c "import secrets; print(secrets.token_hex(32))"
     ```

4. **Run the app:**

   ```bash
   streamlit run app.py
   ```

   Open the URL Streamlit prints (usually `http://localhost:8501`).

## Usage

1. On first launch, go to the **Sign up** tab and create an account.
2. Switch to **Log in** and sign in.
3. Ask a research question — e.g.:
   - *"What are the latest AI regulations announced in the EU?"* → uses web search, cites sources
   - *"What's the weather in Ahmedabad right now?"* → uses the weather tool
   - *"What is the capital of France?"* → answered directly, no tool needed
4. Use the sidebar to start a new conversation or resume a past one (shown by a
   short auto-generated title, not a raw thread ID).

## How it works (high level)

- **Graph**: `chat_node` (the LLM, bound to both tools) decides whether to answer
  directly or call a tool. `tools_condition` routes to the `tools` node if a tool
  call was requested; the tool result is fed back into `chat_node`, which then
  produces the final answer. This loop continues until the model responds without
  requesting another tool call.
- **Memory**: every graph run is checkpointed to sqlite under a `thread_id`. Each
  new conversation gets a `thread_id` of the form `"{username}::{uuid}"`, which is
  how per-user thread isolation is enforced in the sidebar.
- **Citations**: a system prompt is prepended to every LLM call instructing it to
  cite sources with `[1]`, `[2]`-style references and a `Sources:` list when it
  uses the search tool.
- **Auth**: passwords are hashed with bcrypt before being stored in a `users`
  table in the same sqlite file. On login, a JWT (24-hour expiry) is generated and
  kept in Streamlit's session state to represent the logged-in session.

## Known limitations

- Login state lives only in Streamlit's `session_state`, so refreshing the browser
  tab will log you out (no persistent browser cookie yet).
- Thread titles are the first user message, truncated — not an LLM-generated summary.
- Groq's `openai/gpt-oss-20b` model is used by default; swap the model name in
  `agent.py` if you'd like to try a different one.

## Tech stack

`LangGraph` · `LangChain` · `Groq` · `Tavily` · `Open-Meteo` · `Streamlit` · `SQLite` · `PyJWT` · `bcrypt`