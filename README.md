# AI-Chat-Gateway

link :- https://llm-gateway-backend-du4j.onrender.com/

A full-stack chat gateway that routes prompts to OpenAI and Anthropic models through a single API, with real-time streaming and usage analytics.

![Status](https://img.shields.io/badge/status-active-brightgreen) ![Python](https://img.shields.io/badge/python-3.10+-blue) ![React](https://img.shields.io/badge/react-18+-61dafb)

## Features

- **Multi-provider routing** — single endpoint, dispatches to OpenAI or Anthropic based on the selected model
- **Live token streaming** — Server-Sent Events (SSE) stream tokens to the UI as they're generated
- **Rate limiting** — in-memory token bucket limiter (20 req/min per IP, returns `429` with `Retry-After`)
- **Usage tracking** — prompt/completion token counts logged per request, aggregated via `/usage`
- **Configurable per session** — system prompt, temperature, and max tokens adjustable from the UI
- **Stop generation** — abort an in-flight stream mid-response

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | FastAPI, Python, OpenAI SDK, Anthropic SDK |
| Frontend | React (hooks), Fetch + `ReadableStream` for SSE |
| Transport | Server-Sent Events |

## Supported Models

| Model | Provider |
|---|---|
| GPT-4o | OpenAI |
| GPT-4o Mini | OpenAI |
| Claude Opus 4.6 | Anthropic |
| Claude Sonnet 4.6 | Anthropic |
| Claude Haiku 4.5 | Anthropic |

## Project Structure

```
.
├── backend/
│   └── main.py            # FastAPI app: routing, rate limiting, streaming, usage
├── frontend/
│   └── LLMDashboard.jsx   # React dashboard: chat UI, SSE hook, settings, analytics
└── README.md
```

## Setup

### Backend

```bash
cd backend
pip install fastapi uvicorn openai anthropic pydantic

export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here

uvicorn main:app --reload
```

API runs at `http://localhost:8000`.

### Frontend

Drop `LLMDashboard.jsx` into a React app (Vite/CRA) and point `API_BASE` in the file to your backend URL. Then:

```bash
npm install
npm run dev
```

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check + list of available models |
| `GET` | `/models` | Full model metadata (provider, label, token limits) |
| `POST` | `/chat/stream` | Stream a chat completion via SSE |
| `GET` | `/usage` | Aggregated token usage by model and provider |

### `POST /chat/stream`

```json
{
  "model": "claude-sonnet-4-6",
  "messages": [{ "role": "user", "content": "Hello" }],
  "system_prompt": "You are a helpful AI assistant.",
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": true
}
```

Streams SSE events of three types: `delta` (token chunk), `error`, and `done` (final token counts).

## Notes

- Rate limiter and usage store are in-memory — fine for local dev, swap for Redis/a DB in production
- CORS is currently scoped to `localhost:5173` and `localhost:3000` — update `allow_origins` in `main.py` for other origins

## License

MIT
