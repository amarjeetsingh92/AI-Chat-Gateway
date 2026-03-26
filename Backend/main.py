"""
LLM Gateway API — FastAPI Backend
Supports: OpenAI (GPT-4o), Anthropic (Claude)
Features: SSE streaming, rate limiting, prompt routing, usage tracking
"""

import asyncio
import json
import time
from collections import defaultdict
from typing import AsyncGenerator

import anthropic
import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(title="LLM Gateway API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Clients ─────────────────────────────────────────────────────────────────

openai_client = openai.AsyncOpenAI()       # uses OPENAI_API_KEY env var
anthropic_client = anthropic.AsyncAnthropic()  # uses ANTHROPIC_API_KEY env var

# ─── Rate Limiter ────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Simple in-memory token bucket rate limiter per IP.
    In production, use Redis for distributed rate limiting.
    """
    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> tuple[bool, int]:
        now = time.time()
        window_start = now - self.window
        # Purge old timestamps
        self.requests[client_ip] = [
            ts for ts in self.requests[client_ip] if ts > window_start
        ]
        count = len(self.requests[client_ip])
        if count >= self.max_requests:
            oldest = self.requests[client_ip][0]
            retry_after = int(self.window - (now - oldest)) + 1
            return False, retry_after
        self.requests[client_ip].append(now)
        return True, 0

rate_limiter = RateLimiter(max_requests=20, window_seconds=60)

# ─── Usage Tracker ───────────────────────────────────────────────────────────

usage_store: list[dict] = []  # In production: write to DB

def track_usage(model: str, provider: str, prompt_tokens: int, completion_tokens: int):
    usage_store.append({
        "timestamp": time.time(),
        "provider": provider,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    })
    # Keep last 1000 entries in memory
    if len(usage_store) > 1000:
        usage_store.pop(0)

# ─── Models ──────────────────────────────────────────────────────────────────

AVAILABLE_MODELS = {
    "gpt-4o": {"provider": "openai", "label": "GPT-4o", "max_tokens": 4096},
    "gpt-4o-mini": {"provider": "openai", "label": "GPT-4o Mini", "max_tokens": 4096},
    "claude-opus-4-6": {"provider": "anthropic", "label": "Claude Opus 4.6", "max_tokens": 4096},
    "claude-sonnet-4-6": {"provider": "anthropic", "label": "Claude Sonnet 4.6", "max_tokens": 8096},
    "claude-haiku-4-5-20251001": {"provider": "anthropic", "label": "Claude Haiku 4.5", "max_tokens": 4096},
}

# ─── Schemas ─────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str

class ChatRequest(BaseModel):
    model: str = Field(..., description="Model ID from AVAILABLE_MODELS")
    messages: list[Message]
    system_prompt: str = "You are a helpful AI assistant."
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=4096)
    stream: bool = True

# ─── Streaming Generators ────────────────────────────────────────────────────

async def stream_openai(request: ChatRequest) -> AsyncGenerator[str, None]:
    messages = [{"role": "system", "content": request.system_prompt}]
    messages += [{"role": m.role, "content": m.content} for m in request.messages]

    prompt_tokens = 0
    completion_tokens = 0

    try:
        async with openai_client.chat.completions.stream(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ) as stream:
            async for event in stream:
                if event.type == "content.delta":
                    delta = event.delta
                    if delta:
                        payload = json.dumps({"type": "delta", "content": delta})
                        yield f"data: {payload}\n\n"

            # Final usage
            final = await stream.get_final_completion()
            if final.usage:
                prompt_tokens = final.usage.prompt_tokens
                completion_tokens = final.usage.completion_tokens

    except openai.RateLimitError:
        yield f"data: {json.dumps({'type': 'error', 'message': 'OpenAI rate limit exceeded'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    finally:
        track_usage(request.model, "openai", prompt_tokens, completion_tokens)
        yield f"data: {json.dumps({'type': 'done', 'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})}\n\n"


async def stream_anthropic(request: ChatRequest) -> AsyncGenerator[str, None]:
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt_tokens = 0
    completion_tokens = 0

    try:
        async with anthropic_client.messages.stream(
            model=request.model,
            system=request.system_prompt,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ) as stream:
            async for text in stream.text_stream:
                payload = json.dumps({"type": "delta", "content": text})
                yield f"data: {payload}\n\n"

            final = await stream.get_final_message()
            prompt_tokens = final.usage.input_tokens
            completion_tokens = final.usage.output_tokens

    except anthropic.RateLimitError:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Anthropic rate limit exceeded'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    finally:
        track_usage(request.model, "anthropic", prompt_tokens, completion_tokens)
        yield f"data: {json.dumps({'type': 'done', 'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})}\n\n"


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "models": list(AVAILABLE_MODELS.keys())}


@app.get("/models")
async def get_models():
    return {"models": AVAILABLE_MODELS}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, req: Request):
    # Rate limiting
    client_ip = req.client.host
    allowed, retry_after = rate_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    # Validate model
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")

    provider = AVAILABLE_MODELS[request.model]["provider"]

    # Route to the correct provider
    if provider == "openai":
        generator = stream_openai(request)
    elif provider == "anthropic":
        generator = stream_anthropic(request)
    else:
        raise HTTPException(status_code=500, detail="Unknown provider")

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable Nginx buffering
        },
    )


@app.get("/usage")
async def get_usage():
    """Return aggregated token usage stats."""
    if not usage_store:
        return {"total_requests": 0, "by_model": {}, "by_provider": {}, "recent": []}

    by_model: dict[str, dict] = {}
    by_provider: dict[str, dict] = {}

    for entry in usage_store:
        m = entry["model"]
        p = entry["provider"]

        if m not in by_model:
            by_model[m] = {"requests": 0, "total_tokens": 0}
        by_model[m]["requests"] += 1
        by_model[m]["total_tokens"] += entry["total_tokens"]

        if p not in by_provider:
            by_provider[p] = {"requests": 0, "total_tokens": 0}
        by_provider[p]["requests"] += 1
        by_provider[p]["total_tokens"] += entry["total_tokens"]

    return {
        "total_requests": len(usage_store),
        "by_model": by_model,
        "by_provider": by_provider,
        "recent": usage_store[-10:][::-1],
    }
