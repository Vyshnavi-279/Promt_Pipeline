# Prompt Pipeline — Agentic Multi-Stage AI Customer Support

A 3-stage agentic AI pipeline that extracts, reasons, and responds to customer support tickets using OpenRouter LLMs.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    browser (dashboard.html)               │
│  POST /api/llm  ───────────┐  ┌─────────────────────┐   │
│                            │  │ Stage 1: Extract    │   │
│                            │  │ Stage 2: Reason     │   │
│                            │  │ Stage 3: Generate   │   │
│  http://localhost:8765 ◄───┘  └─────────────────────┘   │
└──────────────────────┬──────────────────────────────────┘
                       │
            ┌──────────▼──────────┐
            │    server.py         │  ← reads API key from .env
            │  (Python backend)    │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │   OpenRouter API     │
            │   (LLM models)       │
            └─────────────────────┘
```

### Model Fallback Chain

`server.py` automatically falls back through models if the primary is unavailable:

1. **Primary models:** `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`
2. **Fallback paid:** `openai/gpt-4o-mini`, `anthropic/claude-3-haiku`
3. **Free tier:** `meta-llama/llama-3.2-3b-instruct:free`, `microsoft/phi-3-medium-128k-instruct:free`, `mistralai/mistral-7b-instruct:free`, `google/gemma-2-2b-it:free`

## Quick Start

### 1. Prerequisites

- Python 3.8+
- OpenRouter API key ([get one free](https://openrouter.ai/settings/keys))

### 2. Setup

```bash
# Install dependencies
pip install requests python-dotenv

# Configure your API key
echo "OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx" > .env
```

### 3. Run

```bash
python3 server.py
```

The dashboard opens automatically at **http://localhost:8765**. No browser setup needed — the API key is loaded from `.env` on the server side.

### 4. Use

1. Select a preset scenario or type your own support ticket text
2. Click **"Process & Generate Answer"**
3. The 3-stage pipeline runs:
   - **Step 1** → Extracts customer name, order ID, issue, wait time, sentiment
   - **Step 2** → Reason about priority and routing (urgency + team assignment)
   - **Step 3** → Generates a warm email response

## Files

| File | Purpose |
|------|---------|
| `.env` | OpenRouter API key (never committed to git) |
| `server.py` | Backend server — serves dashboard, proxies LLM calls with model fallback |
| `dashboard.html` | Frontend UI with 3-stage pipeline visualization |
| `pipeline.py` | Standalone CLI pipeline (same 3-stage logic, no dashboard) |

## Troubleshooting

- **"Address already in use"** — kill the old server: `lsof -ti:8765 | xargs kill -9`
- **All models failing** — check your OpenRouter key has credits or switch to free models
- **Rate limited (429)** — the fallback chain automatically tries other models; retry after a few seconds