import os
import json
import sys
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Try to load .env locally; on Render, env vars are set directly
try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
except ImportError:
    pass  # python-dotenv not available on Render (we use env vars directly)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

if not OPENROUTER_API_KEY:
    print("❌ OPENROUTER_API_KEY is not set. Set it as an environment variable or in .env")
    sys.exit(1)

print("✅ OpenRouter API key loaded successfully.")

# Model configurations with fallbacks
PRIMARY_MODEL = "openai/gpt-4o"
REASONING_MODEL = "anthropic/claude-3.5-sonnet"
PRIMARY_MODEL_FALLBACKS = ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"]
REASONING_MODEL_FALLBACKS = ["openai/gpt-4o", "openai/gpt-4o-mini"]
FREE_MODEL_FALLBACKS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-2b-it:free",
]


def _candidate_models(base_model: str) -> list[str]:
    """Build candidate model list with fallbacks."""
    candidates = [base_model]
    if base_model == PRIMARY_MODEL:
        for fb in PRIMARY_MODEL_FALLBACKS:
            if fb not in candidates:
                candidates.append(fb)
    elif base_model == REASONING_MODEL:
        for fb in REASONING_MODEL_FALLBACKS:
            if fb not in candidates:
                candidates.append(fb)
    for fb in FREE_MODEL_FALLBACKS:
        if fb not in candidates:
            candidates.append(fb)
    return candidates


class PipelineAPIHandler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_json(self, status_code, data):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                with open("dashboard.html", "rb") as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self._send_json(500, {"error": "dashboard.html not found"})
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        if parsed.path == "/api/llm":
            return self._handle_llm_call(data)

        self._send_json(404, {"error": "Not found"})

    def _handle_llm_call(self, data):
        prompt = data.get("prompt", "")
        model = data.get("model", PRIMARY_MODEL)

        if not prompt:
            self._send_json(400, {"error": "Missing 'prompt' field"})
            return

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Vyshnavi-279/Promt_Pipeline",
            "X-Title": "Agentic AI Pipeline Server",
        }

        candidates = _candidate_models(model)
        last_error = None

        for candidate_model in candidates:
            payload = {
                "model": candidate_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 200,
            }

            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=90,
                )
                if response.status_code in {404, 429, 402}:
                    last_error = f"OpenRouter API error [{response.status_code}]: {response.text}"
                    continue

                if not response.ok:
                    err_text = response.text
                    self._send_json(response.status_code, {
                        "error": f"OpenRouter API error [{response.status_code}]: {err_text}"
                    })
                    return

                payload_resp = response.json()
                content = payload_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                self._send_json(200, {"content": content.strip()})
                return

            except requests.RequestException as exc:
                last_error = f"Request failed: {str(exc)}"
                continue
            except (KeyError, IndexError, ValueError) as exc:
                last_error = f"Response parsing error: {str(exc)}"
                continue

        self._send_json(503, {
            "error": f"All model candidates exhausted. Last error: {last_error}"
        })


if __name__ == "__main__":
    PORT = int(os.getenv("PORT", "8765"))
    server = HTTPServer(("0.0.0.0", PORT), PipelineAPIHandler)
    print(f"\n🚀 Pipeline Server running on port {PORT}")
    print(f"   Dashboard: http://localhost:{PORT}" if os.getenv("PORT") is None else f"   Dashboard: http://0.0.0.0:{PORT}")
    print("   Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")
        server.server_close()