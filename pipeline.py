import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ---------------------------------------------------------
# GLOBAL CONFIGURATION & API SETUP - OPENROUTER FREE TIER
# ---------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# Preferred free-tier model slugs requested for this assignment.
PREFERRED_PRIMARY_MODEL = "mistralai/mistral-7b-instruct:free"
PREFERRED_REASONING_MODEL = "microsoft/phi-3-medium-128k-instruct:free"

# Active fallback models that have been verified to respond successfully on OpenRouter.
PRIMARY_MODEL = PREFERRED_PRIMARY_MODEL
REASONING_MODEL = PREFERRED_REASONING_MODEL
PRIMARY_MODEL_FALLBACKS = ["qwen/qwen-2.5-7b-instruct", "google/gemma-4-26b-a4b-it:free"]
REASONING_MODEL_FALLBACKS = ["google/gemma-4-26b-a4b-it:free", "qwen/qwen-2.5-7b-instruct"]


def _candidate_models(base_model: str, fallback_models: list[str]) -> list[str]:
    candidates = [base_model]
    for fallback in fallback_models:
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def call_llm(prompt: str, model_name: str) -> str:
    if not OPENROUTER_API_KEY:
        return "API_ERROR: OPENROUTER_API_KEY is not set. Add it to your shell environment or .env file."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Vyshnavi-279/Promt_Pipeline",
        "X-Title": "Agentic AI Homework Pipeline",
    }
    fallback_models = PRIMARY_MODEL_FALLBACKS if model_name == PRIMARY_MODEL else REASONING_MODEL_FALLBACKS
    for candidate_model in _candidate_models(model_name, fallback_models):
        data = {
            "model": candidate_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 400,
        }
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=90,
            )
            if response.status_code in {404, 429, 402}:
                continue
            response.raise_for_status()
            payload = response.json()
            if payload.get("choices"):
                content = payload["choices"][0]["message"].get("content", "")
                if isinstance(content, str):
                    return content.strip()
            if isinstance(payload, dict) and "error" in payload:
                return f"API_ERROR: {payload['error'].get('message', payload['error'])}"
            return f"API_ERROR: unexpected response payload: {payload}"
        except requests.RequestException as exc:
            return f"API_ERROR: {str(exc)}"
        except ValueError as exc:
            return f"API_ERROR: invalid JSON response: {str(exc)}"

    return f"API_ERROR: no working model could satisfy the request for {model_name}."


def _extract_json_object(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"<pad>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\|[^|]+\|>", "", cleaned)
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    return cleaned.strip()


def _fallback_payload(prompt_context: str) -> dict:
    if "Stage 1" in prompt_context:
        return {
            "customer_name": "UNKNOWN",
            "order_id": "UNKNOWN",
            "core_issue": "UNKNOWN",
            "days_waiting": "UNKNOWN",
            "sentiment": "neutral",
        }
    if "Stage 2" in prompt_context:
        return {
            "reasoning_steps": ["Fallback reasoning due to model parsing failure."],
            "priority": "P2 (Standard Business Priority)",
            "route": "customer_relations",
        }
    return {"error": "Parser Exhausted", "raw_output": "fallback payload"}


def safe_parse_json(llm_output: str, prompt_context: str, model_name: str, current_attempt=1, max_attempts=3) -> dict:
    if "API_ERROR:" in llm_output:
        return {"error": "Upstream API failure", "details": llm_output}

    cleaned = _extract_json_object(llm_output)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            if "Stage 2" in prompt_context:
                normalized = {
                    "reasoning_steps": parsed.get("reasoning_steps") if isinstance(parsed.get("reasoning_steps"), list) else ["Fallback reasoning due to model parsing failure."],
                    "priority": parsed.get("priority") if isinstance(parsed.get("priority"), str) else "P2 (Standard Business Priority)",
                    "route": parsed.get("route") if isinstance(parsed.get("route"), str) else "customer_relations",
                }
                return normalized
            return parsed
        return {"error": "Unexpected JSON shape", "raw_output": llm_output}
    except json.JSONDecodeError as exc:
        if current_attempt >= max_attempts:
            return _fallback_payload(prompt_context)

        print(f"⚠️  [Self-Healing Loop] Malformed JSON caught on attempt {current_attempt}. Re-prompting...")
        healing_prompt = f"""
You are an automated format reconciliation sub-agent.
Your previous output failed strict JSON parsing.
Parsing Error Encountered: {exc}
Your Raw Output Was: {llm_output}
Task: Re-format the response into valid, clean, parseable JSON with exactly these keys: reasoning_steps, priority, route.
Return only pure JSON with no markdown wrappers.
"""
        corrected_output = call_llm(healing_prompt, model_name)
        return safe_parse_json(corrected_output, prompt_context, model_name, current_attempt + 1, max_attempts)


# STAGE 1: UNDERSTAND
PROMPT_STAGE_1 = """
You are a precision data-extraction system. Analyze the customer support ticket text.
Extract core entities into the exact JSON schema below. If missing, use "UNKNOWN".
Return only pure JSON with no commentary.

Input: "{text}"

Required JSON schema:
{{
    "customer_name": "string or UNKNOWN",
    "order_id": "string or UNKNOWN",
    "core_issue": "string description",
    "days_waiting": "integer or UNKNOWN",
    "sentiment": "highly frustrated | neutral | satisfied"
}}
"""


def stage1_understand(raw_ticket: str) -> dict:
    prompt = PROMPT_STAGE_1.format(text=raw_ticket)
    response = call_llm(prompt, PRIMARY_MODEL)
    return safe_parse_json(response, "Stage 1 Extractor Schema", PRIMARY_MODEL)


# STAGE 2: REASON
PROMPT_STAGE_2 = """
You are an advanced operations dispatch strategist. Evaluate this tracking record.
Produce exactly one JSON object with only these keys: reasoning_steps, priority, route.
Each reasoning_steps item must be a short plain-English sentence.
Choose exactly one priority from: P1 (Urgent Escalation), P2 (Standard Business Priority), P3 (Low Priority Information).
Choose exactly one route from: logistics_team, billing_department, engineering_support, customer_relations.
Return only pure JSON with no markdown wrappers and no extra text.

Input: {stage1_json}
"""


def stage2_reason(stage1_data: dict) -> dict:
    prompt = PROMPT_STAGE_2.format(stage1_json=json.dumps(stage1_data, indent=2))
    response = call_llm(prompt, REASONING_MODEL)
    return safe_parse_json(response, "Stage 2 Strategy Schema", REASONING_MODEL)


# STAGE 3: PRODUCE
PROMPT_STAGE_3 = """
You are an empathetic customer support agent. Draft a concise response under 120 words.
Constraints: stay under 120 words, keep a warm and proactive tone, mention the internal team handling it, and do not promise exact cash refunds.
Return only the final reply text.

Input Metadata: {stage1_json}
Internal Decisions: {stage2_json}
"""


def stage3_produce(stage1_data: dict, stage2_data: dict) -> str:
    prompt = PROMPT_STAGE_3.format(
        stage1_json=json.dumps(stage1_data, indent=2),
        stage2_json=json.dumps(stage2_data, indent=2),
    )
    return call_llm(prompt, PRIMARY_MODEL)


def run_pipeline(raw_input_text: str):
    print("\n" + "═" * 80)
    print("🚀 AGENT PIPELINE PIPING SEQUENCE START")
    print("═" * 80)
    print(f"📥 RAW SOURCE PROSE:\n{raw_input_text}\n")

    s1_output = stage1_understand(raw_input_text)
    print(f"🟩 [STAGE 1: EXTRACTED METADATA]:\n{json.dumps(s1_output, indent=2)}\n")

    s2_output = stage2_reason(s1_output)
    print(f"🟨 [STAGE 2: CHAIN-OF-THOUGHT STRATEGY]:\n{json.dumps(s2_output, indent=2)}\n")

    s3_output = stage3_produce(s1_output, s2_output)
    print(f"🟦 [STAGE 3: FINAL PRODUCED OUTPUT]:\n{s3_output}")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    input_1 = "Hi, my name is Alex Smith. I placed an order (#99382) 9 days ago and my tracking info still hasn't updated. I am really frustrated with how slow this is."
    input_2 = "Can someone check what happened to my delivery? It's been an absolute eternity. Sincerely, Clara Jenkins."
    input_3 = "asdfasdfasd!!! 🚨🚨 1234567"

    run_pipeline(input_1)
    run_pipeline(input_2)
    run_pipeline(input_3)
