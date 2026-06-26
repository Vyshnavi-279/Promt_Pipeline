import os
import json
import requests
from dotenv import load_dotenv

# Load local environment variables dynamically if present
load_dotenv()

# ---------------------------------------------------------
# GLOBAL CONFIGURATION & API SETUP
# ---------------------------------------------------------
# USING TOP-TIER FREE MODELS TO AVOID 402 CREDIT ERRORS
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")

# Primary model used for structured extraction and formatting tasks
PRIMARY_MODEL = "meta-llama/llama-3.3-70b-instruct:free" 
# Secondary model used for reasoning (Model-Mix Stretch Goal for bonus points!)
REASONING_MODEL = "google/gemini-2.5-flash:free"

def call_llm(prompt: str, model_name: str) -> str:
    """Robust API wrapper for OpenRouter with error forwarding."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Vyshnavi-279/Promt_Pipeline",
        "X-Title": "Agentic AI Homework Pipeline"
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1  # Low temperature ensures highly predictable, structured data
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        # Check if it's an API error response with text details
        if 'response' in locals() and hasattr(response, 'text'):
            return f"API_ERROR: {response.status_code} - {response.text}"
        return f"API_ERROR: {str(e)}"

def safe_parse_json(llm_output: str, prompt_context: str, model_name: str, current_attempt=1, max_attempts=3) -> dict:
    """
    Self-Healing JSON Engine: Detects structural anomalies or Markdown wrappers,
    and automatically re-prompts the LLM with error logs to auto-correct the format.
    """
    if "API_ERROR:" in llm_output:
        return {"error": "Upstream API failure", "details": llm_output}

    cleaned = llm_output.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        if current_attempt >= max_attempts:
            return {"error": "Parser Exhausted", "raw_output": llm_output}
        
        print(f"⚠️  [Self-Healing Loop] Malformed JSON caught on attempt {current_attempt}. Re-prompting...")
        healing_prompt = f"""
        You are an automated format reconciliation sub-agent. Your previous output failed strict JSON parsing.
        
        Parsing Error Encountered: {str(e)}
        Your Raw Output Was:
        {llm_output}
        
        Task: Re-format that exact output into valid, clean, parseable JSON conforming exactly to the structural expectations of: {prompt_context}.
        Do not add code blocks, markdown text, introductory text, or trailing explanations. Return raw JSON string only.
        """
        corrected_output = call_llm(healing_prompt, model_name)
        return safe_parse_json(corrected_output, prompt_context, model_name, current_attempt + 1, max_attempts)

# ---------------------------------------------------------
# STAGE 1: UNDERSTAND (Role + Strict Extractor)
# ---------------------------------------------------------
PROMPT_STAGE_1 = """
You are a precision data-extraction system. Analyze the following customer support ticket text.
Extract core entities and structural metadata into the exact JSON schema defined below.

If critical metadata fields like order_id or names are completely missing or unintelligible from the text, populate that key with "UNKNOWN".

Input Customer Ticket Text:
"{text}"

Output JSON Format Required (Strict):
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

# ---------------------------------------------------------
# STAGE 2: REASON (Chain-of-Thought + Strategic Routing)
# ---------------------------------------------------------
PROMPT_STAGE_2 = """
You are an advanced operations dispatch strategist. Evaluate this structured customer issue tracking record.
You must use a Chain-of-Thought reasoning framework. Think step-by-step out loud inside the 'reasoning_steps' key before deciding the internal priority level and delivery target route.

Input Structured Matrix (Stage 1):
{stage1_json}

Output JSON Format Required (Strict):
{{
    "reasoning_steps": "Analyze step-by-step how the user's emotional sentiment, days waiting, and explicit problem category impact priority and internal team routing.",
    "priority": "P1 (Urgent Escalation) | P2 (Standard Business Priority) | P3 (Low Priority Information)",
    "route": "logistics_team | billing_department | engineering_support | customer_relations"
}}
"""

def stage2_reason(stage1_data: dict) -> dict:
    prompt = PROMPT_STAGE_2.format(stage1_json=json.dumps(stage1_data, indent=2))
    # Using Model-Mix: We pass the data to our dedicated Reasoning model for deeper strategy mapping!
    response = call_llm(prompt, REASONING_MODEL)
    return safe_parse_json(response, "Stage 2 Strategy Schema", REASONING_MODEL)

# ---------------------------------------------------------
# STAGE 3: PRODUCE (Goal-Driven Generation + Constraints)
# ---------------------------------------------------------
PROMPT_STAGE_3 = """
You are an empathetic customer relationship specialist executing formal communication channels.
Draft a concise, high-quality, professional response email based exclusively on the provided upstream metadata parameters.

Operational Execution Constraints:
1. Your response message must stay strictly under 120 words.
2. Maintain a warm, proactive, solution-oriented brand voice.
3. Dynamically reference extracted elements (like customer name and internal group handling it) if known.
4. Do not commit the brand to explicit financial compensations or fixed timelines unless predefined.

Input Customer Profile (Stage 1):
{stage1_json}

Strategic Routing Decisions (Stage 2):
{stage2_json}

Output raw prose email draft directly. Do not include introductory wrappers, markdown block decorations, or sign-offs outside the email.
"""

def stage3_produce(stage1_data: dict, stage2_data: dict) -> str:
    prompt = PROMPT_STAGE_3.format(
        stage1_json=json.dumps(stage1_data, indent=2),
        stage2_json=json.dumps(stage2_data, indent=2)
    )
    return call_llm(prompt, PRIMARY_MODEL)

# ---------------------------------------------------------
# SYSTEM AGENTIC PIPELINE EXECUTION ENGINE
# ---------------------------------------------------------
def run_pipeline(raw_input_text: str):
    print("\n" + "═"*80)
    print(f"🚀 AGENT PIPELINE PIPING SEQUENCE START")
    print("═"*80)
    print(f"📥 RAW SOURCE PROSE:\n{raw_input_text}\n")
    
    # Run Stage 1
    s1_output = stage1_understand(raw_input_text)
    print(f"🟩 [STAGE 1: EXTRACTED METADATA] (Model: {PRIMARY_MODEL}):\n{json.dumps(s1_output, indent=2)}\n")
    
    # Run Stage 2
    s2_output = stage2_reason(s1_output)
    print(f"🟨 [STAGE 2: CHAIN-OF-THOUGHT STRATEGY] (Model: {REASONING_MODEL}):\n{json.dumps(s2_output, indent=2)}\n")
    
    # Run Stage 3
    s3_output = stage3_produce(s1_output, s2_output)
    print(f"🟦 [STAGE 3: FINAL PRODUCED OUTPUT]:\n{s3_output}")
    print("═"*80 + "\n")
    
    return s1_output, s2_output, s3_output

if __name__ == "__main__":
    # Test Suite Cases 
    input_1 = "Hi, my name is Alex Smith. I placed an order (#99382) 9 days ago and my tracking info still hasn't updated. I am really frustrated with how slow this is."
    input_2 = "Can someone check what happened to my delivery? It's been an absolute eternity. Sincerely, Clara Jenkins."
    input_3 = "asdfasdfasd!!! 🚨🚨 1234567"
    
    run_pipeline(input_1)
    run_pipeline(input_2)
    run_pipeline(input_3)