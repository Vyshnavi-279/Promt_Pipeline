import os
import json
import dotenv
import pip
import requests
from dotenv import load_dotenv
load_dotenv()



# ---------------------------------------------------------
# GLOBAL CONFIGURATION & API SETUP
# ---------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")
MODEL_NAME = "google/gemini-2.5-flash" # Or your preferred model from Day 1

def call_llm(prompt: str) -> str:
    """Wrapper function around OpenRouter API to fetch LLM responses."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"API Error: {str(e)}"

def safe_parse_json(llm_output: str, prompt_context: str, current_attempt=1, max_attempts=3) -> dict:
    """
    Bulletproof handoff helper: Parses JSON from the LLM output. 
    If it's malformed, it re-prompts the model with the error to self-correct.
    """
    # Clean up markdown code blocks if the model wrapped its response in ```json ... ```
    cleaned = llm_output.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        if current_attempt >= max_attempts:
            print(f"!!! JSON Parsing failed after {max_attempts} attempts. Returning safety fallback.")
            return {"error": "Invalid JSON structural format returned from LLM", "raw_output": llm_output}
        
        print(f"--- [Handoff Warning] Malformed JSON caught on attempt {current_attempt}. Re-prompting to heal...")
        healing_prompt = f"""
        You previously outputted invalid JSON that caused a parsing error: '{str(e)}'.
        
        Here was your raw output:
        {llm_output}
        
        Please correct the text format and output valid, schema-compliant JSON only. Do not include introductory text or trailing conversational filler.
        Context Requirements:
        {prompt_context}
        """
        corrected_output = call_llm(healing_prompt)
        return safe_parse_json(corrected_output, prompt_context, current_attempt + 1, max_attempts)

# ---------------------------------------------------------
# STAGE 1: UNDERSTAND (Role + Structured Output)
# ---------------------------------------------------------
PROMPT_STAGE_1 = """
You are an expert customer operations coordinator. Your role is to carefully analyze incoming support queries.
Extract key variables and output them exclusively in a clean JSON format.

If critical metadata fields like order_id or names are completely missing or unintelligible from the text, handle it gracefully by populating that key with "UNKNOWN".

Input Text:
"{text}"

Output JSON Format Required:
{{
    "customer_name": "string or UNKNOWN",
    "order_id": "string or UNKNOWN",
    "core_issue": "string",
    "days_waiting": "integer or UNKNOWN",
    "sentiment": "highly frustrated | neutral | satisfied"
}}
"""

def stage1_understand(raw_ticket: str) -> dict:
    prompt = PROMPT_STAGE_1.format(text=raw_ticket)
    response = call_llm(prompt)
    return safe_parse_json(response, "Stage 1 structural output blueprint")

# ---------------------------------------------------------
# STAGE 2: REASON (Chain-of-Thought)
# ---------------------------------------------------------
PROMPT_STAGE_2 = """
You are a senior operations strategist. Evaluate this extracted customer issue summary to assign internal dispatch parameters.
You must use explicit Chain-of-Thought reasoning. Think step-by-step out loud before finalizing the values.

Input JSON from Stage 1:
{stage1_json}

Output JSON Format Required:
{{
    "reasoning_steps": "Write a clear step-by-step breakdown analyzing how days_waiting, core_issue, and sentiment affect priority assessment.",
    "priority": "P1 (Urgent/Escalate) | P2 (Standard Priority) | P3 (Low Priority/Informational)",
    "route": "logistics_team | billing_department | engineering_support | customer_relations"
}}
"""

def stage2_reason(stage1_data: dict) -> dict:
    prompt = PROMPT_STAGE_2.format(stage1_json=json.dumps(stage1_data, indent=2))
    response = call_llm(prompt)
    return safe_parse_json(response, "Stage 2 Chain-of-Thought blueprint")

# ---------------------------------------------------------
# STAGE 3: PRODUCE (Goal-Oriented + Constraints)
# ---------------------------------------------------------
PROMPT_STAGE_3 = """
You are an empathetic, professional customer support agent writing a formal response.
Your goal is to draft a helpful, professional reply to the customer using the extracted data and internal strategic decisions provided.

Constraints:
1. Keep the total response length strictly under 120 words.
2. Maintain a warm, proactive brand tone.
3. Reference specific details provided (like their name or what internal team is investigating it).
4. Do not make structural promises you cannot guarantee (e.g., don't offer precise cash refunds unless authorized).

Input Metadata (Stage 1):
{stage1_json}

Internal Decisions (Stage 2):
{stage2_json}

Provide your final draft response below. No conversational wrappers around the email text.
"""

def stage3_produce(stage1_data: dict, stage2_data: dict) -> str:
    prompt = PROMPT_STAGE_3.format(
        stage1_json=json.dumps(stage1_data, indent=2),
        stage2_json=json.dumps(stage2_data, indent=2)
    )
    return call_llm(prompt)

# ---------------------------------------------------------
# PIPELINE RUNNER ENGINE
# ---------------------------------------------------------
def run_pipeline(raw_input_text: str):
    print("\n" + "="*80)
    print(f"🏁 RUNNING PIPELINE FOR NEW INPUT TICKET")
    print("="*80)
    print(f"📥 RAW INPUT PROSE:\n{raw_input_text}\n")
    
    # Run Stage 1
    s1_output = stage1_understand(raw_input_text)
    print(f"🟩 [STAGE 1 OUTPUT - UNDERSTAND]:\n{json.dumps(s1_output, indent=2)}\n")
    
    # Run Stage 2
    s2_output = stage2_reason(s1_output)
    print(f"🟨 [STAGE 2 OUTPUT - REASON]:\n{json.dumps(s2_output, indent=2)}\n")
    
    # Run Stage 3
    s3_output = stage3_produce(s1_output, s2_output)
    print(f"🟦 [STAGE 3 OUTPUT - PRODUCE / FINAL RESPONSE]:\n{s3_output}")
    print("="*80 + "\n")
    
    return s1_output, s2_output, s3_output

# ---------------------------------------------------------
# EVALUATION TEST RUNS
# ---------------------------------------------------------
if __name__ == "__main__":
    # Test Input 1: Standard structured problem
    input_1 = "Hi, my name is Alex Smith. I placed an order (#99382) 9 days ago and my tracking info still hasn't updated. I am really frustrated with how slow this is."
    
    # Test Input 2: Tricky edge scenario
    input_2 = "Can someone check what happened to my delivery? It's been an absolute eternity. Sincerely, Clara Jenkins."
    
    # Test Input 3: Bad/Malformed input (survives gracefully)
    input_3 = "asdfasdfasd!!! 🚨🚨 1234567"
    
    # Run all three inputs sequentially
    run_pipeline(input_1)
    run_pipeline(input_2)
    run_pipeline(input_3)