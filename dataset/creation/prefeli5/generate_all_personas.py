#!/usr/bin/env python3
"""
Script to generate personas by randomly selecting 10 non-conflicting items from PrefELI5 data
"""

import json
import os
import sys
import random
import time
from typing import List, Dict, Any, Tuple
import openai

# tqdm (with safe fallback)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

def load_prefeli5_data(jsonl_file_path: str) -> List[Dict[str, Any]]:
    """Load data from PrefELI5 JSONL file"""
    try:
        data = []
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        print(f"✅ Loaded {len(data)} preference-question groups")
        return data
    except Exception as e:
        print(f"❌ Failed to load {jsonl_file_path}: {e}")
        return []

def extract_preferences_from_data(data_items: List[Dict[str, Any]]) -> List[str]:
    """Extract only preference texts from data"""
    prefs = []
    for item in data_items:
        if isinstance(item.get("preference"), str) and item["preference"].strip():
            prefs.append(item["preference"].strip())
    return prefs

def build_messages(preferences: List[str]) -> List[Dict[str, str]]:
    """Build messages to send to LLM"""
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(preferences))
    system_prompt = (
        "You are a rigorous consistency checker. "
        "Given a list of user preferences, determine if any of them conflict with one another. "
        "Respond strictly in compact JSON with keys: conflict (boolean), conflicting_pairs (array of objects with keys i, j, reason), notes (string). "
        "Index i and j must be 1-based indices into the provided list. Do not include any text outside the JSON."
    )
    user_prompt = (
        "Here are the user's 10 preferences. Do they contain internal conflicts?\n\n"
        f"Preferences:\n{numbered}\n\n"
        "Return strictly JSON: {\"conflict\": boolean, \"conflicting_pairs\": [{\"i\": int, \"j\": int, \"reason\": string}], \"notes\": string}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def get_openai_client(api_key: str | None) -> openai.OpenAI:
    """Create OpenAI client"""
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise RuntimeError("OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY.")
    openai.api_key = resolved_key
    return openai.OpenAI(api_key=resolved_key)

def call_llm_openai(client: openai.OpenAI, model: str, messages: List[Dict[str, str]], timeout: int) -> str:
    """Call OpenAI API"""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
        top_p=1.0,
        seed=0,
        n=1,
        timeout=timeout,
    )
    return resp.choices[0].message.content

def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response"""
    import re
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError("Failed to parse LLM response as JSON")

def validate_preferences(preferences: List[str], client: openai.OpenAI, model: str, timeout: int, max_retries: int) -> Tuple[bool, Dict[str, Any]]:
    """Check if 10 preferences have conflicts"""
    messages = build_messages(preferences)
    last_err = None
    
    for attempt in range(1, max_retries + 1):
        try:
            content = call_llm_openai(client, model, messages, timeout)
            parsed = parse_json_response(content)
            conflict = bool(parsed.get("conflict", False))
            return (not conflict), parsed
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    
    # Consider as conflict if all retries fail
    return (False, {"error": str(last_err) if last_err else "unknown"})

def create_persona_from_prefeli5(data_items: List[Dict[str, Any]], persona_id: int) -> Dict[str, Any]:
    """Create persona from PrefELI5 data"""
    persona_data = {
        "persona_id": persona_id,
        "generated_at": "2025-01-27T00:00:00.000000",
        "preferences": []
    }

    # Add each data item to persona
    for item in data_items:
        persona_data["preferences"].append({
            "preference": item.get("preference", ""),
            "question": item.get("question", []),
            "explanation": item.get("explanation", "")
        })

    return persona_data

def generate_conflict_free_personas(
    all_data: List[Dict[str, Any]], 
    client: openai.OpenAI, 
    model: str, 
    timeout: int = 60, 
    max_retries: int = 3,
    max_attempts_per_persona: int = 30
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate conflict-free personas"""
    
    available_data = all_data.copy()
    personas = []
    failed_attempts = []
    persona_id = 1
    failed_cnt = 0
    print(f"🎯 Target: Generate conflict-free personas from {len(all_data)} preference-question groups")
    
    while len(available_data) >= 10:
        print(f"\n👤 Attempting to generate Persona {persona_id}... (remaining groups: {len(available_data)})")
        
        # Randomly select 10 groups
        selected_items = random.sample(available_data, 10)
        selected_prefs = extract_preferences_from_data(selected_items)
        
        # Check for conflicts
        is_valid, meta = validate_preferences(selected_prefs, client, model, timeout, max_retries)
        
        if is_valid:
            # Create persona if no conflicts
            persona = create_persona_from_prefeli5(selected_items, persona_id)
            personas.append(persona)
            
            # Remove used groups from available
            for item in selected_items:
                if item in available_data:
                    available_data.remove(item)
            
            print(f"   ✅ Persona {persona_id} generated successfully! (remaining groups: {len(available_data)})")
            persona_id += 1
            failed_cnt = 0
            
        else:
            # Record failure if conflicts exist
            failed_attempts.append({
                "persona_id": persona_id,
                "preferences": selected_prefs,
                "conflict_info": meta
            })
            print(f"   ❌ Persona {persona_id} has conflicts")
            
            failed_cnt += 1
            if failed_cnt == max_attempts_per_persona:
                print(f"   ⚠️  Persona {persona_id} reached maximum attempts ({max_attempts_per_persona}), terminating")
                break
    
    return personas, failed_attempts

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python generate_all_personas.py <output_file> [--api-key KEY] [--model MODEL] [--seed SEED]")
        print("Example: python generate_all_personas.py prefeli5_personas.json --model gpt-4o --seed 0")
        sys.exit(1)
    
    output_file = sys.argv[1]
    
    # Parse command line arguments
    api_key = None
    model = "gpt-4o"
    seed = 0
    
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--api-key" and i + 1 < len(sys.argv):
            api_key = sys.argv[i + 1]
        elif arg == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
        elif arg == "--seed" and i + 1 < len(sys.argv):
            try:
                seed = int(sys.argv[i + 1])
            except Exception:
                print("⚠️  --seed value is not an integer. Using default value 42.")
    
    # Set random seed
    random.seed(seed)
    print(f"🎲 Random seed: {seed}")
    
    # Load PrefELI5 data
    jsonl_file = "final_curated_pairs.jsonl"
    
    if not os.path.exists(jsonl_file):
        print(f"❌ Cannot find {jsonl_file} file.")
        sys.exit(1)
    
    print(f"📁 PrefELI5 data file: {jsonl_file}")
    
    # Load data
    all_data = load_prefeli5_data(jsonl_file)
    
    if not all_data:
        print("❌ No available data.")
        sys.exit(1)
    
    # Create OpenAI client
    try:
        client = get_openai_client(api_key)
        print(f"🤖 OpenAI model: {model}")
    except Exception as e:
        print(f"❌ Failed to create OpenAI client: {e}")
        sys.exit(1)
    
    # Generate conflict-free personas
    start_time = time.time()
    personas, failed_attempts = generate_conflict_free_personas(
        all_data, client, model, timeout=60, max_retries=3
    )
    end_time = time.time()
    
    # Save results
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(personas, f, indent=2, ensure_ascii=False)
    
    # Save failed attempts for debugging
    failed_file = output_file.replace('.json', '_failed_attempts.json')
    with open(failed_file, 'w', encoding='utf-8') as f:
        json.dump(failed_attempts, f, indent=2, ensure_ascii=False)
    
    # Print summary
    total_time = end_time - start_time
    print(f"\n🎉 Generation completed!")
    print(f"✅ Successful personas: {len(personas)}")
    print(f"❌ Failed attempts: {len(failed_attempts)}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"💾 Results saved: {output_file}")
    print(f"📊 Failed attempts log: {failed_file}")
    
    if personas:
        avg_time_per_persona = total_time / len(personas)
        print(f"📈 Average generation time: {avg_time_per_persona:.1f}s/persona")

if __name__ == "__main__":
    main() 