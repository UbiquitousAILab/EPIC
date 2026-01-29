#!/usr/bin/env python3
"""
Script to generate personas by randomly selecting NUM_PREFERENCES that don't conflict from all preferences
Repeat until all preferences are exhausted
"""

import json
import os
import sys
import random
import time
from typing import List, Dict, Any, Tuple
import openai

NUM_PREFERENCES = 10

# tqdm (with safe fallback)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

def load_all_preferences(pq_pairs_file: str) -> List[Dict[str, Any]]:
    """Load all preferences from all_topics_newq.json"""
    try:
        with open(pq_pairs_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"📚 Loaded {len(data)} preferences in total")
        return data
    except Exception as e:
        print(f"❌ Failed to load {pq_pairs_file}: {e}")
        return []

def extract_preferences_from_persona(preferences: List[Dict[str, Any]]) -> List[str]:
    """Extract only preference texts from preference list"""
    prefs = []
    for pref in preferences:
        if isinstance(pref.get("preference"), str) and pref["preference"].strip():
            prefs.append(pref["preference"].strip())
    return prefs

def build_messages(preferences: List[str]) -> List[Dict[str, str]]:
    """Construct messages to send to LLM"""
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(preferences))
    system_prompt = (
        "You are a rigorous consistency checker. "
        "Given a list of user preferences, determine if any of them conflict with one another. "
        "Respond strictly in compact JSON with keys: conflict (boolean), conflicting_pairs (array of objects with keys i, j, reason), notes (string). "
        "Index i and j must be 1-based indices into the provided list. Do not include any text outside the JSON."
    )
    user_prompt = (
        f"Here are the user's {NUM_PREFERENCES} preferences. Do they contain internal conflicts?\n\n"
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
        seed=0, # ! This is required
        n=1,    # ! This is also required
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
    """Check if NUM_PREFERENCES preferences conflict with each other"""
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
    
    # Consider as conflict if all retries failed
    return (False, {"error": str(last_err) if last_err else "unknown"})

def create_persona_from_preferences(preferences: List[Dict[str, Any]], persona_id: int) -> Dict[str, Any]:
    """Create persona data from preference list"""
    persona_data = {
        "persona_id": persona_id,
        "generated_at": "2025-01-27T00:00:00.000000",
        "preferences": preferences
    }
    return persona_data

def generate_conflict_free_personas(
    all_preferences: List[Dict[str, Any]], 
    client: openai.OpenAI, 
    model: str, 
    timeout: int = 60, 
    max_retries: int = 3,
    max_attempts_per_persona: int = 100
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate conflict-free personas"""
    
    available_preferences = all_preferences.copy()
    personas = []
    failed_attempts = []
    persona_id = 1
    
    print(f"🎯 Goal: Generate conflict-free personas from {len(all_preferences)} preferences")
    
    fail_count = 0
    while len(available_preferences) >= NUM_PREFERENCES:
        print(f"\n👤 Attempting to create Persona {persona_id}... (Remaining preferences: {len(available_preferences)})")
        
        # Randomly select NUM_PREFERENCES preferences
        selected_prefs = random.sample(available_preferences, NUM_PREFERENCES)
        selected_texts = extract_preferences_from_persona(selected_prefs)
        
        # Check for conflicts
        is_valid, meta = validate_preferences(selected_texts, client, model, timeout, max_retries)
        
        if is_valid:
            # Create persona if no conflicts
            persona = create_persona_from_preferences(selected_prefs, persona_id)
            personas.append(persona)
            
            # Remove used preferences from available list
            for pref in selected_prefs:
                if pref in available_preferences:
                    available_preferences.remove(pref)
            
            print(f"   ✅ Persona {persona_id} created successfully! (Remaining preferences: {len(available_preferences)})")
            persona_id += 1
            fail_count = 0
            
        else:
            # Record failure if conflicts exist
            failed_attempts.append({
                "persona_id": persona_id,
                "preferences": selected_texts,
                "conflict_info": meta
            })
            print(f"   ❌ Persona {persona_id} has conflicts")

            if fail_count >= 30:
                print("   ⚠️  30 consecutive failures, stopping.")
                break
            fail_count += 1

    return personas, failed_attempts


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python generate_all_personas.py <output_file> [--api-key KEY] [--model MODEL] [--seed SEED]")
        print("Example: python generate_all_personas.py conflict_free_personas.json --model gpt-4o --seed 0")
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
    
    # Fix random seed
    random.seed(seed)
    print(f"🎲 Random seed: {seed}")        
    
    # # selected_pref_newq folder path
    # topics_dir = "selected_pref_newq"
    
    # if not os.path.exists(topics_dir):
    #     print(f"❌ Cannot find {topics_dir} folder.")
    #     sys.exit(1)
    
    # print(f"📁 Topic data folder: {topics_dir}")
    
    # # Step 1: Generate all_topics_newq.json
    # all_topics_file = os.path.join(topics_dir, "all_topics_newq.json")
    # all_preferences = generate_all_topics_json(topics_dir, all_topics_file)
    
    all_preferences = []
    with open("subjective/preference_question_pairs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["preference"].strip() != "Failed":
                all_preferences.append(row)

    if not all_preferences:
        print("❌ No available preferences.")
        sys.exit(1)    
    
    # Create OpenAI client
    # try:
    #     client = get_openai_client(api_key)
    #     print(f"🤖 OpenAI model: {model}")
    # except Exception as e:
    #     print(f"❌ Failed to create OpenAI client: {e}")
    #     sys.exit(1)
    client = openai.Client()

    # Generate conflict-free personas
    start_time = time.time()
    personas, failed_attempts = generate_conflict_free_personas(
        all_preferences, client, model, timeout=60, max_retries=3
    )
    end_time = time.time()
    
    # Save results
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(personas, f, indent=2, ensure_ascii=False)
    
    # Save failed attempts as well (for debugging)
    failed_file = output_file.replace('.json', '_failed_attempts.json')
    with open(failed_file, 'w', encoding='utf-8') as f:
        json.dump(failed_attempts, f, indent=2, ensure_ascii=False)
    
    # Print summary
    total_time = end_time - start_time
    print(f"\n🎉 Generation completed!")
    print(f"✅ Successful personas: {len(personas)}")
    print(f"❌ Failed attempts: {len(failed_attempts)}")
    print(f"⏱️  Time taken: {total_time:.1f} seconds")
    print(f"💾 Results saved: {output_file}")
    print(f"📊 Failure log: {failed_file}")
    
    if personas:
        avg_time_per_persona = total_time / len(personas)
        print(f"📈 Average generation time: {avg_time_per_persona:.1f} seconds/persona")

if __name__ == "__main__":
    main()
