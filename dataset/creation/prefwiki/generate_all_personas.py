#!/usr/bin/env python3
"""
Script to generate personas by randomly selecting 10 non-conflicting preferences from all preferences
Repeats until all 570 preferences are exhausted
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

def load_topic_data(topic_file_path: str) -> List[Dict[str, Any]]:
    """Load preference data from topic-specific JSON file"""
    try:
        with open(topic_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Failed to load {topic_file_path}: {e}")
        return []

def load_all_preferences(all_topics_file: str) -> List[Dict[str, Any]]:
    """Load all preferences from all_topics_newq.json"""
    try:
        with open(all_topics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"📁 Successfully loaded {len(data)} preferences")
        return data
    except Exception as e:
        print(f"❌ Failed to load {all_topics_file}: {e}")
        return []

def extract_preferences_from_persona(preferences: List[Dict[str, Any]]) -> List[str]:
    """Extract preference text from preference list"""
    prefs = []
    for pref in preferences:
        if isinstance(pref.get("original_preference"), str) and pref["original_preference"].strip():
            prefs.append(pref["original_preference"].strip())
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

def create_persona_from_preferences(preferences: List[Dict[str, Any]], persona_id: int) -> Dict[str, Any]:
    """Create persona data from preference list"""
    persona_data = {
        "persona_id": persona_id,
        "generated_at": "2025-01-27T00:00:00.000000",
        "topics": {}
    }

    # Group preferences by topic
    topic_groups: Dict[str, List[Dict[str, Any]]] = {}
    for pref in preferences:
        topic = pref.get("topic", "unknown")
        topic_groups.setdefault(topic, []).append(pref)

    # Store all preferences as array for each topic
    for topic_name, prefs in topic_groups.items():
        persona_data["topics"][topic_name] = [
            {
                "preference": p.get("original_preference", ""),
                "questions": p.get("new_questions", [])
            }
            for p in prefs
            if isinstance(p.get("original_preference", ""), str) and p.get("original_preference", "").strip()
        ]

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
    
    while len(available_preferences) >= 10:
        print(f"\n👤 Attempting to create Persona {persona_id}... (remaining preferences: {len(available_preferences)})")
        
        # Randomly select 10 preferences
        selected_prefs = random.sample(available_preferences, 10)
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
            
            print(f"   ✅ Persona {persona_id} created successfully! (remaining preferences: {len(available_preferences)})")
            persona_id += 1
            
        else:
            # Record failed attempt if conflicts found
            failed_attempts.append({
                "persona_id": persona_id,
                "preferences": selected_texts,
                "conflict_info": meta
            })
            print(f"   ❌ Persona {persona_id} has conflicts")
            
            # Skip to next persona after too many attempts
            if len(failed_attempts) >= max_attempts_per_persona:
                print(f"   ⚠️  Max attempts reached for Persona {persona_id}, skipping")
                persona_id += 1
                failed_attempts = []
    
    return personas, failed_attempts

def generate_all_topics_json(topics_dir: str, output_file: str) -> List[Dict[str, Any]]:
    """Combine all topic-specific JSON files to create all_topics_newq.json"""
    
    print(f"🔄 Combining all topic files from {topics_dir}...")
    
    # Find JSON files (excluding all_topics_newq.json)
    json_files = [f for f in os.listdir(topics_dir) 
                  if f.endswith('.json') and f != 'all_topics_newq.json']
    
    print(f"📁 Found topic files: {len(json_files)}")
    
    all_preferences = []
    topic_counts = {}
    
    for json_file in sorted(json_files):
        topic_name = json_file.replace('_newq.json', '')
        file_path = os.path.join(topics_dir, json_file)
        
        data = load_topic_data(file_path)
        if data:
            # Add topic information to each preference
            for pref in data:
                pref_with_topic = pref.copy()
                pref_with_topic['topic'] = topic_name
                all_preferences.append(pref_with_topic)
            
            topic_counts[topic_name] = len(data)
            print(f"   {topic_name}: {len(data)} preferences")
    
    # Save all_topics_newq.json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_preferences, f, indent=2, ensure_ascii=False)
    
    print(f"✅ all_topics_newq.json created successfully: total {len(all_preferences)} preferences")
    
    # Print topic statistics
    print("\n📊 Preferences count by topic:")
    for topic, count in sorted(topic_counts.items()):
        print(f"   {topic}: {count}")
    
    return all_preferences

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
    
    # selected_pref_newq folder path
    topics_dir = "selected_pref_newq"
    
    if not os.path.exists(topics_dir):
        print(f"❌ {topics_dir} folder not found.")
        sys.exit(1)
    
    print(f"📁 Topic data folder: {topics_dir}")
    
    # Step 1: Generate all_topics_newq.json
    all_topics_file = os.path.join(topics_dir, "all_topics_newq.json")
    all_preferences = generate_all_topics_json(topics_dir, all_topics_file)
    
    if not all_preferences:
        print("❌ No available preferences.")
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
    print(f"⏱️  Elapsed time: {total_time:.1f}s")
    print(f"💾 Results saved: {output_file}")
    print(f"📊 Failure log: {failed_file}")
    
    if personas:
        avg_time_per_persona = total_time / len(personas)
        print(f"📈 Average generation time: {avg_time_per_persona:.1f}s/persona")

if __name__ == "__main__":
    main()
