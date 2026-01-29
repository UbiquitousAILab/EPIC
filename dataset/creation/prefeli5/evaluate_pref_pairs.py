import json
import os
from tqdm import tqdm
from openai import OpenAI
from typing import Dict, Any, List
import xml.etree.ElementTree as ET
import time

def load_evaluation_prompt() -> str:
    """Load the evaluation prompt from the text file."""
    with open("evaluate_pref_prompt.txt", "r") as f:
        return f.read()

def load_pairs() -> list:
    pairs = []
    with open("preference_question_pairs.jsonl", "r") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs

def parse_xml_response(response: str) -> Dict[str, Any]:
    """Parse the XML response from the evaluation."""
    try:
        # Clean up the response to only include the XML portion
        xml_start = response.find("<evaluation>")
        xml_end = response.find("</evaluation>") + len("</evaluation>")
        if xml_start == -1 or xml_end == -1:
            raise ValueError("No valid XML found in response")
            
        xml_content = response[xml_start:xml_end]
        root = ET.fromstring(xml_content)
        
        results = {
            "criteria": {
                "contradiction_check": {},
                "prealignment_check": {},
                "violation_check": {},
            },
            "verdict": root.find("final_assessment/verdict").text
        }
        
        # Parse each criterion
        for criterion in ["contradiction_check", "prealignment_check", "violation_check"]:
            element = root.find(criterion)
            if element is not None:
                result = element.find("result").text
                explanation = element.find("explanation")
                explanation_text = explanation.text if explanation is not None and explanation.text is not None else None
                
                results["criteria"][criterion] = {
                    "result": result,
                    "explanation": explanation_text
                }
            
        return results
    except Exception as e:
        print(f"Error parsing XML response: {e}")
        print(f"Response was: {response}")
        return None

def evaluate_pair(client: OpenAI, prompt: str, pair: Dict[str, str], max_retries: int = 3) -> Dict[str, Any]:
    for attempt in range(max_retries):
        try:
            system_prompt = prompt
            user_prompt = f"""Please evaluate this preference-question pair:
Preference: {pair['preference']}
Question: {pair['question']}"""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                seed=0,
            )
            
            response_text = response.choices[0].message.content
            result = parse_xml_response(response_text)
            
            if result is not None:
                return result
            else:
                print(f"Attempt {attempt + 1}: Got None response, retrying...")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error in API call: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            
    print(f"Failed after {max_retries} attempts")
    return None

def main():
    """Main evaluation function."""
    client = OpenAI()
    
    evaluation_prompt = load_evaluation_prompt()
    
    pairs = load_pairs()
    
    results = []
    
    # Load existing results if any
    try:
        with open("evaluation_results.json", "r") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Loaded {start_idx} existing results")
    except FileNotFoundError:
        results = []
        start_idx = 0

    for i, pair in tqdm(enumerate(pairs[start_idx:], start=start_idx), total=10_000-start_idx):
        result = evaluate_pair(client, evaluation_prompt, pair)
        if result:
            results.append({
                "pair": pair,
                "evaluation": result
            })
            # Save results after each evaluation
            with open("evaluation_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
    
    # Print summary
    valid_pairs = sum(1 for r in results if r["evaluation"]["verdict"] == "VALID")
    print(f"\nEvaluation complete!")
    print(f"Total pairs evaluated: {len(results)}")
    print(f"Valid pairs: {valid_pairs}")
    print(f"Invalid pairs: {len(results) - valid_pairs}")
    
if __name__ == "__main__":
    main()
