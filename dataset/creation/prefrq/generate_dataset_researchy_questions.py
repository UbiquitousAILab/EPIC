import os
import re
import json
import openai
import argparse
from tqdm import tqdm
from typing import Dict, Any
from datasets import load_dataset
import xml.etree.ElementTree as ET

ROOT_DIR = "."

def sample_questions(dataset, split, field, output_file):
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        if "train" in split:
            for row in dataset["train"]:
                intrinsic_scores = row["intrinsic_scores"]
                if intrinsic_scores[field] >= 10:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
        if "test" in split:
            for row in dataset["test"]:
                intrinsic_scores = row["intrinsic_scores"]
                if intrinsic_scores[field] >= 10:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
    print(f"{count} rows saved to {output_file}")
    questions = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
    output_file = output_file.replace(".jsonl", ".txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(q + "\n")
    print(f"{count} rows saved to {output_file}")

def load_prompt(prompt_file):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()

def extract_pqe_from_response(text):
    preference = re.search(r"<preference>(.*?)</preference>", text, re.DOTALL).group(1).strip().strip('"')
    question = re.search(r"<question>(.*?)</question>", text, re.DOTALL).group(1).strip().strip('"')
    explanation = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL).group(1).strip().strip('"')
    return preference, question, explanation

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
            },
            "verdict": root.find("final_assessment/verdict").text
        }
        
        # Parse each criterion
        for criterion in ["contradiction_check", "prealignment_check"]:
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

def evaluate_pair(client: openai.OpenAI, prompt: str, pair: Dict[str, str]) -> Dict[str, Any]:
    """Evaluate a single preference-question pair using GPT-4."""
    system_prompt = prompt
    user_prompt = f"""Please evaluate this preference-question pair:
Preference: {pair['preference']}
Question: {pair['question']}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        
        response_text = response.choices[0].message.content
        return parse_xml_response(response_text)
        
    except Exception as e:
        print(f"Error in API call: {e}")
        return None

def generate_preference_and_explanation_for_question(client, original_question):
    system_prompt = "You are a helpful assistant."
    prompt = load_prompt(os.path.join(ROOT_DIR, "generate_pref_prompt.txt"))
    user_prompt = prompt.format(question=original_question)
    
    eval_system_prompt = load_prompt(os.path.join(ROOT_DIR, "eval_pref_prompt.txt"))
    
    max_trials = 10

    for trial_count in range(1, max_trials + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8
            )
            content = response.choices[0].message.content
            preference, question, explanation = extract_pqe_from_response(content)
            # return preference, question, explanation, content
            result = evaluate_pair(client, eval_system_prompt, {"preference": preference, "question": original_question})
            if result and result["verdict"] == "VALID":
                return preference, original_question, explanation, content
            else:
                # Save failed attempt to failed_attempts.jsonl
                failed_attempt = {
                    "preference": preference,
                    "question": original_question,
                    "explanation": explanation,
                    "result": result
                }
                with open("failed_attempts.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(failed_attempt, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Trial {trial_count + 1} failed for question: {original_question}... Error: {str(e)}")
        
    
    # If all trials failed, log and return error
    print(f"All {max_trials} trials failed for question: {original_question}...")
    return "Failed", "Failed", "Failed", "Failed"

def generate_preference_question_pairs_file(client, input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, total=1269):
            question = json.loads(line)["question"]
            p, q, e, c = generate_preference_and_explanation_for_question(client, question)
            try:
                record = {"preference": p, "question": question, "explanation": e}
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            except:
                f_out.write(c + "\n")
            f_out.flush()
            
    # Generate pretty version
    with open(output_file, "r", encoding="utf-8") as f_in,\
         open(output_file.replace(".jsonl", ".json"), "w", encoding="utf-8") as f_out:
        records = [json.loads(line) for line in f_in if line.strip()]
        json.dump(records, f_out, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Process dataset for researchy questions or corpus.")
    parser.add_argument("mode", choices=["question", "preference", "persona"], help="question, preference, or persona")
    # parser.add_argument("--field", choices=["subjective", "multi-faceted"], help="subjective or multi-faceted")
    args = parser.parse_args()

    os.makedirs(os.path.join(ROOT_DIR, args.field), exist_ok=True)
    sampled_rq_file = os.path.join(ROOT_DIR, f"{args.field}/sampled_researchy_questions.jsonl")
    pq_pairs_file = os.path.join(ROOT_DIR, f"{args.field}/preference_question_pairs.jsonl")

    if args.mode == "question":
        ds = load_dataset("corbyrosset/researchy_questions")
        # sample_questions(ds,"train+test", args.field, output_file=sampled_rq_file)
        sample_questions(ds,"train+test", "subjective", output_file=sampled_rq_file)
    elif args.mode == "preference":
        client = openai.OpenAI()
        generate_preference_question_pairs_file(client, input_file=sampled_rq_file, output_file=pq_pairs_file)
    elif args.mode == "persona":
        pq_pairs = []
        with open(pq_pairs_file, "r", encoding="utf-8") as f:
            for line in f:
                pq_pairs.append(json.loads(line))


if __name__ == "__main__":
    main()