import os
import re
import nlp
import json
import openai
import random
import argparse
from tqdm import tqdm

ROOT_DIR = "."
SEED = 0
K = 50000

def sample_eli5_questions(dataset, output_file):
    all_questions = []
    for row in dataset["train_eli5"]:
        all_questions.append(row)
    for row in dataset["validation_eli5"]:
        all_questions.append(row)
    for row in dataset["test_eli5"]:
        all_questions.append(row)
    random.seed(SEED)
    sampled_questions = random.sample(all_questions, K)
    with open(output_file, "w", encoding="utf-8") as f:
        for question in sampled_questions:
            f.write(json.dumps(question, ensure_ascii=False) + "\n")

def load_prompt(prompt_file):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()

def extract_pqe_from_response(text):
    preference = re.search(r"<preference>(.*?)</preference>", text, re.DOTALL).group(1).strip().strip('"')
    question = re.search(r"<question>(.*?)</question>", text, re.DOTALL).group(1).strip().strip('"')
    explanation = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL).group(1).strip().strip('"')
    return preference, question, explanation

def generate_preference_and_explanation_for_question(client, question):
    system_prompt = "You are a helpful assistant."
    prompt = load_prompt(os.path.join(ROOT_DIR, "generate_pref_prompt.txt"))
    
    user_input = prompt.format(question=question)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.8
        )
        content = response.choices[0].message.content
        preference, question, explanation = extract_pqe_from_response(content)
        return preference, question, explanation, content
        
    except Exception as e:
        print(f"Error generating questions and explanations: {e}")
        return "Error", "Error", "Error", "Error"

def generate_preference_question_pairs_file(client, input_file, output_file):
    count = 0
    
    # Check if output file exists and count existing lines
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f_existing:
            for line in f_existing:
                if line.strip():
                    count += 1
        print(f"Found existing file with {count} lines. Starting from line {count + 1}")
    
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "a", encoding="utf-8") as f_out:
        
        # Skip lines already processed
        for i, line in enumerate(f_in):
            if i < count:
                continue
                
            item = json.loads(line)
            question = item["title"] + ": " + item["selftext"]
            p, q, e, c = generate_preference_and_explanation_for_question(client, question)
            while p == "Error":
                p, q, e, c = generate_preference_and_explanation_for_question(client, question)
            try:
                record = {"preference": p, "question": question, "explanation": e}
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            except:
                f_out.write(c + "\n")
            f_out.flush()
            count += 1
            if count == 10000:
                break

    # Generate pretty version
    with open(output_file, "r", encoding="utf-8") as f_in,\
         open(output_file.replace(".jsonl", ".json"), "w", encoding="utf-8") as f_out:
        records = [json.loads(line) for line in f_in if line.strip()]
        json.dump(records, f_out, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Process dataset for researchy questions or corpus.")
    parser.add_argument("mode", choices=["question", "preference"], help="question, preference, or corpus")
    args = parser.parse_args()

    sampled_eli5_questions = os.path.join(ROOT_DIR, "sampled_eli5_questions_50000.jsonl")
    final_eli5_questions = os.path.join(ROOT_DIR, "final_eli5_questions_10000.jsonl")
    pq_pairs_file = os.path.join(ROOT_DIR, "preference_question_pairs.jsonl")

    if args.mode == "question":
        ds = nlp.load_dataset("eli5")
        sample_eli5_questions(ds, output_file=sampled_eli5_questions)
    elif args.mode == "preference":
        if not os.path.exists(final_eli5_questions):
            print("Please run extract_valid_questions.py first to create final_eli5_questions_10000.jsonl")
        client = openai.OpenAI()
        generate_preference_question_pairs_file(client, input_file=final_eli5_questions, output_file=pq_pairs_file)


if __name__ == "__main__":
    main()