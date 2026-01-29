import os
import sys
import json

questions = []
with open("sampled_eli5_questions_50000.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        questions.append(json.loads(line))

INVALID_TEXTS = ["", "[removed]", "[deleted]", "."]

questions = [q for q in questions if q["selftext"] not in INVALID_TEXTS and "_URL_0_" not in q["selftext"]]
question_ids = [
    q["q_id"]
    for q in questions
    if q["selftext"] not in INVALID_TEXTS and "_URL_0_" not in q["selftext"]]


ELI5_DIR = sys.argv[1]

CC_DOCUMENTS        = os.path.join(ELI5_DIR, "data_creation/processed_data/collected_docs/explainlikeimfive/{i}.json")
OUTPUT_JSONL        = "eli5_supporting_docs_50000.jsonl"
MISSING_LOG         = "eli5_missing_question_ids_50000.txt"
FINAL_QUESTIONS     = "final_eli5_questions_10000.jsonl"

# Load CC_DOCUMENTS (from 0.json to 9.json)
cc_docs = {}
for i in range(10):
    with open(CC_DOCUMENTS.format(i=i), "r", encoding="utf-8") as f:
        for obj in json.load(f):
            q_id, doc = obj
            # print(q_id)
            cc_docs[q_id] = doc
        print(f"{CC_DOCUMENTS.format(i=i)} is loaded")

# Find Supporting Documents
supporting_docs = []
missing_question_ids = []
existing_question_indices = []
seen = set()

for i, question_id in enumerate(question_ids):
    if len(existing_question_indices) == 10000:
        break
    docs = cc_docs.get(question_id)
    if docs is None:
        missing_question_ids.append(question_id)
    else:
        existing_question_indices.append(i)
        for doc in docs:
            if doc["ccid"] not in seen:
                seen.add(doc["ccid"])
                supporting_docs.append(doc)
        

# Save results
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for obj in supporting_docs:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
with open(MISSING_LOG, "w", encoding="utf-8") as f:
    for question_id in missing_question_ids:
        f.write(question_id + "\n")
with open(FINAL_QUESTIONS, "w", encoding="utf-8") as f:
    for i, question_i in enumerate(existing_question_indices):
        f.write(json.dumps(questions[question_i], ensure_ascii=False) + "\n")
print(f"{len(supporting_docs)} docs saved, {len(missing_question_ids)} q_ids missing out of {len(question_ids)}")