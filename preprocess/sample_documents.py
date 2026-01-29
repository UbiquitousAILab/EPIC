import os
import json
import random
import argparse
from tqdm import tqdm


def sample_wiki_documents(input_dir, output_dir, sample_size):
    # Load raw files
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    print(f"Found {len(jsonl_files)} jsonl files")
    
    all_docs = []
    for file in tqdm(jsonl_files, desc="Loading documents"):
        with open(os.path.join(input_dir, file), 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                all_docs.append(doc)
    print(f"Total {len(all_docs)} documents loaded")

    # Random sample
    print(f"Will sample {sample_size} documents")
    sampled_docs = random.sample(all_docs, sample_size)
    
    # Save result
    count = 0
    if sample_size == 100_000:
        output_file = os.path.join(output_dir, "sampled_wiki_doc.jsonl")
    else:
        output_file = os.path.join(output_dir, f"sampled_wiki_doc_{sample_size}.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in sampled_docs:
            record = {
                "id": count,
                "title": doc["title"],
                "text": doc["text"],
            }
            count += 1
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"{len(sampled_docs)} docs saved in {output_file}")

def sample_eli5_documents(input_dir, output_dir, sample_size):
    # Load raw files
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    print(f"Found {len(json_files)} jsonl files")

    cc_docs = []
    ccids = set()
    for file in tqdm(json_files, desc="Loading documents"):
        with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
            for obj in json.load(f):
                for doc in obj[1]:
                    if doc["ccid"] in ccids:
                        continue
                    ccids.add(doc["ccid"])
                    cc_docs.append(doc)
    print(f"Total {len(cc_docs)} documents loaded")

    # Random sample
    print(f"Will sample {sample_size} documents")
    sampled_docs = random.sample(cc_docs, sample_size)

    # Save result
    count = 0
    if sample_size == 20_000:
        output_file = os.path.join(output_dir, "sampled_eli5_doc.jsonl")
    else:
        output_file = os.path.join(output_dir, f"sampled_eli5_doc_{sample_size}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for obj in sampled_docs:
            record = {
                "id": count,
                "title": obj["ccid"],
                "text": obj["text"][0],
            }
            count += 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(sampled_docs)} docs saved in {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_type", type=str, required=True, 
                        choices=["wiki", "eli5"],
                        help="Document type: wiki, eli5, or ccnews")
    parser.add_argument("--input_dir", type=str, default=None, help="Input directory")
    parser.add_argument("--sample_size", type=int, default=10_000, help="Number of samples")
    args = parser.parse_args()

    random.seed(0)

    output_dir = "."
    if args.doc_type == "wiki":
        sample_wiki_documents(input_dir=args.input_dir, # filtered_wiki_json directory: Wikidump + WikiExtractor
                              output_dir=output_dir,
                              sample_size=args.sample_size)
    elif args.doc_type == "eli5":
        sample_eli5_documents(input_dir=os.path.join(args.input_dir, "data_creation/processed_data/collected_docs/explainlikeimfive"), # ELI5 supporting documents directory
                              output_dir=output_dir,
                              sample_size=args.sample_size)
    

if __name__ == "__main__":
    main()
