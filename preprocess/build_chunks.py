import re
import json
import argparse


def load_documents(file_path):
    """Load the documents"""
    all_docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                doc = json.loads(line)
                if doc.get("text", "").strip():
                    all_docs.append(doc)
            except json.JSONDecodeError:
                continue
    return all_docs

def clean_text(text):
    # Remove tags in the form of &lt;...&gt;
    text = re.sub(r'&lt;.*?&gt;', '', text)
    return text.strip()

def chunk_documents_sentencewise(docs, chunk_size=100):
    """Split the documents into sentences and group them into 100-word chunks (including removal of unnecessary tags)"""
    chunks = []
    for doc in docs:
        # 1. Remove unnecessary tags
        clean_doc_text = clean_text(doc["text"])
        # 2. Split into sentences (periods, exclamation marks, question marks, line breaks, etc.)
        sentences = re.split(r'(?<=[.!?]) +', clean_doc_text)
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            words = sentence.split()
            n_words = len(words)
            if n_words == 0:
                continue

            # If a single sentence is longer than the chunk_size,
            # include only up to chunk_size and start a new chunk
            if n_words > chunk_size:
                if current_chunk:
                    chunks.append({
                        "id": doc["id"],
                        "title": doc["title"],
                        "text": " ".join(current_chunk)
                    })
                    current_chunk = []
                    current_length = 0
                chunks.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": sentence
                })
                continue
            # If adding a sentence to a chunk exceeds the chunk_size,
            # include it in the current chunk and then start a new chunk
            if current_length + n_words > chunk_size:
                # Add the sentence to the current chunk
                current_chunk.append(sentence)
                current_length += n_words
                
                # Save the current chunk
                chunks.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": " ".join(current_chunk)
                })
                
                # Start a new chunk
                current_chunk = []
                current_length = 0
            else:
                current_chunk.append(sentence)
                current_length += n_words

        # Process the final chunk
        if current_chunk:
            chunks.append({
                "id": doc["id"],
                "title": doc["title"],
                "text": " ".join(current_chunk)
            })

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_file", type=str, required=True, help="Document file path")
    args = parser.parse_args()

    input_file = args.doc_file
    output_file = args.doc_file.replace("doc", "chunk")
    docs = load_documents(input_file)
    chunks = chunk_documents_sentencewise(docs, chunk_size=100)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"{len(chunks)} docs saved in {output_file}")


if __name__ == "__main__":
    main()
