import os
import json
import time
import faiss
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class EPICGeneration:
    def __init__(self, utils):
        self.utils = utils
        self.method = utils.method
        self.device = utils.device
        self.output_dir = utils.output_dir
        self.emb_model_name = utils.emb_model_name
        self.doc_mode = utils.doc_mode
        self.chunk_file = utils.chunk_file
        self.embedding_file = utils.embedding_file
        self.batch_size = getattr(utils, 'batch_size', 16)

    def process_query(self, query, preference_text, preferences, index, method_dir, generation_prompt, chunk_metadata=None):
        """Instruction-index retrieval: top-1 preference augmentation + FAISS search."""
        question = query["question"]
        if not chunk_metadata:
            raise ValueError("chunk_metadata is empty; run indexing first to create kept.jsonl")

        start_retrieval = time.time()
        query_emb = self.utils.embed_query_mp(question)

        preference_embs = []
        for pref in preferences:
            preference_embs.append(self.utils.embed_query_mp(pref).squeeze(0))
        preference_embs = np.vstack(preference_embs)
        sims = np.dot(preference_embs, query_emb.T).squeeze()
        top_pref_idx = int(np.argmax(sims))
        top_pref_text = preferences[top_pref_idx]

        query_emb = query_emb + self.utils.embed_query_mp(top_pref_text)
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

        search_k = self.utils.top_k
        _, I = index.search(query_emb.astype(np.float32), search_k)

        retrieved = []
        retrieved_instructions = []
        for idx in I[0]:
            if idx < 0 or idx >= len(chunk_metadata):
                continue
            meta = chunk_metadata[idx]
            retrieved.append(meta["text"])
            retrieved_instructions.append(meta.get("instruction", ""))
            if len(retrieved) >= search_k:
                break

        retrieval_time = time.time() - start_retrieval

        context_parts = []
        for i, (doc, inst) in enumerate(zip(retrieved, retrieved_instructions)):
            if inst:
                context_parts.append(
                    f"Document {i+1}:\nInterpretation Guidance: {inst}\nContent: {doc}"
                )
            else:
                context_parts.append(f"Document {i+1}: {doc}")
        context = "\n\n".join(context_parts)

        filled_prompt = generation_prompt.replace("{context}", context).replace("{question}", question)

        try:
            max_tokens = 8192 if self.utils.llm_model_name == "openai/gpt-oss-20b" else 2048
            generated_text = self.utils.generate_message_vllm(
                messages=[{"role": "user", "content": filled_prompt}],
                system_prompt="You are a helpful assistant for generating responses.",
                max_tokens=max_tokens,
            )

            if generated_text is None:
                print("Warning: LLM returned None response for generation - returning None")
                return None

            return {
                "preference": preference_text,
                "question": question,
                "response_to_q": generated_text,
                "retrieved_docs": context,
                "retrieval_time": retrieval_time,
            }

        except Exception as e:
            print(f"Failed to generate response: {e} - returning None")
            return None

    def run_generation_with_cache(self, persona_index, method_dir, cached_resources):
        print(f"\n=== Starting generation for persona {persona_index} ===")

        data_dir = os.path.join(self.utils.data_dir, str(persona_index))
        model_name_clean = self.emb_model_name.replace("/", "_")
        index_basename = f"index_{model_name_clean}.faiss"

        index_file = None
        for cand in (
            os.path.join(data_dir, index_basename),
            os.path.join(method_dir, index_basename),
        ):
            if os.path.exists(cand):
                index_file = cand
                break
        if index_file is None:
            raise FileNotFoundError(
                f"FAISS index not found for persona {persona_index}. "
                f"Tried {data_dir} and {method_dir}. Run indexing first."
            )

        print(f"Loading FAISS index from: {index_file}")
        index = faiss.read_index(index_file)

        kept_file = os.path.join(method_dir, "kept.jsonl")
        if not os.path.exists(kept_file):
            raise FileNotFoundError(
                f"kept.jsonl not found at {kept_file}. Run indexing first."
            )

        chunk_metadata = []
        with open(kept_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                chunk_metadata.append({
                    "text": item["text"],
                    "preference_ids": item.get(
                        "relevant_preferences",
                        item.get("preference_ids", item.get("relevant_preference", [])),
                    ),
                    "instruction": item.get("instruction", ""),
                    "reason": item.get("reason", ""),
                })

        print(f"✅ Loaded {len(chunk_metadata)} chunks from {kept_file}")
        if index.ntotal != len(chunk_metadata):
            print(
                f"⚠️ Index size ({index.ntotal}) != kept.jsonl rows ({len(chunk_metadata)}). "
                "Re-run indexing if results look wrong."
            )

        pref_counts = {}
        for meta in chunk_metadata:
            for pref in meta.get("preference_ids", []):
                key = (pref[:50] + "...") if len(pref) > 50 else pref
                pref_counts[key] = pref_counts.get(key, 0) + 1
        print(f"Chunks per preference: {pref_counts}")

        persona = self.utils.load_persona_data(persona_index)
        print(f"Loaded persona data for index {persona_index}")

        generation_prompt = self.utils.load_prompt_template(self.utils.generation_prompt)
        print("✅ Using cached models")

        all_results = []
        retrieval_times = []

        for block in persona["preference_blocks"]:
            preference_text = block["preference"]
            preferences = [b["preference"] for b in persona["preference_blocks"]]
            queries = block["queries"]

            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = [
                    executor.submit(
                        self.process_query,
                        query,
                        preference_text,
                        preferences,
                        index,
                        method_dir,
                        generation_prompt,
                        chunk_metadata,
                    )
                    for query in queries
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Processing queries for preference: {preference_text[:50]}...",
                ):
                    result = future.result()
                    if result:
                        all_results.append(result)
                        retrieval_times.append(result["retrieval_time"])

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        output_file = os.path.join(method_dir, f"gen_{self.method}_flat_{persona_index}.json")
        self.utils.save_json(output_file, all_results)
        print(f"✅ Generation results saved to {output_file}")

        if retrieval_times:
            avg_time = np.mean(retrieval_times)
            max_time = np.max(retrieval_times)
            min_time = np.min(retrieval_times)
        else:
            avg_time = max_time = min_time = 0.0

        if self.utils.llm_model_name == "openai/gpt-oss-20b":
            llm_name = "_oss"
        elif self.utils.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
            llm_name = "_qwen"
        else:
            llm_name = ""

        fieldnames = [
            "method", "persona_index",
            "avg_retrieval_time(s)", "max_retrieval_time(s)", "min_retrieval_time(s)",
        ]
        row = {
            "method": f"{self.method}{llm_name}",
            "persona_index": f"{persona_index}",
            "avg_retrieval_time(s)": f"{avg_time:.4f}",
            "max_retrieval_time(s)": f"{max_time:.4f}",
            "min_retrieval_time(s)": f"{min_time:.4f}",
        }
        self.utils.save_csv(
            os.path.join(self.output_dir, self.utils.generation_report_file),
            fieldnames,
            row,
        )

        print(f"\n=== Completed generation for persona {persona_index} ===")
        print(f"Average retrieval time: {avg_time:.4f} seconds")
        return method_dir
