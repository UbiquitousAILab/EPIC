import os
import json
import time
import faiss
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class EPICIndexing:
    def __init__(self, utils):
        self.utils = utils
        self.method = utils.method
        self.device = utils.device
        self.output_dir = utils.output_dir
        self.emb_model_name = utils.emb_model_name
        self.doc_mode = utils.doc_mode
        self.chunk_file = utils.chunk_file
        self.embedding_file = utils.embedding_file
        self.batch_size = getattr(utils, 'batch_size', 16)  # Default batch size if not set
    
    def run_indexing_with_cache(self, persona_index, cached_resources):
        if self.utils.llm_model_name == "openai/gpt-oss-20b":
            llm_name = "_oss"
        elif self.utils.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
            llm_name = "_qwen"
        else:
            llm_name = ""

        print(f"\n=== Starting indexing for persona {persona_index} ===")

        if self.utils.llm_model_name == "openai/gpt-oss-20b":
            method_dir = os.path.join(self.output_dir, f"{self.method}_oss/{persona_index}")
            data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
        elif self.utils.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
            method_dir = os.path.join(self.output_dir, f"{self.method}_qwen/{persona_index}")
            data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
        else:
            method_dir = os.path.join(self.output_dir, f"{self.method}/{persona_index}")
            data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
        
        
        embeddings_file = os.path.join(data_dir, f"embeddings_{self.emb_model_name.replace('/', '_')}.npy")
        index_file = os.path.join(data_dir, f"index_{self.emb_model_name.replace('/', '_')}.faiss")

        os.makedirs(method_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        print(f"Output directory: {method_dir}")

        print("✅ Using cached resources (models, chunks, embeddings)")
        chunks = cached_resources["chunks"]
        chunk_embeddings = cached_resources["embeddings"]
        print(f"Using {len(chunks)} cached chunks and their embeddings")

        persona = self.utils.load_persona_data(persona_index)
        print(f"Loaded persona data for index {persona_index}")

        print("Loading or generating preference embeddings...")
        preference_list = [block["preference"] for block in persona["preference_blocks"]]
        preference_emb_file_prefix = "nv_" if self.emb_model_name == "nvidia/NV-Embed-v2" else ""
        if self.utils.dataset_name == "PrefWiki":
            preference_emb_file = os.path.join(self.utils.root_dir, f"indexing/{preference_emb_file_prefix}preference_embeddings_{persona_index}_prefwiki_mp.npy")
        elif self.utils.dataset_name == "PrefELI5":
            preference_emb_file = os.path.join(self.utils.root_dir, f"indexing/{preference_emb_file_prefix}preference_embeddings_{persona_index}_prefeli5_mp.npy")
        elif self.utils.dataset_name == "PrefRQ":
            preference_emb_file = os.path.join(self.utils.root_dir, f"indexing/{preference_emb_file_prefix}preference_embeddings_{persona_index}_rq_mp.npy")
        elif self.utils.dataset_name == "PrefEval":
            preference_emb_file = os.path.join(self.utils.root_dir, f"indexing/{preference_emb_file_prefix}preference_embeddings_{persona_index}_prefeval_mp.npy")

        if os.path.exists(preference_emb_file):
            print("Loading existing preference embeddings...")
            preference_embeddings = np.load(preference_emb_file)
        else:
            print("Generating new preference embeddings...")
            preference_embeddings = self.utils.embed_texts_mp(preference_list)\
            
            preference_embeddings = preference_embeddings / np.linalg.norm(preference_embeddings, axis=1, keepdims=True)
            np.save(preference_emb_file, preference_embeddings)
            print(f"Saved preference embeddings to {preference_emb_file}")
            
            
        chunk_embeddings_norm = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        print(f"Using {len(preference_list)} preference embeddings")
        
        start_total = time.time()

        print(self.utils.threshold)
        
        print("\nStarting cosine similarity filtering...")
        kept_save, kept_chunks, filtered_save = [], [], []
        relevant_preferences = []
        relevant_similarities = []
        
        batch_size = self.batch_size
        for i in tqdm(range(0, len(chunk_embeddings_norm), batch_size), desc=f"Filtering persona {persona_index}"):
            batch_embeddings = chunk_embeddings_norm[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]
            
            sims = np.dot(preference_embeddings, batch_embeddings.T)
            above = sims > self.utils.threshold
            mask = np.any(above, axis=0)
            
            for j, (chunk, is_kept) in enumerate(zip(batch_chunks, mask)):
                if is_kept:
                    kept_chunks.append(chunk)
                    relevant_idx = np.where(above[:, j])[0]
                    relevant_prefs = [preference_list[k] for k in relevant_idx]
                    relevant_sim_values = [sims[k, j] for k in relevant_idx]
                    
                    kept_save.append({
                        "chunk": chunk,
                        "relevant_preferences": relevant_prefs,
                        "relevant_similarities": [float(sim) for sim in relevant_sim_values]
                    })
                    relevant_preferences.append(relevant_prefs)
                    relevant_similarities.append(relevant_sim_values)
                else:
                    filtered_save.append({"chunk": chunk})
        
        filter_time = time.time() - start_total
        print(f"Cosine filtering completed. Kept {len(kept_chunks)} chunks out of {len(chunks)}")
        
        cosine_filtering_file = os.path.join(method_dir, "cosine_filtering_results.jsonl")
        self.utils.save_jsonl(cosine_filtering_file, kept_save)
        print(f"✅ Cosine filtering results saved to {cosine_filtering_file}")

      
        print("\nStarting LLM filtering...")
        
        result_info_file = os.path.join(method_dir, "result_info.jsonl")
        inst_file = os.path.join(method_dir, "instructions.jsonl")
        
        filtering_prompt_system = self.utils.load_prompt_template(self.utils.filtering_system)
        filtering_prompt_user = self.utils.load_prompt_template(self.utils.filtering_user)

        preference_text = "\n".join([f"{i+1}. '{p}'" for i, p in enumerate(preference_list)])

        start_llm = time.time()
        results = []

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.utils.process_chunk_rand_prefs, idx, kept_chunks[idx], preference_text, filtering_prompt_user, filtering_prompt_system, relevant_preferences[idx], kept_save[idx]): idx for idx in range(len(kept_chunks))}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"LLM: persona {persona_index}", leave=False, ncols=100):
                result = future.result()
                if result:
                    results.append(result)

            result_info_file = os.path.join(method_dir, "result_info.jsonl")

            self.utils.save_json(result_info_file, results)
            print(f"✅ Result info saved to {result_info_file}")
            
            llm_time = time.time() - start_llm

        filtered, kept = [], [], []
        failed_chunks = [] 
        success_count = 0
        failed_count = 0

        for item in results:
            if item["status"] == "failed":
                failed_chunks.append(item)
                failed_count += 1
                filtered.append({"chunk": item["chunk"]})
            else:
                success_count += 1
                if item["decision"] == "Discard":
                    filtered.append({"chunk": item["chunk"]})
                elif item["decision"] == "Keep":
                    kept_item = {
                        "chunk": item["chunk"],
                        "reason": item["reason"],
                        "relevant_preference": item["relevant_preference"]
                    }
                    kept_item["instruction"] = item.get("instruction", "")
                    kept.append(kept_item)
        
        print(f"LLM filtering completed. Success: {success_count}, Failed: {failed_count}")

        print(f"Results - Filtered: {len(filtered)}, Kept: {len(kept)}")

        if failed_chunks:
            failed_file = os.path.join(method_dir, "failed_chunks.jsonl")
            self.utils.save_jsonl(failed_file, failed_chunks)
            print(f"⚠️ Failed chunks saved to {failed_file}")

        instruction_prompt_system = self.utils.load_prompt_template(self.utils.instruction_system)
        instruction_prompt_user = self.utils.load_prompt_template(self.utils.instruction_user)

        start_instruction = time.time()
        inst_final = []

        if os.path.exists(inst_file):
            print(f"⚠️ Found existing instructions file. Skipping instruction generation: {inst_file}")
            with open(inst_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        inst_final.append(json.loads(line))
            inst_time = 0.0
        else:
            if kept:
                first_entry = kept[0]
                first_chunk = first_entry["chunk"]
                preferences = first_entry.get("relevant_preference", [])
                preference_text_inst = (
                    "\n".join([f"- {p}" for p in preferences])
                    if isinstance(preferences, list)
                    else preferences
                )
                filled_user_prompt = instruction_prompt_user.format(
                    preference=preference_text_inst,
                    chunk=first_chunk,
                    reason=first_entry["reason"],
                )
                prompt_sample = {
                    "system_prompt": instruction_prompt_system,
                    "user_prompt": filled_user_prompt,
                    "full_conversation": (
                        f"System: {instruction_prompt_system}\n\nUser: {filled_user_prompt}"
                    ),
                }
                prompt_file = os.path.join(method_dir, "instruction_prompt_sample.json")
                self.utils.save_json(prompt_file, prompt_sample)
                print(f"✅ Instruction prompt sample saved to {prompt_file}")

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self.utils.inst_single,
                        entry,
                        instruction_prompt_user,
                        inst_prompt_system=instruction_prompt_system,
                    )
                    for entry in kept
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Generating instructions",
                    leave=False,
                    ncols=100,
                ):
                    try:
                        inst_final.append(future.result())
                    except Exception as e:
                        print(f"Instruction generation failed: {e}")
            inst_time = time.time() - start_instruction
            with open(inst_file, "w", encoding="utf-8") as f:
                for item in inst_final:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"✅ Instruction info saved to {inst_file}")

        print(f"Instruction generation completed. Produced {len(inst_final)} instructions")

        print("\nCreating FAISS index (instruction embeddings)...")
        start_faiss = time.time()

        chunk_metadata = []
        for idx, item in enumerate(inst_final):
            chunk = item["chunk"]
            instruction = item.get("instruction", "")
            relevant_prefs = item.get("relevant_preference", [])
            if isinstance(relevant_prefs, str):
                relevant_prefs = [relevant_prefs]
            chunk_metadata.append({
                "id": idx,
                "text": chunk,
                "instruction": instruction,
                "relevant_preferences": relevant_prefs,
                "preference_ids": relevant_prefs,
                "reason": item.get("reason", ""),
                "active": True,
            })

        instructions_list = [item["instruction"] for item in chunk_metadata]
        print(f"Generating embeddings for {len(instructions_list)} instructions...")
        embeddings = self.utils.embed_texts_mp(instructions_list)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        print(f"Generated {len(embeddings)} instruction embeddings")

        dim = embeddings.shape[1]
        print("Creating FAISS IndexFlatIP...")
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

        print("Saving results...")
        faiss.write_index(index, index_file)
        print(f"FAISS saved in {index_file}")
        np.save(embeddings_file, embeddings)

        kept_file = os.path.join(method_dir, "kept.jsonl")
        with open(kept_file, "w", encoding="utf-8") as f:
            for item in chunk_metadata:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✅ Kept chunks with metadata saved to {kept_file}")

        pref_mapping = {
            "preference_to_idx": {pref: idx for idx, pref in enumerate(preference_list)},
            "preference_list": preference_list,
            "total_chunks": len(chunk_metadata),
            "method": self.method,
        }
        pref_mapping_file = os.path.join(method_dir, "preference_mapping.json")
        self.utils.save_json(pref_mapping_file, pref_mapping)
        print(f"✅ Saved preference mapping to {pref_mapping_file}")

        print(f"\n📊 Instruction indexing summary: {len(chunk_metadata)} chunks indexed")
        
        faiss_time = time.time() - start_faiss
        total_time = time.time() - start_total
        
        print("\nGenerating report...")
        fieldnames = ["method", "persona_index", "cosine_kept", "random_kept", "cluster_kept", "llm_filtered", "instruction_generated", "kept", "cosine_filter_time(s)", "random_filter_time(s)", "cluster_filter_time(s)", "llm_time(s)", "instruction_time(s)", "faiss_time(s)", "total_time(s)"]
        row = {
            "method": f"{self.method}{llm_name}",
            "persona_index": f"{persona_index}",
            "cosine_kept": len(kept_chunks),
            "random_kept": len(kept_chunks),
            "cluster_kept": 0,
            "llm_filtered": len(filtered),
            "instruction_generated": len(inst_final),
            "kept": len(kept),
            "cosine_filter_time(s)": f"{filter_time:.2f}",
            "random_filter_time(s)": f"{filter_time:.2f}",
            "cluster_filter_time(s)": "0",
            "llm_time(s)": f"{llm_time:.2f}",
            "instruction_time(s)": f"{inst_time:.2f}",
            "faiss_time(s)": f"{faiss_time:.2f}",
            "total_time(s)": f"{total_time:.2f}"
        }
        self.utils.save_csv(os.path.join(self.output_dir, self.utils.indexing_report_file), fieldnames, row)
        
        print(f"\n=== Completed indexing for persona {persona_index} ===")
        print(f"Total time: {total_time:.2f} seconds")
        return method_dir
