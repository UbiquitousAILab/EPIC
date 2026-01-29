import os
import json
import time
import argparse
import numpy as np
import faiss
from EPIC_utils import EPICUtils
from EPIC_indexing import EPICIndexing
from EPIC_generation import EPICGeneration
from EPIC_evaluation import EPICEvaluation
import multiprocessing

class EPICMain:
    def __init__(self, mode="all", method="all", device="cuda:0", output_dir="output", dataset="PrefWiki", emb_model_name="facebook/contriever", doc_mode="wiki", vllm_server_url="http://localhost:8008/v1", llm_model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.mode = mode
        self.method = method
        self.device = device
        self.output_dir = output_dir

        self.emb_model_name = emb_model_name
        self.doc_mode = doc_mode
        self.vllm_server_url = vllm_server_url
        self.llm_model_name = llm_model_name
        self.utils = EPICUtils(
            mode=mode,
            method=method,
            device=device,
            output_dir=output_dir,
            dataset=dataset,
            emb_model_name=emb_model_name,
            doc_mode=doc_mode,
            vllm_server_url=vllm_server_url,
            llm_model_name=llm_model_name,
        )
        
        self.indexing = EPICIndexing(self.utils)
        self.generation = EPICGeneration(self.utils)
        self.evaluation = EPICEvaluation(self.utils)
        self._models_loaded = False
        self._chunks_cache = None
        self._embeddings_cache = None
        self._related_chunks_cache = None
        self._related_embeddings_cache = None
    
    def _load_common_resources(self):
        print("Loading common resources...")
        
        self.utils.load_models()
        
        with open(self.utils.chunk_file, "r", encoding="utf-8") as f:
                self._chunks_cache = [json.loads(line)["text"] for line in f]

        self._embeddings_cache = np.load(self.utils.embedding_file)
        
        self._models_loaded = True
        if self._chunks_cache is not None:  
            print(f"   Chunks: {len(self._chunks_cache)}")
        if self._embeddings_cache is not None:
            print(f"   Embeddings: {self._embeddings_cache.shape}")
    
    def get_cached_resources(self):
        if not self._models_loaded:
            self._load_common_resources()
        
        return {
            "chunks": self._chunks_cache,
            "embeddings": self._embeddings_cache,
            "models_loaded": True
        }
    
    def run_batch_processing(self, persona_indices):
        print(f"\nStarting batch processing for {len(persona_indices)} personas...")
        
        self._load_common_resources()

        for persona_index in persona_indices:
            self.run_single_persona(persona_index)
    
    def run_single_persona(self, persona_index):
        print(f"\n=== Processing persona {persona_index}  ===")
        
        if self.llm_model_name == "openai/gpt-oss-20b":
            method_dir = os.path.join(self.utils.output_dir, f"{self.method}_oss/{persona_index}")
            data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
        elif self.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
            method_dir = os.path.join(self.utils.output_dir, f"{self.method}_qwen/{persona_index}")
            data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
        else:
            method_dir = os.path.join(self.utils.output_dir, f"{self.method}/{persona_index}")
            data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")

        print(f"🔍 Data directory: {data_dir}")
        os.makedirs(method_dir, exist_ok=True)
        print(self.utils.output_dir)
        cached_resources = self.get_cached_resources()

        # 1. Indexing
        if self.mode in ["indexing", "all"]:
            print("\n1. Starting indexing...")
            faiss_index_path = os.path.join(data_dir, f"index_flat_{self.emb_model_name.replace('/', '_')}.faiss")

            print(f"faiss_index_path: {faiss_index_path}, dataset: {self.utils.dataset_name}")

            if os.path.exists(faiss_index_path):
                print(f"✅ Indexing already completed. Skipping...")
            else:
                self.indexing.run_indexing_with_cache(persona_index, cached_resources)
                print(f"✅ Indexing completed. Results saved to {method_dir}")
        
        # 2. Generation
        if self.mode in ["generation", "all"]:
            print("\n2. Starting generation...")

            gen_file = os.path.join(method_dir, f"gen_{self.method}_flat_{persona_index}.json")

            if os.path.exists(gen_file):
                print(f"✅ Generation already completed. Skipping...")
            else:
                self.generation.run_generation_with_cache(persona_index, method_dir, cached_resources)
                print(f"✅ Generation completed. Results saved to {gen_file}")
        
        # 3. Evaluation
        if self.mode in ["evaluation", "all"]:
            print("\n3. Starting evaluation...")
            eval_file = os.path.join(method_dir, f"eval_{self.method}_flat_{persona_index}.json")
            if os.path.exists(eval_file):
                print(f"✅ Evaluation already completed. Skipping...")
            else:
                self.evaluation.run_evaluation_with_cache(persona_index, method_dir, cached_resources)
                print(f"✅ Evaluation completed. Results saved to {eval_file}")
        
        print(f"\n=== Completed persona {persona_index} ===")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["EPIC"])
    parser.add_argument("--persona_index", type=str, required=True, help="Persona index (0-10) or 'all'")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0)")
    parser.add_argument("--mode", type=str, required=True, choices=["indexing", "generation", "evaluation", "all"], help="Mode to run: 'indexing', 'generation', 'evaluation', or 'all'")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--emb_model_name", type=str, default="facebook/contriever")
    parser.add_argument("--doc_mode", type=str, required=True, choices=["wiki", "eli5", "lmsys"], help="Document mode: 'wiki' for PrefWiki, PrefRQ, 'eli5' for PrefELI5, 'lmsys' for PrefEval")
    parser.add_argument("--vllm_server_url", type=str, default="8008", help="vLLM server URL or port number (e.g., 8006 or http://localhost:8008/v1)")
    parser.add_argument("--llm_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="LLM model name")
    args = parser.parse_args()

    # vLLM URL Processing
    vllm_server_url = args.vllm_server_url
    if vllm_server_url.isdigit():
        vllm_server_url = f"http://localhost:{vllm_server_url}/v1"
    elif not vllm_server_url.startswith("http"):
        vllm_server_url = f"http://localhost:{vllm_server_url}/v1"
    
    # Persona index setting
    if args.dataset == "PrefWiki":
        indices = list(range(57)) if args.persona_index == "all" else [int(args.persona_index)]
    elif args.dataset == "PrefRQ":
        indices = list(range(90)) if args.persona_index == "all" else [int(args.persona_index)]
    elif args.dataset == "PrefELI5":
        indices = list(range(73)) if args.persona_index == "all" else [int(args.persona_index)]
    elif args.dataset == "PrefEval":
        indices = list(range(57)) if args.persona_index == "all" else [int(args.persona_index)]

    # EPICMain instance creation
    epic = EPICMain(
        mode=args.mode,
        method=args.method,
        device=args.device,
        output_dir=args.output_dir,
        dataset=args.dataset,
        emb_model_name=args.emb_model_name,
        doc_mode=args.doc_mode,
        vllm_server_url=vllm_server_url,
        llm_model_name=args.llm_model_name,
    )

    if indices:
        epic.run_batch_processing(indices)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
