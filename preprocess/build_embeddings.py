import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers import DPRQuestionEncoder, DPRContextEncoder


def mean_pooling(token_embeddings, mask):
    """The mean_pooling function used in Contriever.py"""
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def load_chunks(file_path):
    """Load chunks from JSONL file"""
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(data["text"])
    return chunks

def load_models(model_name):
    """Load embedding model & tokenizer - Apply NVEmbedV2 approach"""
    print(f"Loading {model_name} model...")
    
    if model_name.startswith("dpr"):
        # Load DPR model
        if model_name == "dpr-question":
            model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        else:  # dpr-context
            model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    elif model_name.startswith("nvidia"):
        # Load NVIDIA model with NVEmbedV2 approach
        print("Using NVEmbedV2 approach for NVIDIA model...")
        model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            # device_map="auto"  # Not using accelerate
        )
        tokenizer = None  # AutoModel doesn't need separate tokenizer
        
        # Set batch size (NVEmbedV2 approach)
        if torch.cuda.device_count() > 1:
            batch_size = 16 * torch.cuda.device_count()
            print(f"Using {torch.cuda.device_count()} GPUs with batch size {batch_size}")
        else:
            batch_size = 16
            print(f"Using single GPU with batch size {batch_size}")
        
        return tokenizer, model, None, batch_size  # device=None (using device_map="auto")
    else:
        # Keep existing approach
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            device_map="auto"
        ).eval()
    
    # GPU setting
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Set the batch size
    if torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs with device_map='auto'!")
        batch_size = 32 * num_gpus
        print(f"Model distributed across {num_gpus} GPUs with batch size {batch_size}")
    else:
        batch_size = 32
        print(f"Model loaded on {device} with batch size {batch_size}")
    
    return tokenizer, model, device, batch_size

def generate_embeddings(chunks, tokenizer, model, device, batch_size=32, model_name="facebook/contriever"):
    """Generate embeddings optimized with NVEmbedV2 approach"""
    
    # NVIDIA model processing (NVEmbedV2 approach)
    if model_name.startswith("nvidia"):
        print(f"Using NVEmbedV2 approach for {model_name}...")
        
        # Use NVEmbedV2's batch_encode approach
        if len(chunks) <= batch_size:
            # Single batch processing
            params = {
                "prompts": chunks,
                "max_length": 512,
                "instruction": "",
                "batch_size": batch_size
            }
            results = model.encode(**params)
        else:
            # Batch processing
            pbar = tqdm(total=len(chunks), desc="Generating embeddings")
            results = []
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                params = {
                    "prompts": batch_chunks,
                    "max_length": 512,
                    "instruction": "",
                    "batch_size": batch_size
                }
                batch_result = model.encode(**params)
                results.append(batch_result)
                pbar.update(batch_size)
            
            pbar.close()
            results = torch.cat(results, dim=0)
        
        # Process results
        if isinstance(results, torch.Tensor):
            results = results.cpu().numpy()
        
        # Normalization (NVEmbedV2 approach)
        results = (results.T / np.linalg.norm(results, axis=1)).T
        
        return results
    
    # Process existing models
    if hasattr(model, 'encode') and hasattr(model, 'tokenizer'):
        print(f"Using SentenceTransformer.encode() for {model_name}...")
        
        embeddings = model.encode(
            chunks, 
            batch_size=batch_size, 
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings
    
    # Keep existing batch processing logic
    if len(chunks) <= batch_size:
        batch = chunks
        tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
            "max_length": 512
        }
        if model_name.startswith("dpr"):
            tokenizer_kwargs["max_length"] = 512
        
        inputs = tokenizer(batch, **tokenizer_kwargs)
        if device is not None:
            inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = process_model_outputs(outputs, inputs, model_name)
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
        
        return embeddings
    else:
        # Batch processing
        pbar = tqdm(total=len(chunks), desc="Generating embeddings")
        results = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            tokenizer_kwargs = {
                "padding": True,
                "truncation": True,
                "return_tensors": "pt",
                "max_length": 512
            }
            if model_name.startswith("dpr"):
                tokenizer_kwargs["max_length"] = 512
            
            inputs = tokenizer(batch, **tokenizer_kwargs)
            if device is not None:
                inputs = inputs.to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_result = process_model_outputs(outputs, inputs, model_name)
                
                if isinstance(batch_result, torch.Tensor):
                    batch_result = batch_result.cpu().numpy()
                results.append(batch_result)
                pbar.update(batch_size)
        
        pbar.close()
        return np.concatenate(results, axis=0)

def process_model_outputs(outputs, inputs, model_name):
    """A helper function that consistently processes model outputs"""
    if model_name.startswith("dpr"):
        return outputs.pooler_output
    elif hasattr(outputs, 'last_hidden_state'):
        return mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
    elif isinstance(outputs, dict):
        if 'sentence_embedding' in outputs:
            return outputs['sentence_embedding']
        elif 'sentence_embeddings' in outputs:
            return outputs['sentence_embeddings']
        elif 'last_hidden_state' in outputs:
            return mean_pooling(outputs['last_hidden_state'], inputs['attention_mask'])
        else:
            raise ValueError(f"Unexpected model output format: {outputs.keys()}")
    else:
        raise ValueError(f"Unsupported model output type: {type(outputs)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/contriever", help="Embedding model name")
    parser.add_argument("--chunk_file", type=str, required=True, help="Chunk file path")
    args = parser.parse_args()

    # Load chunks
    chunks = load_chunks(args.chunk_file)
    print(f"Loaded {len(chunks)} chunks from {args.chunk_file}")

    # Generate embeddings
    tokenizer, model, device, batch_size = load_models(args.model_name)
    print("Generating embeddings...")
    
    # Adjust batch size for NVIDIA models
    if args.model_name.startswith("nvidia"):
        optimal_batch_size = min(batch_size, 16)  # Use smaller batch size for NVIDIA models
    else:
        optimal_batch_size = min(batch_size, 32)
    
    print(f"Using optimal batch size: {optimal_batch_size}")
    embeddings = generate_embeddings(chunks, tokenizer, model, device, optimal_batch_size, args.model_name)
    
    # Normalization (NVIDIA models are already normalized)
    if not args.model_name.startswith("nvidia"):
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Save result
    model_name_clean = args.model_name.replace("/", "_")
    if "10000" in args.chunk_file:
        output_file = args.chunk_file.replace("chunk_10000.jsonl", f"embedding_{model_name_clean}_10000.npy")
    elif "2000" in args.chunk_file:
        output_file = args.chunk_file.replace("chunk_2000.jsonl", f"embedding_{model_name_clean}_2000.npy")
    else:
        output_file = args.chunk_file.replace("chunk.jsonl", f"embedding_{model_name_clean}.npy")
    
    print(f"Saving {len(embeddings)} embeddings to {output_file}")
    np.save(output_file, embeddings)
    print("Done!")

if __name__ == "__main__":
    main()