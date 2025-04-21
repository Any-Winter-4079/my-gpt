import os
import tiktoken
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from datasets import load_dataset

eot_token = 50256 # EOT token for GPT-2
num_proc_load_dataset = 8 # num workers

def load_tokenizer():
    """Load GPT-2 tokenizer using tiktoken"""
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
        print("Loaded GPT-2 tokenizer from tiktoken")
        return tokenizer
    except Exception as e:
        print("Could not load tokenizer from tiktoken")
        raise NotImplementedError("Please install the tiktoken library for GPT-2 tokenization") from e

# Initialize tokenizer in worker process
_worker_tokenizer = None
def init_worker():
    """Initialize global tokenizer for each worker process"""
    global _worker_tokenizer
    _worker_tokenizer = tiktoken.get_encoding("gpt2")

def tokenize_docs_batch(docs, eot_token):
    """Tokenize a batch of documents using the worker's tokenizer"""
    global _worker_tokenizer
    
    # Make sure we have a tokenizer
    if _worker_tokenizer is None:
        _worker_tokenizer = tiktoken.get_encoding("gpt2")
    
    all_tokens = []
    
    # Process each document in the batch
    for doc in docs:
        # Start with EOT token
        tokens = [eot_token]
        
        # Get text from document
        text = doc["text"] if "text" in doc else next(iter(doc.values()))
        
        # Tokenize if we have valid text
        if text and isinstance(text, str):
            doc_tokens = _worker_tokenizer.encode(text)
            tokens.extend(doc_tokens)
        
        all_tokens.append(tokens)
    
    return all_tokens

def prepare_webtext_data_mp(data_dir="./data", shard_size=100_000_000, num_workers=None, batch_size=500):
    """
    Download and tokenize OpenWebText dataset using optimized multiprocessing.
    
    Args:
        data_dir: Directory to save tokenized shards
        shard_size: Target size for each shard in tokens (e.g., 100M tokens)
        num_workers: Number of worker processes (default: CPU count - 1)
        batch_size: Number of documents to process in each batch
    
    Returns:
        Path to the data directory
    """
    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Using {num_workers} worker processes for tokenization with batch size {batch_size}")
    
    # Download the dataset
    print("Loading OpenWebText dataset...")
    try:
        webtext = load_dataset("openwebtext", split="train", trust_remote_code=True, num_proc=num_proc_load_dataset)
        print(f"Successfully loaded OpenWebText with {len(webtext)} documents")
    except Exception as e:
        raise RuntimeError(f"Error loading OpenWebText: {e}")
    
    # Initialize variables for sharding
    shard_index = 0
    shard_tokens = []  # Tokens in the current shard
    total_tokens = 0
    
    # Create progress bar for documents
    doc_pbar = tqdm(total=len(webtext), desc="[Overall] Documents processed", position=0)
    
    # Create progress bar for current shard
    token_pbar = tqdm(total=shard_size, desc=f"[Shard {shard_index}] Tokens collected", position=1, unit=" tokens")
    
    try:
        # Create a pool of workers with initialized tokenizers
        with mp.Pool(num_workers, initializer=init_worker) as pool:
            # Process documents in batches
            for batch_idx in range(0, len(webtext), batch_size):
                batch_end = min(batch_idx + batch_size, len(webtext))
                batch = [webtext[i] for i in range(batch_idx, batch_end)]
                
                # Divide batch into smaller chunks for each worker
                chunk_size = max(1, len(batch) // num_workers)
                doc_chunks = [batch[i:i+chunk_size] for i in range(0, len(batch), chunk_size)]
                
                # Process each chunk in parallel
                results = pool.map(partial(tokenize_docs_batch, eot_token=eot_token), doc_chunks)
                
                # Flatten the results: list of token lists (one per document)
                batch_tokens_list = [tokens for chunk_result in results for tokens in chunk_result]
                
                # Process each document's tokens
                for doc_tokens in batch_tokens_list:
                    start_idx = 0
                    while start_idx < len(doc_tokens):
                        # Determine how many tokens can fit in the current shard
                        remaining = shard_size - len(shard_tokens)
                        tokens_to_take = min(remaining, len(doc_tokens) - start_idx)
                        
                        # Add the tokens to the current shard
                        shard_tokens.extend(doc_tokens[start_idx:start_idx+tokens_to_take])
                        token_pbar.update(tokens_to_take)
                        start_idx += tokens_to_take
                        
                        # If the current shard is full, save it and start a new shard
                        if len(shard_tokens) == shard_size:
                            split = "val" if shard_index == 0 else "train"
                            filename = os.path.join(data_dir, f"webtext_{split}_{shard_index:06d}")
                            
                            # Save current shard using uint16
                            tokens_np = np.array(shard_tokens, dtype=np.uint16)
                            np.save(filename, tokens_np)
                            
                            # Close and print progress
                            token_pbar.close()
                            print(f"\nSaved {len(tokens_np)} tokens to {filename}.npy\n")
                            
                            shard_index += 1
                            total_tokens += len(tokens_np)
                            shard_tokens = []  # Reset for the new shard
                            
                            # Create new progress bar for the next shard
                            token_pbar = tqdm(total=shard_size, desc=f"[Shard {shard_index}] Tokens collected", 
                                              position=1, unit=" tokens")
                
                # Update the overall document progress bar for the whole batch
                doc_pbar.update(len(batch))
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current progress...")
    
    finally:
        # Save any remaining tokens as the final shard (may be smaller than shard_size)
        if shard_tokens:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(data_dir, f"webtext_{split}_{shard_index:06d}")
            tokens_np = np.array(shard_tokens, dtype=np.uint16)
            np.save(filename, tokens_np)
            
            try:
                token_pbar.close()
                doc_pbar.close()
            except Exception:
                pass
            
            print(f"\nSaved {len(tokens_np)} tokens to final shard {filename}.npy\n")
            total_tokens += len(tokens_np)
        else:
            try:
                token_pbar.close()
                doc_pbar.close()
            except Exception:
                pass
        
        print(f"Finished processing dataset. Created {shard_index + 1} shards with {total_tokens} total tokens.")
        return data_dir

def load_and_decode_shard(shard_path, max_tokens=None):
    """
    Load tokens from a shard file and decode them to text.
    
    Args:
        shard_path: Path to the .npy shard file
        max_tokens: Optional limit on how many tokens to load
        
    Returns:
        tokens: numpy array of tokens
        text: decoded text
    """
    # Load the tokenizer
    tokenizer = load_tokenizer()
    
    # Load the tokens from the shard
    print(f"Loading tokens from {shard_path}...")
    tokens = np.load(shard_path)
    
    # Limit tokens if requested
    if max_tokens is not None and max_tokens < len(tokens):
        print(f"Limiting to first {max_tokens} tokens out of {len(tokens)} total")
        tokens = tokens[:max_tokens]
    
    # Decode tokens to text
    print(f"Decoding {len(tokens)} tokens...")
    text = tokenizer.decode(tokens.tolist())
    
    print(f"Shard contains {len(tokens)} tokens")
    print(f"Sample text (first 5000 chars):\n{text[:5000]}...")
    
    return tokens, text

# Test the functions
if __name__ == "__main__":
    
    # Test tokenizer
    tokenizer = load_tokenizer()
    test_text = "Hello, this is a test for GPT-2 tokenization."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"Original: {test_text}")
    print(f"Tokenized: {tokens}")
    print(f"Decoded: {decoded}")
    
    # Process OpenWebText dataset with optimized multiprocessing
    data_dir = prepare_webtext_data_mp(batch_size=500)
    
    # After dataset processing, or if interrupted, test loading a shard
    shard_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    if shard_files:
        # Load and decode the first shard, but only the first 1000 tokens
        shard_path = os.path.join(data_dir, shard_files[0])
        tokens, text = load_and_decode_shard(shard_path, max_tokens=1000)
