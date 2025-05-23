import os
import time
import math
import torch
import tiktoken
import numpy as np
import torch.nn as nn
from models.gpt import GPT
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from utils import get_device, get_default_dtype
from torch.utils.data import IterableDataset, DataLoader

# Configuration
CONFIG = {
    "vocab_size": 50304,
    "embed_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "up_proj_factor": 4,
    "max_seq_len": 1024,
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 6e-4,
    "max_lr_ratio": 1.0,
    "min_lr_ratio": 0.1,
    "warmup_steps": 10,
    "total_steps": 50,
    "betas": (0.9, 0.95),
    "device": get_device(),
    "data_dir": "./data",
    "grad_clip": 1.0,
}

class NPZIterableDataset(IterableDataset):
    def __init__(self, data_dir, batch_size, seq_len):
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npy")
        ])
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __iter__(self):
        for file in self.files:
            # Load as uint16 and convert to int32 for PyTorch compatibility
            tokens = np.load(file).astype(np.int32)
            total_len = len(tokens)
            chunk_len = self.seq_len * self.batch_size

            for i in range(0, total_len - chunk_len + 1, chunk_len):
                chunk = tokens[i : i + chunk_len]
                batch = chunk.reshape(self.batch_size, self.seq_len)

                # Cast to torch.long (int64)
                yield torch.tensor(batch, dtype=torch.long)

# Load dataset from preprocessed .npy shards
def load_data_from_disk(data_dir, batch_size, seq_len):
    dataset = NPZIterableDataset(data_dir, batch_size, seq_len)
    dataloader = DataLoader(dataset, batch_size=None)
    return dataloader

# Learning rate schedule with warmup and cosine decay
def get_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio, max_lr_ratio):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps * max_lr_ratio
        elif step > total_steps:
            return min_lr_ratio
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        return min_lr_ratio + (max_lr_ratio - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)

# Exclude LayerNorm and bias parameters from weight decay
def get_optimizer(model, lr, betas):
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.dim() == 1 or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_grouped = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return optim.AdamW(optimizer_grouped, lr=lr, betas=betas)

# Initialize model
def train():
    device = CONFIG["device"]

    tokenizer = tiktoken.get_encoding("gpt2")
    
    if device.type == "cuda":
        torch.set_float32_matmul_precision('high')

    model = GPT(
        vocab_size=CONFIG["vocab_size"],
        embed_dim=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        up_proj_factor=CONFIG["up_proj_factor"],
        max_seq_len=CONFIG["max_seq_len"],
        dropout=CONFIG["dropout"],
    ).to(device)

    if device.type == "cuda":
        model = torch.compile(model)

    dataloader = load_data_from_disk(CONFIG["data_dir"],
                                     CONFIG["batch_size"],
                                     CONFIG["max_seq_len"])
    optimizer = get_optimizer(model,
                              CONFIG["lr"],
                              CONFIG["betas"])
    scheduler = get_lr_scheduler(optimizer,
                                 CONFIG["warmup_steps"],
                                 CONFIG["total_steps"],
                                 CONFIG["min_lr_ratio"],
                                 CONFIG["max_lr_ratio"])
    loss_fn = nn.CrossEntropyLoss()
    
    step = 0
    model.train()
    start_time = time.time()
    tokens_processed = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        batch = batch.to(device)
        # Print first 10 tokens and decoded text from the first 1–3 samples
        if False:
            print("Batch first tokens preview + decoded:")
            for i in range(min(3, batch.shape[0])):
                token_list = batch[i][:20].tolist()
                decoded = tokenizer.decode(token_list)
                print(f"  Sample {i} tokens: {token_list}")
                print(f"  Sample {i} text  : {decoded!r}")

        input_ids = batch[:, :-1]
        target_ids = batch[:, 1:]

        # mps
        # logits = model(input_ids)
        # loss = loss_fn(logits.reshape(-1, CONFIG["vocab_size"]), target_ids.reshape(-1))
        
        # cuda
        with torch.autocast(device_type=device.type, dtype=get_default_dtype()):
            logits = model(input_ids)
            loss = loss_fn(logits.reshape(-1, CONFIG["vocab_size"]), target_ids.reshape(-1))

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        current_lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        
        step += 1
        tokens_processed += batch.numel()
        
        if step % 1 == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps" and hasattr(torch, 'mps'):
                torch.mps.synchronize()
            
            elapsed_time = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed_time
            print(
                f"Step {step}/{CONFIG['total_steps']}: "
                f"Loss={loss.item():.6f}, "
                f"LR={current_lr:.8f}, "
                f"Grad_norm={grad_norm:.4f}, "
                f"TPS={tokens_per_sec:.2f}, "
                f"Time={1000 * elapsed_time:.2f}ms"
            )
            start_time = time.time()
            tokens_processed = 0
        
        if step >= CONFIG["total_steps"]:
            break

if __name__ == "__main__":
    train()