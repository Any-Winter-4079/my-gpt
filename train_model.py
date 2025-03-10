import os
import time
import math
import torch
import numpy as np
import torch.nn as nn
from models.gpt import GPT
import torch.optim as optim
from utils import get_device
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import IterableDataset, DataLoader

# Configuration
CONFIG = {
    "vocab_size": 50257,
    "embed_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "mlp_hidden_dim": 3072,
    "max_seq_len": 1024,
    "dropout": 0.1,
    "batch_size": 8,
    "lr": 3e-4,
    "warmup_steps": 1000,
    "total_steps": 50000,
    "betas": (0.9, 0.95),
    "device": get_device(),
    "data_dir": "./data",  # Directory where .npy tokenized files are stored
}

# Iterable Dataset for streaming from disk
class NPZIterableDataset(IterableDataset):
    def __init__(self, data_dir, batch_size, seq_len):
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")])
        self.batch_size = batch_size
        self.seq_len = seq_len
    
    def __iter__(self):
        for file in self.files:
            tokens = np.load(file).astype(np.int64)  # Convert uint16 to int64 for PyTorch compatibility
            num_samples = len(tokens) - self.seq_len
            for i in range(0, num_samples, self.batch_size):
                batch = np.array([tokens[j : j + self.seq_len] for j in range(i, min(i + self.batch_size, num_samples))], dtype=np.int64)
                yield torch.tensor(batch, dtype=torch.long)

# Load dataset from preprocessed .npy shards
def load_data_from_disk(data_dir, batch_size, seq_len):
    dataset = NPZIterableDataset(data_dir, batch_size, seq_len)
    dataloader = DataLoader(dataset, batch_size=None)
    return dataloader

# Learning rate schedule with warmup and cosine decay
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

# Exclude LayerNorm and bias parameters from weight decay
def get_optimizer(model, lr, betas):
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.dim() == 1 or "ln" in name:
            no_decay_params.append(param)  # LayerNorm and biases
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
    
    if device.type == "cuda":
        torch.set_float32_matmul_precision('high')

    model = GPT(
        vocab_size=CONFIG["vocab_size"],
        embed_dim=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        mlp_hidden_dim=CONFIG["mlp_hidden_dim"],
        max_seq_len=CONFIG["max_seq_len"],
        dropout=CONFIG["dropout"],
    ).to(device)

    if device.type == "cuda":
        model = torch.compile(model)

    dataloader = load_data_from_disk(CONFIG["data_dir"], CONFIG["batch_size"], CONFIG["max_seq_len"])
    optimizer = get_optimizer(model, CONFIG["lr"], CONFIG["betas"])
    scheduler = get_lr_scheduler(optimizer, CONFIG["warmup_steps"], CONFIG["total_steps"])
    loss_fn = nn.CrossEntropyLoss()
    
    step = 0
    model.train()
    start_time = time.time()
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        batch = batch.to(device)
        input_ids = batch[:, :-1]
        target_ids = batch[:, 1:]
        
        # if device.type == "cuda":
        #     with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        #         logits = model(input_ids)
        #         loss = loss_fn(logits.reshape(-1, CONFIG["vocab_size"]), target_ids.reshape(-1))
        # else:
        #     logits = model(input_ids)
        #     loss = loss_fn(logits.reshape(-1, CONFIG["vocab_size"]), target_ids.reshape(-1))

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = loss_fn(logits.reshape(-1, CONFIG["vocab_size"]), target_ids.reshape(-1))

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        step += 1
        
        if step % 10 == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps" and hasattr(torch, 'mps'):
                torch.mps.synchronize()
            
            elapsed_time = time.time() - start_time
            print(f"Step {step}/{CONFIG['total_steps']}: Loss = {loss.item():.4f}, Time per 10 steps = {elapsed_time:.2f} sec")
            start_time = time.time()
        
        if step >= CONFIG["total_steps"]:
            break

if __name__ == "__main__":
    train()
