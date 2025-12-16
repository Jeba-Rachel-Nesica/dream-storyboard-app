#!/usr/bin/env python3
"""
Comfort Agent Training with GPT2-Large
Simple, reliable version
"""

import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdamW,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from src.data_utils import PairDataset, set_seed


def ensure_pad(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def collate_pad(batch, pad_id: int):
    lens = [b["input_ids"].size(0) for b in batch]
    max_len = max(lens)

    def pad_tensor(t: torch.Tensor, value: int):
        if t.size(0) == max_len:
            return t
        pad = torch.full((max_len - t.size(0),), value, dtype=t.dtype)
        return torch.cat([t, pad], dim=0)

    input_ids = []
    attention_mask = []
    labels = []

    for b in batch:
        ids = pad_tensor(b["input_ids"], pad_id)
        attn = pad_tensor(b["attention_mask"], 0)
        labs = pad_tensor(b["labels"], pad_id).clone()
        labs[ids == pad_id] = -100
        
        input_ids.append(ids)
        attention_mask.append(attn)
        labels.append(labs)

    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "labels": torch.stack(labels, dim=0)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    base_model = cfg["base_model"]
    print(f"Loading tokenizer from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    special_tokens = {"additional_special_tokens": ["<NIGHTMARE>", "<COMFORT>"]}
    tokenizer.add_special_tokens(special_tokens)
    tokenizer = ensure_pad(tokenizer)
    print(f"Vocabulary size: {len(tokenizer)}")

    train_file = cfg["data"]["pairs_train"]
    val_file = cfg["data"]["pairs_val"]
    
    print(f"Loading training data from {train_file}")
    train_ds = PairDataset(train_file, tokenizer, cfg["sft"]["max_len"])
    print(f"Training examples: {len(train_ds)}")
    
    print(f"Loading validation data from {val_file}")
    val_ds = PairDataset(val_file, tokenizer, cfg["sft"]["max_len"])
    print(f"Validation examples: {len(val_ds)}")

    batch_size = cfg["sft"]["batch_size"]
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_pad(batch, tokenizer.pad_token_id)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_pad(batch, tokenizer.pad_token_id)
    )

    print(f"Loading model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.resize_token_embeddings(len(tokenizer))
    
    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    model = model.to(device)

    epochs = cfg["sft"]["epochs"]
    lr = float(cfg["sft"]["lr"])
    
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    print(f"Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")

    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Comfort {epoch+1}/{epochs}")
        
        for batch in train_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} train loss: {avg_train_loss:.4f}")
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Val {epoch+1}/{epochs}")
            for batch in val_pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                val_pbar.set_postfix({"val_loss": f"{outputs.loss.item():.4f}"})
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} val loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"✓ New best: {best_val_loss:.4f}")

    print("\n" + "=" * 70)
    print("MERGING AND SAVING")
    print("=" * 70)
    
    model = model.merge_and_unload()
    
    output_dir = cfg["checkpoints"]["sft_comfort"]
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✓ Saved to: {output_dir}")
    print(f"✓ Best val loss: {best_val_loss:.4f}")
    print("\nNext: python train_fear.py")


if __name__ == "__main__":
    main()