"""
Data utilities for loading and processing datasets
"""

import json
import random
import torch
from torch.utils.data import Dataset


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PairDataset(Dataset):
    """
    Dataset for nightmare-comfort pairs.
    Used for training the Comfort agent.
    """
    
    def __init__(self, jsonl_path: str, tokenizer, max_length: int, mode: str = "comfort"):
        """
        Args:
            jsonl_path: Path to JSONL file with nightmare-comfort pairs
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            mode: "comfort" or "fear"
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.data = []
        
        # Load data
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        nightmare = item["nightmare"]
        
        # Handle different field names - YOUR DATA USES "comfort_rewrite"
        comfort = item.get("comfort_rewrite") or item.get("comfort") or item.get("response", "")
        
        # Build input sequence
        if self.mode == "comfort":
            text = f"<NIGHTMARE> {nightmare}\n<COMFORT> {comfort}<|endoftext|>"
            role_token = "<COMFORT>"
        else:
            # For other modes
            text = f"<NIGHTMARE> {nightmare}\n<FEAR> {comfort}<|endoftext|>"
            role_token = "<FEAR>"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # Create labels: -100 for everything before the role token
        labels = input_ids.clone()
        
        # Find where the role token starts
        role_token_id = self.tokenizer.encode(role_token, add_special_tokens=False)[0]
        
        # Mask everything before (and including) the role token
        for i in range(len(labels)):
            if input_ids[i] == role_token_id:
                # Found role token, mask everything up to here
                labels[:i+1] = -100
                break
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class NightmareDataset(Dataset):
    """
    Dataset for nightmares only.
    Used for training the Fear agent.
    """
    
    def __init__(self, txt_path: str, tokenizer, max_length: int):
        """
        Args:
            txt_path: Path to text file with one nightmare per line
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.nightmares = []
        
        # Load nightmares
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.nightmares.append(line)
    
    def __len__(self):
        return len(self.nightmares)
    
    def __getitem__(self, idx):
        nightmare = self.nightmares[idx]
        
        # Build input sequence: nightmare -> shortened fear version
        # For training, we use the nightmare as both input and target
        # The model learns to generate concise, fearful versions
        text = f"<NIGHTMARE> {nightmare}\n<FEAR> {nightmare}<|endoftext|>"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # Create labels: -100 for everything before <FEAR>
        labels = input_ids.clone()
        
        # Find <FEAR> token
        fear_token_id = self.tokenizer.encode("<FEAR>", add_special_tokens=False)[0]
        
        # Mask everything before (and including) <FEAR>
        for i in range(len(labels)):
            if input_ids[i] == fear_token_id:
                labels[:i+1] = -100
                break
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }