# debug_data.py
import json
from src.data_utils import PairDataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2-medium")
for t in ["<NIGHTMARE>", "<COMFORT>"]:
    if tok.convert_tokens_to_ids(t) == tok.unk_token_id:
        tok.add_tokens([t])

ds = PairDataset("data/final_train.jsonl", tok, ("<NIGHTMARE>", "<COMFORT>"))

# Check first example
sample = ds[0]
print(f"Input IDs length: {len(sample['input_ids'])}")
print(f"Labels: {sample['labels']}")
print(f"Non-ignored labels: {sum(1 for x in sample['labels'] if x != -100)}")
print(f"Decoded: {tok.decode(sample['input_ids'])}")