# generate_script.py
"""
Final-step inference for Nightmare ➜ Comfort generation.

Loads a PPO-trained comfort model (AutoModelForCausalLMWithValueHead),
sets proper PAD + left padding, and generates comforting rewrites.

Usage examples (PowerShell):
  # 1) Single prompt
  python generate_script.py `
    --model_dir checkpoints\ppo_comfort\final_model `
    --input "I'm falling from a tall building"

  # 2) Batch JSONL (each line: {"nightmare": "..."} or a raw string line)
  python generate_script.py `
    --model_dir checkpoints\ppo_comfort\final_model `
    --infile data\final_test.jsonl `
    --outfile outputs\comfort_results.jsonl `
    --pretty
"""
import argparse, json, os, sys, re, random
from typing import List

import torch
from transformers import GPT2Tokenizer
from trl import AutoModelForCausalLMWithValueHead


def load_model(model_dir: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] device={device}")
    print(f"[load] model_dir={model_dir}")

    tok = GPT2Tokenizer.from_pretrained(model_dir, use_fast=True)

    # Ensure a true PAD token + left padding
    if (tok.pad_token is None) or (tok.pad_token_id == tok.eos_token_id):
        tok.add_special_tokens({"pad_token": "[PAD]"})
    tok.padding_side = "left"

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_dir)
    # Resize embeddings on the wrapped base model (value-head wrapper has no .resize_token_embeddings)
    base = getattr(model, "pretrained_model", None) or getattr(model, "transformer", None)
    if base is not None:
        base.resize_token_embeddings(len(tok))

    model = model.to(device)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.use_cache = True  # OK for inference

    return tok, model, device


def finish_sentence(text: str) -> str:
    # Trim to last sentence end if one exists; otherwise add a period.
    for end in (".", "!", "?"):
        idx = text.rfind(end)
        if idx != -1:
            return text[: idx + 1].strip()
    text = text.strip()
    return text if not text else text + "."


# Micro de-templating for variety (optional, lightweight)
REPLACE_MAP = {
    r"\byou take a deep breath\b": [
        "you breathe steadily",
        "your breath settles",
        "you notice your breath slowing",
    ],
    r"\byou walk away with confidence\b": [
        "you feel steadier as you move on",
        "you leave the fear behind you",
        "you step forward with ease",
    ],
    r"\byou step into open light\b": [
        "you step into gentle light",
        "you find yourself in open light",
        "you move into a warm light",
    ],
}
def diversify(text: str) -> str:
    t = text
    for pat, choices in REPLACE_MAP.items():
        if re.search(pat, t, flags=re.I):
            t = re.sub(pat, random.choice(choices), t, flags=re.I)
    return t


GEN_KWARGS_DEFAULT = dict(
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    top_k=0,                   # nucleus-only sampling
    repetition_penalty=1.15,   # reduce templating
    no_repeat_ngram_size=3,    # avoid repeated 3-grams
    max_new_tokens=48,         # concise, focused comfort
)


def make_queries(nightmares: List[str]) -> List[str]:
    return [f"Nightmare: {n}\nComfort:" for n in nightmares]


def generate_batch(tok: GPT2Tokenizer, model, device: str, nightmares: List[str], gen_kwargs=None) -> List[str]:
    gen_kwargs = gen_kwargs or {}
    queries = make_queries(nightmares)

    enc = tok(queries, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Merge defaults with any overrides (e.g., max_new_tokens from CLI)
    gk = dict(GEN_KWARGS_DEFAULT)
    gk.update(gen_kwargs)

    # TRL's ValueHead delegates generate() to underlying base model
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        **gk,
    )  # [B, seq+new]

    # Slice out only the generated tail for each row
    B = input_ids.size(0)
    res_texts: List[str] = []
    for i in range(B):
        prompt_len = (attention_mask[i] > 0).nonzero(as_tuple=True)[0][-1].item() + 1
        # tail tokens after prompt_len
        tail = outputs[i, prompt_len:]
        text = tok.decode(tail, skip_special_tokens=True).strip()
        # If the model echoed "Comfort:", remove it
        if text.lower().startswith("comfort:"):
            text = text[len("comfort:"):].strip()
        text = finish_sentence(text)
        text = diversify(text)
        res_texts.append(text)
    return res_texts


def read_infile(path: str) -> List[str]:
    nightmares = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "nightmare" in obj:
                    nightmares.append(str(obj["nightmare"]))
                elif isinstance(obj, str):
                    nightmares.append(obj)
                else:
                    nightmares.append(line)  # fallback
            except json.JSONDecodeError:
                nightmares.append(line)
    return nightmares


def write_outfile(path: str, nightmares: List[str], comforts: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for n, c in zip(nightmares, comforts):
            rec = {"nightmare": n, "comfort": c}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[write] saved {len(comforts)} lines -> {path}")


def main():
    ap = argparse.ArgumentParser(description="Nightmare → Comfort inference (PPO final step)")
    ap.add_argument("--model_dir", type=str, required=True, help="Path to PPO comfort model (e.g., checkpoints\\ppo_comfort\\final_model)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=str, help="Single nightmare string")
    g.add_argument("--infile", type=str, help="Path to JSONL/TXT (one nightmare per line or JSON with {'nightmare': ...})")
    ap.add_argument("--outfile", type=str, help="Where to write JSONL results for batch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--pretty", action="store_true", help="Print human-readable pairs to stdout")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tok, model, device = load_model(args.model_dir)

    if args.input is not None:
        comforts = generate_batch(
            tok, model, device,
            [args.input],
            {"max_new_tokens": args.max_new_tokens},
        )
        print("\n==============================")
        print("NIGHTMARE:\n", args.input)
        print("\nCOMFORT:\n", comforts[0])
        print("==============================\n")
    else:
        nightmares = read_infile(args.infile)
        if not nightmares:
            print(f"[error] No nightmares found in {args.infile}", file=sys.stderr)
            sys.exit(1)
        comforts = generate_batch(
            tok, model, device,
            nightmares,
            {"max_new_tokens": args.max_new_tokens},
        )
        if args.outfile:
            write_outfile(args.outfile, nightmares, comforts)

        if args.pretty:
            for i, (n, c) in enumerate(zip(nightmares, comforts), 1):
                print(f"\n#{i}\nNightmare: {n}\nComfort : {c}")
        elif not args.outfile:
            # Print to stdout if no outfile provided
            for n, c in zip(nightmares, comforts):
                print(json.dumps({"nightmare": n, "comfort": c}, ensure_ascii=False))


if __name__ == "__main__":
    main()
