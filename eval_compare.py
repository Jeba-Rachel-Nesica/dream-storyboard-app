# eval_sft_vs_ppo_plots.py
"""
Compare SFT vs SFT+PPO on held-out nightmares using:
- Valence shift (VADER compound comfort - nightmare)  (↑)
- Agency term rate per 100 tokens                      (↑)
- BERTScore Precision (comfort vs nightmare)           (↑ or ≈)
- rep-3 rate (3-gram repetition)                       (↓)
- Words per sentence                                   (readability)

Outputs:
  <outdir>/sft_detailed.csv
  <outdir>/ppo_detailed.csv
  <outdir>/combined_detailed.csv
  <outdir>/summary.csv
  <outdir>/bar_metrics.png            # grouped bars with error bars
  <outdir>/delta_paired.png           # paired deltas (PPO - SFT) per metric (optional scatter)

Notes:
- Uses greedy decoding to keep SFT vs PPO deterministic and fair.
- Falls back from roberta-large → roberta-base if BERTScore download times out.
"""

import os, json, argparse, re, statistics, random
from typing import List, Dict, Tuple

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

# sentiment (valence)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# BERTScore
from bert_score import score as bertscore_score

import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
AGENCY_TERMS = set("""
can choose decide notice pause breathe step open exit safe steady control calm
""".split())

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[a-zA-Z']+")

def sent_len_words(text: str) -> float:
    sents = [s.strip() for s in SENT_SPLIT.split(text.strip()) if s.strip()]
    if not sents:
        return float(len(text.split()))
    counts = [len(WORD_RE.findall(s)) for s in sents]
    return sum(counts)/len(counts)

def agency_rate(text: str) -> float:
    toks = [t.lower() for t in WORD_RE.findall(text)]
    if not toks: return 0.0
    hits = sum(1 for t in toks if t in AGENCY_TERMS)
    return 100.0 * hits / len(toks)  # per 100 tokens

def rep3_rate(text: str) -> float:
    toks = text.split()
    if len(toks) < 6:
        return 0.0
    triples = [tuple(toks[i:i+3]) for i in range(len(toks)-2)]
    if not triples: return 0.0
    return 1.0 - (len(set(triples)) / len(triples))  # 0=good, 1=bad

def load_jsonl_or_txt(path: str, limit: int = None) -> List[str]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "nightmare" in obj:
                    items.append(str(obj["nightmare"]))
                elif isinstance(obj, str):
                    items.append(obj)
            except json.JSONDecodeError:
                items.append(line)
            if limit and len(items) >= limit:
                break
    return items

def ensure_tokenizer(model_dir: str) -> GPT2Tokenizer:
    tok = GPT2Tokenizer.from_pretrained(model_dir, use_fast=True)
    if (tok.pad_token is None) or (tok.pad_token_id == tok.eos_token_id):
        tok.add_special_tokens({"pad_token": "[PAD]"})
    tok.padding_side = "left"
    return tok

def load_model(model_dir: str, device: str):
    tok = ensure_tokenizer(model_dir)
    # GPT2LMHeadModel can load PPO checkpoints; it will ignore v_head weights (expected warning)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tok))
    model.to(device)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.eval()
    return tok, model

@torch.no_grad()
def generate_batch(tok, model, device, prompts: List[str], max_new_tokens=64) -> List[str]:
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    out = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        do_sample=False,          # greedy for fairness/determinism
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        no_repeat_ngram_size=3,
    )
    results = []
    for i in range(out.size(0)):
        # length of non-padding prompt tokens for row i
        prompt_len = (enc["input_ids"][i] != tok.pad_token_id).sum().item()
        tail = out[i, prompt_len:]
        text = tok.decode(tail, skip_special_tokens=True).strip()
        text = re.sub(r"^\s*comfort:\s*", "", text, flags=re.I).strip()
        results.append(text)
    return results

def evaluate_set(nightmares: List[str], comforts: List[str], analyzer) -> Dict[str, List[float]]:
    valence_n = [analyzer.polarity_scores(n)["compound"] for n in nightmares]
    valence_c = [analyzer.polarity_scores(c)["compound"] for c in comforts]
    valence_shift = [c - n for n, c in zip(valence_n, valence_c)]
    agency = [agency_rate(c) for c in comforts]
    rep3 = [rep3_rate(c) for c in comforts]
    wps = [sent_len_words(c) for c in comforts]
    return dict(
        valence_shift=valence_shift,
        agency=agency,
        rep3=rep3,
        words_per_sent=wps,
        valence_n=valence_n,
        valence_c=valence_c,
    )

def summarize(tag: str, metrics: Dict[str, List[float]]) -> Dict[str, float]:
    out = {"tag": tag}
    for k, arr in metrics.items():
        if not isinstance(arr, list):
            continue
        if len(arr) == 0:
            out[f"{k}_mean"] = 0.0
            out[f"{k}_std"] = 0.0
        else:
            out[f"{k}_mean"] = statistics.mean(arr)
            out[f"{k}_std"]  = statistics.pstdev(arr)
    return out

def plot_bars(summary_df: pd.DataFrame, outpath: str):
    # Expect two rows: SFT and SFT+PPO
    metrics = ["valence_shift", "agency", "bertscore_P", "words_per_sent", "rep3"]
    labels = ["Valence↑", "Agency↑", "BERTScore-P↑", "Words/Sent", "rep-3↓"]

    means = []
    stds  = []
    for tag in ["SFT", "SFT+PPO"]:
        row = summary_df[summary_df["tag"] == tag].iloc[0]
        means.append([row[f"{m}_mean"] for m in metrics])
        stds.append([row[f"{m}_std"]  for m in metrics])

    # Grouped bars with error bars
    x = range(len(metrics))
    width = 0.35

    fig = plt.figure(figsize=(10, 5))
    plt.bar([i - width/2 for i in x], means[0], width, yerr=stds[0], label="SFT", capsize=4)
    plt.bar([i + width/2 for i in x], means[1], width, yerr=stds[1], label="SFT+PPO", capsize=4)
    plt.xticks(list(x), labels, rotation=0)
    plt.ylabel("Score")
    plt.title("SFT vs SFT+PPO — Evaluation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_deltas(sft_df: pd.DataFrame, ppo_df: pd.DataFrame, outpath: str):
    # Optional: scatter of per-example deltas for key metrics
    metrics = ["valence_shift", "agency", "bertscore_P"]
    labels = ["Δ Valence", "Δ Agency", "Δ BERT-P"]
    fig = plt.figure(figsize=(9, 3))
    for i, (m, lab) in enumerate(zip(metrics, labels), 1):
        ax = plt.subplot(1, 3, i)
        delta = ppo_df[m].values - sft_df[m].values
        ax.scatter(range(len(delta)), delta, s=8)
        ax.axhline(0.0)
        ax.set_title(lab)
        ax.set_xlabel("Example")
        ax.set_ylabel("PPO − SFT")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Compare SFT vs SFT+PPO (with plots)")
    ap.add_argument("--sft_model", required=True, help="Path to SFT model dir")
    ap.add_argument("--ppo_model", required=True, help="Path to PPO model dir")
    ap.add_argument("--data", required=True, help="Path to JSONL/TXT (each line has {'nightmare': ...} or raw)")
    ap.add_argument("--outdir", default="eval_out")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of examples")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bert_model", default="roberta-large", help="Backbone for BERTScore")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    nightmares = load_jsonl_or_txt(args.data, limit=args.limit or None)
    print(f"[info] examples: {len(nightmares)}  device: {device}")

    prompts = [f"Nightmare: {n}\nComfort:" for n in nightmares]

    print(f"[load] SFT: {args.sft_model}")
    tok_sft, model_sft = load_model(args.sft_model, device)

    print(f"[load] PPO: {args.ppo_model}")
    tok_ppo, model_ppo = load_model(args.ppo_model, device)

    print("[gen] SFT outputs...")
    sft_out = []
    for i in tqdm(range(0, len(prompts), 16)):
        sft_out += generate_batch(tok_sft, model_sft, device, prompts[i:i+16], args.max_new_tokens)

    print("[gen] PPO outputs...")
    ppo_out = []
    for i in tqdm(range(0, len(prompts), 16)):
        ppo_out += generate_batch(tok_ppo, model_ppo, device, prompts[i:i+16], args.max_new_tokens)

    analyzer = SentimentIntensityAnalyzer()
    sft_metrics = evaluate_set(nightmares, sft_out, analyzer)
    ppo_metrics = evaluate_set(nightmares, ppo_out, analyzer)

    print(f"[bert] Computing BERTScore-P with {args.bert_model} ...")
    try:
        P_sft, _, _ = bertscore_score(sft_out, nightmares, model_type=args.bert_model, lang="en")
        P_ppo, _, _ = bertscore_score(ppo_out, nightmares, model_type=args.bert_model, lang="en")
    except Exception as e:
        print("[bert] Falling back to roberta-base due to error:", e)
        P_sft, _, _ = bertscore_score(sft_out, nightmares, model_type="roberta-base", lang="en")
        P_ppo, _, _ = bertscore_score(ppo_out, nightmares, model_type="roberta-base", lang="en")

    sft_metrics["bertscore_P"] = [p.item() for p in P_sft]
    ppo_metrics["bertscore_P"] = [p.item() for p in P_ppo]

    # Save detailed per-example CSVs
    sft_rows, ppo_rows = [], []
    for i, n in enumerate(nightmares):
        sft_rows.append({
            "idx": i, "nightmare": n, "comfort": sft_out[i],
            "valence_shift": sft_metrics["valence_shift"][i],
            "agency": sft_metrics["agency"][i],
            "rep3": sft_metrics["rep3"][i],
            "words_per_sent": sft_metrics["words_per_sent"][i],
            "bertscore_P": sft_metrics["bertscore_P"][i],
        })
        ppo_rows.append({
            "idx": i, "nightmare": n, "comfort": ppo_out[i],
            "valence_shift": ppo_metrics["valence_shift"][i],
            "agency": ppo_metrics["agency"][i],
            "rep3": ppo_metrics["rep3"][i],
            "words_per_sent": ppo_metrics["words_per_sent"][i],
            "bertscore_P": ppo_metrics["bertscore_P"][i],
        })

    sft_df = pd.DataFrame(sft_rows)
    ppo_df = pd.DataFrame(ppo_rows)
    sft_csv = os.path.join(args.outdir, "sft_detailed.csv")
    ppo_csv = os.path.join(args.outdir, "ppo_detailed.csv")
    sft_df.to_csv(sft_csv, index=False)
    ppo_df.to_csv(ppo_csv, index=False)

    comb = pd.concat([sft_df.assign(tag="SFT"),
                      ppo_df.assign(tag="SFT+PPO")], ignore_index=True)
    comb_csv = os.path.join(args.outdir, "combined_detailed.csv")
    comb.to_csv(comb_csv, index=False)

    sft_sum = summarize("SFT", sft_metrics)
    ppo_sum = summarize("SFT+PPO", ppo_metrics)
    summary = pd.DataFrame([sft_sum, ppo_sum])
    summary_csv = os.path.join(args.outdir, "summary.csv")
    summary.to_csv(summary_csv, index=False)

    print("\n=== SUMMARY (means ± sd) ===\n")
    for row in [sft_sum, ppo_sum]:
        print(f"[{row['tag']}]")
        for key in ("valence_shift", "agency", "rep3", "bertscore_P", "words_per_sent"):
            print(f"{key:14s}: {row[key+'_mean']:.3f} ± {row[key+'_std']:.3f}")
        print()

    # Significance (optional)
    try:
        from scipy.stats import wilcoxon
        ws_v = wilcoxon(ppo_df["valence_shift"].values, sft_df["valence_shift"].values, zero_method="zsplit")
        ws_a = wilcoxon(ppo_df["agency"].values,        sft_df["agency"].values,        zero_method="zsplit")
        print("Wilcoxon (Valence shift):", ws_v)
        print("Wilcoxon (Agency)      :", ws_a)
    except Exception as e:
        print("(Wilcoxon skipped; install scipy if you want significance.)", e)

    # ---- PLOTS ----
    bar_png = os.path.join(args.outdir, "bar_metrics.png")
    plot_bars(summary, bar_png)

    delta_png = os.path.join(args.outdir, "delta_paired.png")
    try:
        plot_deltas(sft_df, ppo_df, delta_png)
    except Exception as _:
        pass

    print("\nSaved:")
    print(" -", sft_csv)
    print(" -", ppo_csv)
    print(" -", comb_csv)
    print(" -", summary_csv)
    print(" -", bar_png)
    print(" -", delta_png)

if __name__ == "__main__":
    main()
