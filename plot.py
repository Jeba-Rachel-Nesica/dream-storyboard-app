# plot_eval_compare.py
"""
Create SFT vs SFT+PPO comparison plots from eval_out/summary.csv.

Inputs (produced by eval_compare.py):
  eval_out/summary.csv  # has rows for tags: "SFT", "SFT+PPO"

Outputs (saved next to summary.csv):
  valence_shift_comparison.png
  agency_comparison.png
  bertscoreP_comparison.png
  wps_comparison.png
  controls_comparison.png
  summary_copy.csv       # convenience copy of the summary used
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def bar_with_error(df, mean_col, std_col, title, y_label, outfile):
    plt.figure(figsize=(6, 4))
    x = np.arange(len(df["tag"]))
    y = df[mean_col].values
    yerr = df[std_col].values if std_col in df.columns else None
    plt.bar(x, y, yerr=yerr, capsize=6)
    plt.xticks(x, df["tag"].tolist())
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()

def grouped_controls(df, outfile):
    # Plot rates for soften/helper/exit (0–1). One bar per tag per control.
    plt.figure(figsize=(7, 4))
    tags = df["tag"].tolist()
    x = np.arange(len(tags))
    width = 0.25
    plt.bar(x - width, df["has_soften_mean"].values, width, label="soften")
    plt.bar(x,          df["has_helper_mean"].values, width, label="helper")
    plt.bar(x + width,  df["has_exit_mean"].values,   width, label="exit")
    plt.xticks(x, tags)
    plt.ylim(0, 1)
    plt.ylabel("Rate (0–1)")
    plt.title("Control Markers Present (rate)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser("Plot SFT vs SFT+PPO comparison from summary.csv")
    ap.add_argument("--summary", default="eval_out/summary.csv",
                    help="Path to summary.csv produced by eval_compare.py")
    args = ap.parse_args()

    if not os.path.exists(args.summary):
        raise FileNotFoundError(f"Could not find {args.summary}. "
                                "Run eval_compare.py first to generate it.")

    outdir = os.path.dirname(os.path.abspath(args.summary))
    df = pd.read_csv(args.summary)

    # Keep consistent order: SFT first, then SFT+PPO (if present)
    order = ["SFT", "SFT+PPO"]
    df["tag"] = pd.Categorical(df["tag"], categories=order, ordered=True)
    df = df.sort_values("tag")

    # Save a copy of the summary we plotted from
    df.to_csv(os.path.join(outdir, "summary_copy.csv"), index=False)

    # Individual comparison plots (1 metric per figure)
    bar_with_error(df, "valence_shift_mean", "valence_shift_std",
                   "Valence Shift (↑ better)", "Valence shift",
                   os.path.join(outdir, "valence_shift_comparison.png"))

    bar_with_error(df, "agency_mean", "agency_std",
                   "Agency Markers (↑ better)", "Agency (term rate %)",
                   os.path.join(outdir, "agency_comparison.png"))

    bar_with_error(df, "bertscore_P_mean", "bertscore_P_std",
                   "Coherence vs Nightmare (BERTScore P)", "BERTScore P",
                   os.path.join(outdir, "bertscoreP_comparison.png"))

    bar_with_error(df, "words_per_sent_mean", "words_per_sent_std",
                   "Readability (Words per Sentence)", "Words per sentence",
                   os.path.join(outdir, "wps_comparison.png"))

    # Control markers grouped bar
    needed = {"has_soften_mean","has_helper_mean","has_exit_mean"}
    if needed.issubset(set(df.columns)):
        grouped_controls(df, os.path.join(outdir, "controls_comparison.png"))

    print(f"Saved plots to: {outdir}")
    print("Files:")
    for f in ["valence_shift_comparison.png",
              "agency_comparison.png",
              "bertscoreP_comparison.png",
              "wps_comparison.png",
              "controls_comparison.png",
              "summary_copy.csv"]:
        p = os.path.join(outdir, f)
        if os.path.exists(p):
            print(" -", p)

if __name__ == "__main__":
    main()
