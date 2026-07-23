"""make_comparison_figs.py — Fig.2 (six-method ROC) + Fig.3 (dual frontier) from comparison.json
Usage:  python make_comparison_figs.py <path/to/comparison.json> <out_dir>"""
import json, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

cmp_path = sys.argv[1] if len(sys.argv) > 1 else "results/comparison.json"
out = sys.argv[2] if len(sys.argv) > 2 else "figs"
os.makedirs(out, exist_ok=True)
rows = json.load(open(cmp_path, encoding="utf-8"))
by = {r["method"]: r for r in rows}
nice = {"proposed_interpretable_mean_shift": ("Proposed (interpretable)", "#c0392b", 2.6),
        "neural_multilingual_minilm_mean_shift": ("MiniLM embeddings", "#3F9E4D", 2.0),
        "transformer_dictabert_mean_shift": ("DictaBERT", "#1E5FA8", 1.4),
        "transformer_xlmr_mean_shift": ("XLM-R", "#6B5BA5", 1.2),
        "classical_tfidf_binseg_rbf": ("Classical (Binseg-RBF)", "#8C97A6", 1.2),
        "llm_perplexity_mgpt_mean_shift": ("mGPT perplexity", "#E08A2B", 1.2)}

# Fig 2 — ROC
plt.figure(figsize=(4.4, 4.1))
for m, (lab, col, lw) in nice.items():
    if m not in by: continue
    r = by[m]; xs = [p[0] for p in r["roc"]]; ys = [p[1] for p in r["roc"]]
    plt.plot(xs, ys, color=col, lw=lw, label="%s (%.2f)" % (lab, r["test_auc"]))
plt.plot([0, 1], [0, 1], "k--", lw=0.7)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("Six-method ROC (test split)"); plt.legend(fontsize=6.6, loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(out, "F_roc6.png"), dpi=170); plt.close()

# Fig 3 — dual frontier (the two calibration-stable methods)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, m in zip(axes, ["proposed_interpretable_mean_shift", "neural_multilingual_minilm_mean_shift"]):
    fr = by[m]["frontier"]
    G = np.array([[fr["%s_%d" % (c, B)] for B in (20, 40, 80, 160)]
                  for c in ("exodus", "psalms", "mishnah")])
    im = ax.imshow(G, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["20", "40", "80", "160"], fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels(["Exodus", "Psalms", "Mishnah"], fontsize=8)
    for i in range(3):
        for j in range(4):
            ax.text(j, i, "%.2f" % G[i, j], ha="center", va="center", fontsize=8)
    ax.set_title(nice[m][0], fontsize=9); ax.set_xlabel("block size (verses)", fontsize=8)
fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
plt.savefig(os.path.join(out, "F_frontier2.png"), dpi=170, bbox_inches="tight"); plt.close()
print("wrote", out + "/F_roc6.png", out + "/F_frontier2.png")
