"""make_profile_fig.py — Fig.1: example divergence profiles (detected/missed/negative), v2-compatible.
Run from 02_hebseam_methods:
    python make_profile_fig.py <dataset_dir> results\scores_proposed.json results\stylometric_vocab.json <threshold> figs
Threshold = proposed method's calibrated threshold (see comparison.json, field "threshold").
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common.detection import fit_standardizer, standardize
from methods.run_proposed import load_verses, feature_matrix
import config as C

DS = Path(sys.argv[1])
SCORES = Path(sys.argv[2]); VOCAB = Path(sys.argv[3])
THR = float(sys.argv[4]); OUT = Path(sys.argv[5] if len(sys.argv) > 5 else "figs")
OUT.mkdir(exist_ok=True)

scores = json.load(open(SCORES, encoding="utf-8"))["scores"]
vocab = json.load(open(VOCAB, encoding="utf-8"))
truth = {p.stem.split("-")[-1]: json.load(open(p, encoding="utf-8"))
         for p in (DS / "truth").glob("truth-*.json")}

def divergence_profile(pid):
    verses = load_verses(DS, pid)
    X = feature_matrix(verses, vocab)
    Z = standardize(X, fit_standardizer(X))
    sims = (Z[:-1] * Z[1:]).sum(1) / (np.linalg.norm(Z[:-1], axis=1) * np.linalg.norm(Z[1:], axis=1) + 1e-9)
    return 1.0 - sims

def pick(pred):
    for pid in sorted(scores):
        t, s = truth[pid], scores[pid]
        if pred(t, s): return pid, t, s

det = pick(lambda t, s: t["label"] == 1 and t["contrast"] == "mishnah" and t["block_size"] == 160 and s["score"] >= THR)
mis = pick(lambda t, s: t["label"] == 1 and t["contrast"] == "exodus" and t["block_size"] == 20 and s["score"] < THR)
neg = pick(lambda t, s: t["label"] == 0 and t["case_type"] == "pure" and s["score"] < THR)

fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.6))
for ax, (pid, t, s), title in zip(axes, [det, mis, neg], [
        "(a) Detected: Mishnah insert, B=160 (score %.2f > %.2f)" % (det[2]["score"], THR),
        "(b) Missed: Exodus insert, B=20 (score %.2f < %.2f)" % (mis[2]["score"], THR),
        "(c) Negative: pure Genesis (score %.2f < %.2f)" % (neg[2]["score"], THR)]):
    D = divergence_profile(pid)
    ax.plot(D, lw=1.0, color="#1E5FA8")
    if t["label"] == 1 and t.get("boundaries"):
        b0, b1 = min(t["boundaries"]), max(t["boundaries"])
        lo = max(0, (b0 - C.WINDOW_LEN) // C.STRIDE)
        hi = min(len(D) - 1, b1 // C.STRIDE)
        ax.axvspan(lo, hi, color="#c0392b", alpha=0.18, label="true seam band")
        ax.legend(fontsize=7, loc="upper right")
    ax.set_title(title, fontsize=9); ax.set_ylabel("divergence", fontsize=8)
    ax.tick_params(labelsize=7)
axes[-1].set_xlabel("window transition index", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "F_profiles.png", dpi=170)
print("wrote", OUT / "F_profiles.png")
