from __future__ import annotations
"""
run_llm_perplexity.py — GENERATIVE LLM arm (option 2, per supervisor's 'GPT/LLaMA').
Score = perplexity-surprise seam statistic from a causal multilingual LM (mGPT).
Per window: mean token NLL. Seam score = mean_shift over the 1-D NLL profile
(z-scored), same W as other arms -> comparable.
Usage (Colab GPU):
  python methods/run_llm_perplexity.py <dataset> results/scores_llm_perplexity.json --device cuda
Model default ai-forever/mGPT (1.3B, supports Hebrew). ~30-45 min on T4.
"""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("USE_TF", "0")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.detection import mean_shift_score, split_to_verse
from common.io_utils import problem_ids, write_json
import config as C


def load_lines(dataset: Path, pid: str) -> list[str]:
    path = dataset / "problems" / f"problem-{pid}.txt"
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def make_windows(lines: list[str]) -> list[str]:
    return [" ".join(lines[i:i + C.WINDOW_LEN])
            for i in range(0, len(lines) - C.WINDOW_LEN + 1, C.STRIDE)]


def window_nll(texts, tokenizer, model, device, batch_size, max_length):
    import torch
    vals = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            batch = texts[s:s + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
            out = model(**enc)
            logits = out.logits[:, :-1, :]
            labels = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].float()
            nll = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), labels, reduction="none")
            per_win = (nll * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vals.extend(per_win.cpu().tolist())
    return np.asarray(vals, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Causal-LM perplexity seam baseline")
    ap.add_argument("dataset"); ap.add_argument("output")
    ap.add_argument("--model", default="ai-forever/mGPT")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else \
             ("cpu" if args.device == "auto" else args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)

    dataset = Path(args.dataset)
    scores: dict[str, dict] = {}
    ids = problem_ids(dataset)
    for i, pid in enumerate(ids, start=1):
        texts = make_windows(load_lines(dataset, pid))
        nll = window_nll(texts, tok, model, device, args.batch_size, args.max_length)
        z = (nll - nll.mean()) / (nll.std() + 1e-9)
        # 1-D profile: cosine mean-shift degenerates (always 0/2); use |mean diff|
        W = C.MEAN_SHIFT_CONTEXT
        n = len(z)
        if n < 2 * W + 1:
            score, k = 0.0, 0
        else:
            import numpy as _np
            cs = _np.concatenate([[0.0], _np.cumsum(z)])
            ts = _np.arange(W, n - W + 1)
            left = (cs[ts] - cs[ts - W]) / W
            right = (cs[ts + W] - cs[ts]) / W
            d = _np.abs(right - left)
            j = int(d.argmax()); score, k = float(d[j]), int(ts[j])
        scores[pid] = {"score": float(score), "pred_transition": int(k),
                       "pred_verse": int(split_to_verse(k, stride=C.STRIDE, window_len=C.WINDOW_LEN))}
        if i % 10 == 0:
            print(f"scored {i}/{len(ids)}", flush=True)

    write_json(Path(args.output), {
        "method": "llm_perplexity_mgpt_mean_shift",
        "representation": f"{args.model} per-window mean NLL (causal LM surprise)",
        "detector": f"mean_shift_W{C.MEAN_SHIFT_CONTEXT} on z-scored NLL profile",
        "scores": scores})
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
