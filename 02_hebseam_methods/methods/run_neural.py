from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# The baseline is PyTorch-only. Prevent Transformers from importing optional
# TensorFlow/Keras components from the user's environment.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.detection import fit_standardizer, mean_shift_score, split_to_verse, standardize
from common.io_utils import problem_ids, write_json
import config as C


def load_lines(dataset: Path, pid: str) -> list[str]:
    path = dataset / "problems" / f"problem-{pid}.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_windows(lines: list[str]) -> list[str]:
    return [
        " ".join(lines[i:i + C.WINDOW_LEN])
        for i in range(0, len(lines) - C.WINDOW_LEN + 1, C.STRIDE)
    ]


def mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_windows(
    texts: list[str], tokenizer, model, device: str, batch_size: int, max_length: int
) -> np.ndarray:
    import torch

    vectors: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
    return np.vstack(vectors).astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyTorch-only multilingual neural embedding baseline for HebSeam"
    )
    parser.add_argument("dataset", help="Path to any valid HebSeam benchmark instance")
    parser.add_argument("output", help="Output scores JSON")
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install the method dependencies: pip install -r requirements.txt") from exc

    dataset = Path(args.dataset).resolve()
    output = Path(args.output)
    if not (dataset / "problems").is_dir():
        raise SystemExit(f"Invalid HebSeam dataset: missing {dataset / 'problems'}")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    print(f"Loading neural model: {args.model} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)

    ids = problem_ids(dataset)
    if not ids:
        raise SystemExit(f"No problem-*.jsonl files found in {dataset / 'problems'}")

    scores: dict[str, dict[str, float | int]] = {}
    for index, pid in enumerate(ids, start=1):
        texts = make_windows(load_lines(dataset, pid))
        x = encode_windows(texts, tokenizer, model, device, args.batch_size, args.max_length)
        z = standardize(x, fit_standardizer(x))
        score, split_index = mean_shift_score(z, C.MEAN_SHIFT_CONTEXT)
        scores[pid] = {
            "score": float(score),
            "pred_transition": int(split_index),
            "pred_verse": int(
                split_to_verse(split_index, stride=C.STRIDE, window_len=C.WINDOW_LEN)
            ),
        }
        if index % 20 == 0 or index == len(ids):
            print(f"scored {index}/{len(ids)}", flush=True)

    write_json(
        output,
        {
            "method": "neural_multilingual_minilm_mean_shift",
            "representation": args.model,
            "framework": "pytorch_transformers",
            "detector": f"mean_shift_W{C.MEAN_SHIFT_CONTEXT}",
            "scores": scores,
        },
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
