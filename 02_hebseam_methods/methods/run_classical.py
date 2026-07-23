from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import ruptures as rpt
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from common.detection import split_to_verse
from common.io_utils import problem_ids, read_json, write_json
import config as C


def load_lines(dataset: Path, pid: str) -> list[str]:
    return [x for x in (dataset / "problems" / f"problem-{pid}.txt").read_text(encoding="utf-8").splitlines() if x]


def windows(lines: list[str]) -> list[str]:
    return [" ".join(lines[i:i + C.WINDOW_LEN]) for i in range(0, len(lines) - C.WINDOW_LEN + 1, C.STRIDE)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical TF-IDF + Binary Segmentation baseline")
    parser.add_argument("dataset")
    parser.add_argument("output")
    args = parser.parse_args()
    dataset, output = Path(args.dataset), Path(args.output)
    split = read_json(dataset / "split.json")
    train_ids = [str(i) for i in split["train"]]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 3),
        max_features=3000,
        sublinear_tf=True,
        norm="l2",
    )
    vectorizer.fit(text for pid in train_ids for text in windows(load_lines(dataset, pid)))

    scores: dict[str, dict[str, float | int]] = {}
    ids = problem_ids(dataset)
    for i, pid in enumerate(ids, start=1):
        x = vectorizer.transform(windows(load_lines(dataset, pid))).toarray().astype(np.float64)
        n = x.shape[0]
        try:
            algo = rpt.Binseg(model="rbf", jump=1, min_size=2).fit(x)
            breakpoint = int(algo.predict(n_bkps=1)[0])
            full_cost = float(algo.cost.error(0, n))
            split_cost = float(algo.cost.error(0, breakpoint) + algo.cost.error(breakpoint, n))
            score = max(0.0, full_cost - split_cost)
        except Exception:
            breakpoint, score = 0, 0.0
        scores[pid] = {
            "score": score,
            "pred_transition": breakpoint,
            "pred_verse": split_to_verse(breakpoint, stride=C.STRIDE, window_len=C.WINDOW_LEN),
        }
        if i % 40 == 0:
            print(f"scored {i}/{len(ids)}", flush=True)

    write_json(output, {
        "method": "classical_tfidf_binseg_rbf",
        "representation": "training-fitted character-trigram TF-IDF",
        "detector": "ruptures Binary Segmentation, one breakpoint, RBF cost",
        "scores": scores,
    })
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
