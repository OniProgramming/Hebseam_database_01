from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from common.detection import fit_standardizer, mean_shift_score, split_to_verse, standardize
from common.io_utils import problem_ids, read_json, read_jsonl, write_json
import config as C
from common.corpus import Verse


def trigrams(text: str) -> list[str]:
    compact = text.replace(" ", "")
    return [compact[i:i + 3] for i in range(max(0, len(compact) - 2))]


def load_verses(dataset: Path, pid: str) -> list[Verse]:
    return [Verse.from_dict(row) for row in read_jsonl(dataset / "problems" / f"problem-{pid}.jsonl")]


def build_training_vocab(dataset: Path, train_ids: list[str]) -> dict[str, list[str]]:
    function_counts: Counter[str] = Counter()
    trigram_counts: Counter[str] = Counter()
    for pid in train_ids:
        for verse in load_verses(dataset, pid):
            for token in verse.tokens:
                if token.morph.startswith(C.FUNCTION_POS_PREFIXES) and token.surface:
                    function_counts[token.surface] += 1
            trigram_counts.update(trigrams(verse.text))
    return {
        "function_surface": [x for x, _ in function_counts.most_common(C.N_FUNCTION_SURFACES)],
        "trigram": [x for x, _ in trigram_counts.most_common(C.N_TRIGRAMS)],
    }


def window_starts(n_verses: int) -> range:
    return range(0, n_verses - C.WINDOW_LEN + 1, C.STRIDE)


def feature_matrix(verses: list[Verse], vocab: dict[str, list[str]]) -> np.ndarray:
    f_index = {v: i for i, v in enumerate(vocab["function_surface"])}
    t_index = {v: i for i, v in enumerate(vocab["trigram"])}
    rows: list[np.ndarray] = []
    for start in window_starts(len(verses)):
        window = verses[start:start + C.WINDOW_LEN]
        surfaces = [t.surface for v in window for t in v.tokens if t.surface]
        n = max(len(surfaces), 1)
        counts = Counter(surfaces)
        lexical = np.array([
            len(counts) / n,
            sum(1 for value in counts.values() if value == 1) / n,
            np.mean([len(s) for s in surfaces]) if surfaces else 0.0,
        ], dtype=np.float64)

        function = np.zeros(len(f_index), dtype=np.float64)
        for v in window:
            for token in v.tokens:
                idx = f_index.get(token.surface)
                if idx is not None:
                    function[idx] += 1
        if function.sum() > 0:
            function /= function.sum()

        tri = np.zeros(len(t_index), dtype=np.float64)
        for g in trigrams(" ".join(surfaces)):
            idx = t_index.get(g)
            if idx is not None:
                tri[idx] += 1
        if tri.sum() > 0:
            tri /= tri.sum()
        rows.append(np.concatenate([lexical, function, tri]))
    return np.vstack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the proposed interpretable stylometric detector")
    parser.add_argument("dataset")
    parser.add_argument("output")
    args = parser.parse_args()
    dataset, output = Path(args.dataset), Path(args.output)
    split = read_json(dataset / "split.json")
    train_ids = [str(i) for i in split["train"]]
    vocab = build_training_vocab(dataset, train_ids)
    write_json(output.with_name("stylometric_vocab.json"), vocab)

    scores: dict[str, dict[str, float | int]] = {}
    ids = problem_ids(dataset)
    for i, pid in enumerate(ids, start=1):
        matrix = feature_matrix(load_verses(dataset, pid), vocab)
        z = standardize(matrix, fit_standardizer(matrix))
        score, split_index = mean_shift_score(z, C.MEAN_SHIFT_CONTEXT)
        scores[pid] = {
            "score": score,
            "pred_transition": split_index,
            "pred_verse": split_to_verse(split_index, stride=C.STRIDE, window_len=C.WINDOW_LEN),
        }
        if i % 40 == 0:
            print(f"scored {i}/{len(ids)}", flush=True)

    write_json(output, {
        "method": "proposed_interpretable_mean_shift",
        "representation": f"lexical(3)+surface-function({len(vocab['function_surface'])})+char-trigram({len(vocab['trigram'])})",
        "detector": f"mean_shift_W{C.MEAN_SHIFT_CONTEXT}",
        "scores": scores,
    })
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
