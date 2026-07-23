from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from common.io_utils import read_json, write_json
import config as C


def percentile_threshold(values: list[float], target_fpr: float) -> float:
    if not values:
        raise ValueError("No training pure-host scores available for threshold calibration")
    try:
        return float(np.quantile(values, 1.0 - target_fpr, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, 1.0 - target_fpr, interpolation="higher"))


def stratified_bootstrap_auc(scores: np.ndarray, labels: np.ndarray, repetitions: int) -> tuple[float, float]:
    rng = np.random.default_rng(7)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    values: list[float] = []
    for _ in range(repetitions):
        idx = np.concatenate([
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ])
        values.append(float(roc_auc_score(labels[idx], scores[idx])))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def nearest_boundary_error(pred_verse: int, boundaries: list[int]) -> int:
    return min(abs(int(pred_verse) - int(boundary)) for boundary in boundaries)


def evaluate(method_data: dict, truth: dict[str, dict], train_ids: list[str], test_ids: list[str]) -> dict:
    scores = method_data["scores"]
    # Threshold is calibrated only on genuinely continuous training hosts.
    train_pure = [float(scores[p]["score"]) for p in train_ids if truth[p]["case_type"] == "pure"]
    threshold = percentile_threshold(train_pure, C.TARGET_FPR)

    # Primary evaluation: foreign-source splices vs continuous pure hosts.
    primary = [p for p in test_ids if truth[p]["case_type"] in {"foreign", "pure"}]
    y = np.array([1 if truth[p]["case_type"] == "foreign" else 0 for p in primary], dtype=int)
    s = np.array([float(scores[p]["score"]) for p in primary], dtype=float)
    pred = s >= threshold
    auc = float(roc_auc_score(y, s))
    ci_low, ci_high = stratified_bootstrap_auc(s, y, C.BOOTSTRAP_REPLICATES)

    pure_ids = [p for p in test_ids if truth[p]["case_type"] == "pure"]
    sham_ids = [p for p in test_ids if truth[p]["case_type"] == "sham"]
    pure_fpr = float(np.mean([float(scores[p]["score"]) >= threshold for p in pure_ids]))
    sham_alarm = float(np.mean([float(scores[p]["score"]) >= threshold for p in sham_ids]))

    detected_foreign = [
        p for p in test_ids
        if truth[p]["case_type"] == "foreign" and float(scores[p]["score"]) >= threshold
    ]
    errors = [nearest_boundary_error(int(scores[p]["pred_verse"]), truth[p]["boundaries"]) for p in detected_foreign]
    localization = float(np.mean([e <= C.LOCALIZATION_TOLERANCE_VERSES for e in errors])) if errors else None

    frontier: dict[str, float] = {}
    for contrast in C.CONTRASTS:
        for block_size in C.INSERT_SIZES:
            ids = [
                p for p in test_ids
                if truth[p]["case_type"] == "foreign"
                and truth[p]["contrast"] == contrast
                and int(truth[p]["block_size"]) == block_size
            ]
            frontier[f"{contrast}_{block_size}"] = float(np.mean([
                float(scores[p]["score"]) >= threshold for p in ids
            ]))

    fpr_curve, tpr_curve, _ = roc_curve(y, s)
    return {
        "method": method_data["method"],
        "representation": method_data.get("representation"),
        "detector": method_data.get("detector"),
        "threshold": threshold,
        "test_auc": auc,
        "auc_ci_95": [ci_low, ci_high],
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "pure_host_fpr": pure_fpr,
        "sham_splice_alarm_rate": sham_alarm,
        "n_detected_foreign": len(detected_foreign),
        "localization_hit_rate": localization,
        "localization_tolerance_verses": C.LOCALIZATION_TOLERANCE_VERSES,
        "median_boundary_error": float(np.median(errors)) if errors else None,
        "mean_boundary_error": float(np.mean(errors)) if errors else None,
        "frontier": frontier,
        "roc": [[float(x), float(yv)] for x, yv in zip(fpr_curve, tpr_curve)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all score files with the train/test firewall")
    parser.add_argument("dataset")
    parser.add_argument("results")
    args = parser.parse_args()
    dataset, results = Path(args.dataset), Path(args.results)
    split = read_json(dataset / "split.json")
    train_ids = [str(i) for i in split["train"]]
    test_ids = [str(i) for i in split["test"]]
    truth = {
        path.stem.split("-")[-1]: read_json(path)
        for path in (dataset / "truth").glob("truth-*.json")
    }

    rows: list[dict] = []
    for path in sorted(results.glob("scores_*.json")):
        rows.append(evaluate(read_json(path), truth, train_ids, test_ids))
    if not rows:
        raise FileNotFoundError(f"No scores_*.json files found in {results}")

    write_json(results / "comparison.json", rows)
    with (results / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "auc", "ci_low", "ci_high", "recall", "precision",
            "pure_fpr", "sham_alarm", "localization", "median_boundary_error",
        ])
        for row in rows:
            writer.writerow([
                row["method"], row["test_auc"], *row["auc_ci_95"], row["recall"],
                row["precision"], row["pure_host_fpr"], row["sham_splice_alarm_rate"],
                row["localization_hit_rate"], row["median_boundary_error"],
            ])

    print(f"{'method':42s} {'AUC':>6s} {'recall':>8s} {'prec':>7s} {'pureFPR':>8s} {'sham':>7s} {'loc':>7s}")
    for row in rows:
        loc = "NA" if row["localization_hit_rate"] is None else f"{row['localization_hit_rate']:.3f}"
        print(
            f"{row['method'][:42]:42s} {row['test_auc']:6.3f} {row['recall']:8.3f} "
            f"{row['precision']:7.3f} {row['pure_host_fpr']:8.3f} "
            f"{row['sham_splice_alarm_rate']:7.3f} {loc:>7s}"
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5.0, 4.5))
        for row in rows:
            xy = np.asarray(row["roc"])
            plt.plot(xy[:, 0], xy[:, 1], linewidth=2, label=f"{row['method']} (AUC={row['test_auc']:.2f})")
        plt.plot([0, 1], [0, 1], "--", linewidth=1)
        plt.xlabel("False-positive rate")
        plt.ylabel("True-positive rate")
        plt.title("Held-out benchmark comparison")
        plt.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        plt.savefig(results / "roc_comparison.png", dpi=200)
        plt.close()
    except Exception as exc:
        print(f"ROC figure skipped: {exc}")

    print(f"Saved {results / 'comparison.json'} and {results / 'comparison.csv'}")


if __name__ == "__main__":
    main()
