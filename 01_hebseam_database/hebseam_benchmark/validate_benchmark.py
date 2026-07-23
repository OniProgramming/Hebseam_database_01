from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Validate a generated HebSeam instance")
    p.add_argument("dataset")
    args = p.parse_args()
    root = Path(args.dataset)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    ids = split["train"] + split["test"]
    assert len(ids) == len(set(ids)) == manifest["n_cases"]
    for pid in ids:
        truth = json.loads((root / "truth" / f"truth-{pid}.json").read_text(encoding="utf-8"))
        lines = (root / "problems" / f"problem-{pid}.txt").read_text(encoding="utf-8").splitlines()
        assert len(lines) == truth["n_verses"] == manifest["final_verses_per_case"]
        assert truth["split"] in {"train", "test"}
        assert truth["case_type"] in {"foreign", "pure", "sham"}
        if truth["case_type"] == "foreign": assert len(truth["boundaries"]) == 2 and truth["label"] == 1
        if truth["case_type"] == "pure": assert truth["boundaries"] == [] and truth["label"] == 0
    for rel, expected in checksums.items():
        actual = sha256(root / rel)
        if actual != expected: raise ValueError(f"Checksum mismatch: {rel}")
    print(f"VALID: {manifest['name']} {manifest['version']} ({manifest['n_cases']} cases)")


if __name__ == "__main__": main()
