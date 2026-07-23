from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def validate_dataset(path: Path) -> Path:
    dataset = path.expanduser().resolve()
    required = [dataset / "problems", dataset / "truth", dataset / "split.json"]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise SystemExit("Invalid HebSeam dataset. Missing: " + ", ".join(missing))
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HebSeam methods against an independently supplied benchmark folder"
    )
    parser.add_argument("dataset", help="Path to any valid generated/frozen HebSeam database")
    parser.add_argument("--results", default="results")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--skip-transformer", action="store_true")
    parser.add_argument(
        "--neural-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--transformer-model", default="xlm-roberta-base")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable
    dataset = validate_dataset(Path(args.dataset))
    results = Path(args.results).expanduser().resolve()
    results.mkdir(parents=True, exist_ok=True)

    run(py, str(root / "methods" / "run_proposed.py"), str(dataset), str(results / "scores_proposed.json"))
    run(py, str(root / "methods" / "run_classical.py"), str(dataset), str(results / "scores_classical.json"))

    if not args.skip_neural:
        run(
            py,
            str(root / "methods" / "run_neural.py"),
            str(dataset),
            str(results / "scores_neural.json"),
            "--model",
            args.neural_model,
            "--device",
            args.device,
        )

    if not args.skip_transformer:
        run(
            py,
            str(root / "methods" / "run_transformer.py"),
            str(dataset),
            str(results / "scores_transformer.json"),
            "--model",
            args.transformer_model,
            "--device",
            args.device,
        )

    run(py, str(root / "compare.py"), str(dataset), str(results))


if __name__ == "__main__":
    main()
