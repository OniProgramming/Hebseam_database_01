# HebSeam Methods and Comparative Evaluation

This folder is intentionally independent from the public benchmark generator. It does not generate or modify the database. Every method receives the database location through the `dataset` command-line argument, so the same code can be applied to:

- the official frozen HebSeam database;
- a newly generated HebSeam database;
- a separately downloaded compatible database.

## Methods

- `run_proposed.py`: interpretable stylometric representation + mean-shift detection.
- `run_classical.py`: character-trigram TF-IDF + classical change-point detection.
- `run_neural.py`: multilingual MiniLM embeddings loaded directly with Hugging Face Transformers and PyTorch; TensorFlow/Keras is not used.
- `run_transformer.py`: raw XLM-R embeddings + mean-shift detection.
- `compare.py`: training-threshold calibration and held-out comparison.

Prediction scripts read only `problems/`. They do not read `truth/`. Ground truth is used only by `compare.py`.

## Install

```bash
pip install -r requirements.txt
```

The neural models are downloaded automatically on their first run.

## Run everything on a separately supplied database

```bash
python run_all.py <DB_PATH> --results results
```

Windows/PyCharm example:

```bash
python run_all.py ../01_hebseam_database/data/generated/hebseam-v1.0.0 --results results
```

CPU-only explicit run:

```bash
python run_all.py <DB_PATH> --results results --device cpu
```

Quick smoke test:

```bash
python run_all.py <DB_PATH> --results results --skip-neural --skip-transformer
```

## Continue after the earlier Keras error

The updated neural baseline no longer imports `sentence-transformers`, TensorFlow, or Keras. Reinstall the method requirements and rerun:

```bash
pip uninstall -y sentence-transformers
pip install -r requirements.txt
python run_all.py <DB_PATH> --results results
```

Existing `scores_proposed.json` and `scores_classical.json` may be overwritten safely.

## Run separately

```bash
python methods/run_proposed.py <DB_PATH> results/scores_proposed.json
python methods/run_classical.py <DB_PATH> results/scores_classical.json
python methods/run_neural.py <DB_PATH> results/scores_neural.json --device cpu
python methods/run_transformer.py <DB_PATH> results/scores_transformer.json --device cpu
python compare.py <DB_PATH> results
```
