from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .corpus import Verse, devocalize, load_osis, pseudo_verses


@dataclass(frozen=True)
class Settings:
    name: str
    version: str
    seed: int
    final_verses: int
    insert_sizes: tuple[int, ...]
    contrasts: tuple[str, ...]
    repetitions_per_cell: int
    pure_cases: int
    sham_cases: int
    interior_margin: int
    genesis_train: tuple[int, int]
    genesis_test: tuple[int, int]
    mishnah_target_tokens: int


@dataclass
class Case:
    split: str
    case_type: str
    contrast: str
    block_size: int
    verses: list[Verse]
    boundaries: list[int]
    provenance: dict[str, Any]


def load_settings(path: Path) -> tuple[Settings, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    b, s, p = raw["benchmark"], raw["splits"], raw["preprocessing"]
    cfg = Settings(
        name=str(b["name"]), version=str(b["version"]), seed=int(b["seed"]),
        final_verses=int(b["final_verses"]), insert_sizes=tuple(map(int, b["insert_sizes"])),
        contrasts=tuple(map(str, b["contrasts"])), repetitions_per_cell=int(b["repetitions_per_cell"]),
        pure_cases=int(b["pure_cases"]), sham_cases=int(b["sham_cases"]),
        interior_margin=int(b["interior_margin"]), genesis_train=tuple(map(int, s["genesis_train"])),
        genesis_test=tuple(map(int, s["genesis_test"])),
        mishnah_target_tokens=int(p["mishnah_target_tokens_per_pseudo_verse"]),
    )
    validate_settings(cfg)
    return cfg, raw


def validate_settings(c: Settings) -> None:
    if c.repetitions_per_cell % 2 or c.pure_cases % 2 or c.sham_cases % 2:
        raise ValueError("repetitions_per_cell, pure_cases, and sham_cases must be even")
    if max(c.insert_sizes) >= c.final_verses:
        raise ValueError("Every insertion size must be smaller than final_verses")
    if 2 * c.interior_margin >= c.final_verses - max(c.insert_sizes):
        raise ValueError("interior_margin leaves no valid insertion position")
    if c.genesis_train[1] > c.genesis_test[0]:
        raise ValueError("Genesis train and test pools must be disjoint")


def flatten_text(value: Any, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for item in value:
            flatten_text(item, out)


def load_mishnah(data_dir: Path, target_tokens: int) -> list[Verse]:
    segments: list[str] = []
    for path in sorted(data_dir.glob("m_*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        flatten_text(payload.get("text", []), segments)
    words: list[str] = []
    for segment in segments:
        segment = segment.replace("\u05be", " ")
        words.extend(devocalize(w) for w in re.split(r"\s+", segment) if devocalize(w))
    if not words:
        raise FileNotFoundError("No Mishnah JSON files matching m_*.json were found or parsed")
    return pseudo_verses(words, source="mishnah", target_tokens=target_tokens)


def split_pool(items: list[Verse], split: str) -> tuple[list[Verse], int]:
    midpoint = len(items) // 2
    return (items[:midpoint], 0) if split == "train" else (items[midpoint:], midpoint)


def sample_contiguous(rng: np.random.Generator, pool: list[Verse], length: int) -> tuple[int, list[Verse]]:
    if length > len(pool):
        raise ValueError(f"Requested {length} verses from a pool of {len(pool)}")
    start = int(rng.integers(0, len(pool) - length + 1))
    return start, pool[start:start + length]


def sample_nonoverlap_pair(rng: np.random.Generator, pool: list[Verse], host_len: int, block_len: int):
    for _ in range(10000):
        h_start, host = sample_contiguous(rng, pool, host_len)
        b_start, block = sample_contiguous(rng, pool, block_len)
        if h_start + host_len <= b_start or b_start + block_len <= h_start:
            return h_start, host, b_start, block
    raise RuntimeError("Could not sample a non-overlapping host and same-source block")


def insertion_position(rng: np.random.Generator, host_len: int, margin: int) -> int:
    return int(rng.integers(margin, host_len - margin + 1))


def foreign_case(rng, c: Settings, split, contrast, block_size, gen_pool, gen_offset, source_pool, source_offset):
    base_len = c.final_verses - block_size
    host_start, host = sample_contiguous(rng, gen_pool, base_len)
    source_start, block = sample_contiguous(rng, source_pool, block_size)
    at = insertion_position(rng, base_len, c.interior_margin)
    return Case(split, "foreign", contrast, block_size, host[:at] + block + host[at:], [at, at + block_size], {
        "host_source": "genesis", "host_start": gen_offset + host_start, "host_length_before_insertion": base_len,
        "insert_source": contrast, "insert_source_start": source_offset + source_start,
    })


def pure_case(rng, c: Settings, split, gen_pool, gen_offset):
    start, verses = sample_contiguous(rng, gen_pool, c.final_verses)
    return Case(split, "pure", "pure", 0, verses, [], {"host_source": "genesis", "host_start": gen_offset + start})


def sham_case(rng, c: Settings, split, block_size, gen_pool, gen_offset):
    base_len = c.final_verses - block_size
    hs, host, bs, block = sample_nonoverlap_pair(rng, gen_pool, base_len, block_size)
    at = insertion_position(rng, base_len, c.interior_margin)
    return Case(split, "sham", "same_source", block_size, host[:at] + block + host[at:], [at, at + block_size], {
        "host_source": "genesis", "host_start": gen_offset + hs, "host_length_before_insertion": base_len,
        "insert_source": "genesis", "insert_source_start": gen_offset + bs,
    })


def build_cases(data_dir: Path, c: Settings) -> list[Case]:
    rng = np.random.default_rng(c.seed)
    genesis = load_osis(data_dir / "Gen.xml", "genesis")
    exodus = load_osis(data_dir / "Exod.xml", "exodus")
    psalms = load_osis(data_dir / "Ps.xml", "psalms")
    mishnah = load_mishnah(data_dir, c.mishnah_target_tokens)
    pools = {
        "train": (genesis[c.genesis_train[0]:c.genesis_train[1]], c.genesis_train[0]),
        "test": (genesis[c.genesis_test[0]:c.genesis_test[1]], c.genesis_test[0]),
    }
    sources = {"exodus": exodus, "psalms": psalms, "mishnah": mishnah}
    cases: list[Case] = []
    per_split = c.repetitions_per_cell // 2
    for split in ("train", "test"):
        gen_pool, gen_offset = pools[split]
        for contrast in c.contrasts:
            source_pool, source_offset = split_pool(sources[contrast], split)
            for size in c.insert_sizes:
                for _ in range(per_split):
                    cases.append(foreign_case(rng, c, split, contrast, size, gen_pool, gen_offset, source_pool, source_offset))
        for _ in range(c.pure_cases // 2):
            cases.append(pure_case(rng, c, split, gen_pool, gen_offset))
        for _ in range(c.sham_cases // 2):
            cases.append(sham_case(rng, c, split, int(rng.choice(c.insert_sizes)), gen_pool, gen_offset))
    rng.shuffle(cases)
    return cases


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible HebSeam benchmark instance")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/generated/hebseam-v1.0.0-seed20260706")
    parser.add_argument("--seed", type=int, help="Override the seed from config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path, data_dir, output_dir = Path(args.config), Path(args.data_dir), Path(args.output_dir)
    settings, raw_config = load_settings(config_path)
    if args.seed is not None:
        settings = Settings(**{**settings.__dict__, "seed": args.seed})
        raw_config["benchmark"]["seed"] = args.seed

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} exists; use --force to replace it")
        shutil.rmtree(output_dir)
    (output_dir / "problems").mkdir(parents=True)
    (output_dir / "truth").mkdir(parents=True)

    cases = build_cases(data_dir, settings)
    split: dict[str, list[str]] = {"train": [], "test": []}
    case_index: list[dict[str, Any]] = []
    checksums: dict[str, str] = {}
    for i, case in enumerate(cases):
        pid = f"{i:04d}"
        split[case.split].append(pid)
        jsonl_path = output_dir / "problems" / f"problem-{pid}.jsonl"
        txt_path = output_dir / "problems" / f"problem-{pid}.txt"
        truth_path = output_dir / "truth" / f"truth-{pid}.json"
        write_jsonl(jsonl_path, (v.to_dict() for v in case.verses))
        txt_path.write_text("\n".join(v.text for v in case.verses) + "\n", encoding="utf-8")
        write_json(truth_path, {
            "id": pid, "split": case.split, "label": 1 if case.case_type == "foreign" else 0,
            "case_type": case.case_type, "contrast": case.contrast, "block_size": case.block_size,
            "boundaries": case.boundaries, "n_verses": len(case.verses), "provenance": case.provenance,
        })
        case_index.append({"id": pid, "split": case.split, "n_verses": len(case.verses)})
        for p in (jsonl_path, txt_path, truth_path):
            checksums[str(p.relative_to(output_dir))] = sha256(p)

    write_json(output_dir / "split.json", split)
    write_json(output_dir / "cases.json", case_index)
    write_json(output_dir / "generation_config.json", raw_config)
    manifest = {
        "name": settings.name, "version": settings.version, "seed": settings.seed,
        "n_cases": len(cases), "final_verses_per_case": settings.final_verses,
        "primary_task": "foreign-source splice versus continuous pure host",
        "stress_test": "same-source sham-splice alarm rate",
        "ground_truth_type": "controlled structural ground truth",
        "generator": "hebseam-benchmark", "checksums_file": "checksums.json",
    }
    write_json(output_dir / "manifest.json", manifest)
    checksums["split.json"] = sha256(output_dir / "split.json")
    checksums["cases.json"] = sha256(output_dir / "cases.json")
    checksums["generation_config.json"] = sha256(output_dir / "generation_config.json")
    checksums["manifest.json"] = sha256(output_dir / "manifest.json")
    write_json(output_dir / "checksums.json", checksums)
    print(f"Generated {len(cases)} cases in {output_dir}")
    print(f"train={len(split['train'])}, test={len(split['test'])}, seed={settings.seed}")


if __name__ == "__main__":
    main()
