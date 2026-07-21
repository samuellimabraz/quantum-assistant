#!/usr/bin/env python3
"""Extract a stratified random sample of the synthetic test split for manual audit (R1.2).

Selects up to ``--per-cell`` items from each (question_type, modality) cell of
the test split and writes them to a JSONL with the fields needed for manual
review: task_id, question, answer, test_code, entry_point, image_path,
category, question_type, source. No GPU, no model calls.

Usage:
    python scripts/sample_audit_set.py \
        --dataset /workspace/data/quantum-assistant/data \
        --split test \
        --per-cell 15 \
        --seed 42 \
        --out /workspace/paper/review_data/audit_set.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_dataset_split(dataset_path: Path, split: str):
    """Load a HuggingFace dataset split, supporting save_to_disk and parquet layouts."""
    from datasets import load_dataset, load_from_disk

    try:
        d = load_from_disk(str(dataset_path))
        return d[split]
    except (FileNotFoundError, KeyError):
        parquet_files = sorted(dataset_path.glob(f"{split}-*.parquet"))
        if not parquet_files:
            raise ValueError(
                f"No save_to_disk and no {split}-*.parquet files at {dataset_path}"
            )
        return load_dataset(
            "parquet", data_files=[str(p) for p in parquet_files], split="train"
        )


def image_descriptor(image: Any) -> str | None:
    """Summarize an image field without writing the bytes to the JSONL."""
    if image is None:
        return None
    if isinstance(image, str):
        return image
    if isinstance(image, dict):
        return image.get("path") or image.get("bytes")[:32].hex() if image.get("bytes") else None
    if hasattr(image, "save"):
        buf = io.BytesIO()
        try:
            image.save(buf, format="PNG")
            return f"PIL<{image.size[0]}x{image.size[1]}, {buf.tell()} bytes>"
        except Exception:
            return f"PIL<{image.size[0]}x{image.size[1]}>"
    return repr(image)[:80]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--per-cell", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory to dump image bytes as PNGs (optional).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ds = load_dataset_split(args.dataset, args.split)

    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(ds):
        qtype = row.get("type", row.get("question_type", "qa"))
        modality = "multimodal" if row.get("image") is not None else "text"
        buckets[(qtype, modality)].append(idx)

    selected: list[int] = []
    for key, indices in sorted(buckets.items()):
        rng.shuffle(indices)
        selected.extend(indices[: args.per_cell])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    images_dir: Path | None = args.images_dir
    if images_dir is not None:
        images_dir.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as out:
        for idx in selected:
            row = ds[idx]
            image_ref = image_descriptor(row.get("image"))
            image_path: str | None = None
            if images_dir is not None and hasattr(row.get("image"), "save"):
                image_path = str(images_dir / f"audit_{idx:05d}.png")
                row["image"].save(image_path)
            sample = {
                "task_id": f"synthetic/{args.split}/{idx}",
                "question_type": row.get("type", row.get("question_type", "qa")),
                "category": row.get("category", ""),
                "is_multimodal": row.get("image") is not None,
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "test_code": row.get("test_code", ""),
                "entry_point": row.get("entry_point", ""),
                "source": row.get("source", ""),
                "image_ref": image_ref,
                "image_path": image_path,
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[sample_audit_set] Wrote {len(selected)} samples to {args.out}")
    print("[sample_audit_set] Bucket counts:")
    for (qtype, modality), indices in sorted(buckets.items()):
        taken = min(len(indices), args.per_cell)
        print(f"  {qtype:22s} × {modality:10s}  taken {taken}/{len(indices)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
