#!/usr/bin/env python3
"""Re-aggregate a saved Qiskit HumanEval (Hard) result JSON by difficulty and category.

Reads the per-problem ``results[]`` array saved by
``evaluate.runners.qiskit_humaneval.QiskitHumanEvalRunner._save_results`` and
joins it with the original QHE-Hard dataset JSON to recover each problem's
``difficulty`` and thematic tag. Emits a compact summary JSON with per-bucket
Pass@1 and bucket sizes.

Usage:
    python scripts/stratify_qhe_hard.py \
        --results outputs/evaluate/qiskit-humaneval-hard/<model>/<run>.json \
        --dataset /workspace/qiskit-human-eval/dataset/dataset_qiskit_test_human_eval_hard.json \
        --out outputs/evaluate/stratified/<model>.json

Pure re-aggregation, no GPU. Missing fields degrade gracefully (only the
buckets that have data are reported).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

CATEGORY_KEYS = ("category", "topic", "tag", "tags")
DIFFICULTY_KEYS = ("difficulty", "level")


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_results_glob(pattern: str) -> Path:
    """Resolve a glob to a single results file, preferring the most recent."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No results file matches: {pattern}")
    return Path(matches[-1])


def get_first(container: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = container.get(key)
        if isinstance(value, list):
            if value:
                return str(value[0])
        elif value is not None:
            return str(value)
    return None


def index_dataset(dataset_path: Path) -> dict[str, dict[str, str]]:
    """Map task_id -> {difficulty, category} from the raw QHE dataset JSON."""
    data = load_json(dataset_path)
    items = data if isinstance(data, list) else data.get("data", data)
    index: dict[str, dict[str, str]] = {}
    for item in items:
        task_id = item.get("task_id") or item.get("id") or item.get("name")
        if task_id is None:
            continue
        index[str(task_id)] = {
            "difficulty": get_first(item, DIFFICULTY_KEYS) or "unknown",
            "category": get_first(item, CATEGORY_KEYS) or "unknown",
        }
    return index


def bucket_pass_rate(
    per_problem: list[dict[str, Any]],
    index: dict[str, dict[str, str]],
    key: str,
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in per_problem:
        task_id = str(row.get("task_id", ""))
        metadata = index.get(task_id, {})
        bucket = metadata.get(key, "unknown")
        # Use pass_rate (fraction of samples that passed). For n=1 this is 0/1.
        passed = row.get("num_passed")
        total = row.get("num_samples") or row.get("num_total")
        if passed is None or total in (None, 0):
            continue
        buckets[bucket].append(float(passed) / float(total))

    summary: dict[str, dict[str, Any]] = {}
    for bucket, values in buckets.items():
        summary[bucket] = {
            "count": len(values),
            "pass_at_1": sum(values) / len(values) if values else None,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results",
        required=True,
        help="Path or glob to the saved QHE-Hard result JSON (picks the latest match).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to the raw QHE-Hard dataset JSON.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output summary JSON path.")
    args = parser.parse_args()

    results_path = resolve_results_glob(args.results)
    results = load_json(results_path)
    per_problem = results.get("results") or results.get("per_problem_results") or []
    if not per_problem:
        print(f"[stratify_qhe_hard] No per-problem results in {results_path}", file=sys.stderr)
        return 1

    index = index_dataset(args.dataset)

    by_difficulty = bucket_pass_rate(per_problem, index, "difficulty")
    by_category = bucket_pass_rate(per_problem, index, "category")

    summary = {
        "source_results": str(results_path),
        "source_dataset": str(args.dataset),
        "total_problems": len(per_problem),
        "by_difficulty": by_difficulty,
        "by_category": by_category,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[stratify_qhe_hard] Wrote {args.out}")
    for title, buckets in (("By difficulty", by_difficulty), ("By category", by_category)):
        print(f"  {title}:")
        for bucket, stats in sorted(buckets.items()):
            pass1 = stats.get("pass_at_1")
            pass1_str = f"{pass1:.4f}" if pass1 is not None else "-"
            print(f"    {bucket:30s}  n={stats['count']:4d}  pass@1={pass1_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
