#!/usr/bin/env python3
"""DEPRECATED (ESWA revision): Image-masking matched-task delta was replaced by
``per_image_type_breakdown.py`` + within-subset base-vs-FT delta. The blanket
masking confounds image contribution with answerability on samples whose
text explicitly references the image (e.g. "the circuit shown", "this Bloch
sphere"). This script remains only as a reference for the earlier plan and
is no longer invoked by the orchestrator.

----- Original docstring -----
Join two synthetic evaluation runs (image kept vs masked) and report matched-task deltas (R1.1).

Expects two result JSONs produced by ``evaluate.runners.synthetic.SyntheticDatasetRunner``:
    * condition A: images kept    (``--with-image`` result)
    * condition B: images masked  (``--masked`` result)

Both runs must be over the *same* multimodal subset (produced via
``multimodal_only: true``). For each task_id present in both, computes
``pass@1(A) - pass@1(B)`` and aggregates by question_type. Emits a compact
summary JSON; prints a table to stdout.

Usage:
    python scripts/image_masking_delta.py \
        --with-image outputs/evaluate/synthetic/<model>/<run_A>.json \
        --masked     outputs/evaluate/synthetic/<model>/<run_B>.json \
        --model      qwen3-vl-ft-r32-2ep \
        --out        outputs/evaluate/image_masking/qwen3-vl-ft-r32-2ep.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def resolve_glob(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matches: {pattern}")
    return Path(matches[-1])


def load_run(path: Path) -> dict[str, dict[str, Any]]:
    """Return {task_id: row} for the synthetic runner's ``results`` array."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("results") or []
    return {row["task_id"]: row for row in rows}


def pass_at_1(row: dict[str, Any]) -> float | None:
    metrics = row.get("metrics") or {}
    if "pass@1" in metrics:
        return float(metrics["pass@1"])
    if "pass_rate" in metrics:
        return float(metrics["pass_rate"])
    # QA rows have no pass@1; use success flag as a fallback indicator.
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--with-image", required=True, help="Path or glob to condition A JSON.")
    parser.add_argument("--masked", required=True, help="Path or glob to condition B JSON.")
    parser.add_argument("--model", required=True, help="Model id for the summary record.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    path_a = resolve_glob(args.with_image)
    path_b = resolve_glob(args.masked)
    with_image = load_run(path_a)
    masked = load_run(path_b)

    common = sorted(set(with_image) & set(masked))
    if not common:
        print("[image_masking_delta] No common task_ids between the two runs.", file=sys.stderr)
        return 1

    per_row: list[dict[str, Any]] = []
    by_type: dict[str, list[float]] = defaultdict(list)
    by_image_type: dict[str, list[float]] = defaultdict(list)
    overall_a: list[float] = []
    overall_b: list[float] = []

    for task_id in common:
        a = with_image[task_id]
        b = masked[task_id]
        pa = pass_at_1(a)
        pb = pass_at_1(b)
        if pa is None or pb is None:
            continue
        delta = pa - pb
        qtype = a.get("question_type", "unknown")
        # Image type may be embedded under multiple keys; tolerate absence.
        itype = (
            a.get("image_type")
            or (a.get("metadata") or {}).get("image_type")
            or "unknown"
        )
        per_row.append(
            {
                "task_id": task_id,
                "question_type": qtype,
                "image_type": itype,
                "pass_at_1_with_image": pa,
                "pass_at_1_masked": pb,
                "delta": delta,
            }
        )
        by_type[qtype].append(delta)
        by_image_type[itype].append(delta)
        overall_a.append(pa)
        overall_b.append(pb)

    def summarize(buckets: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for key, values in buckets.items():
            if not values:
                continue
            out[key] = {
                "count": len(values),
                "mean_delta": sum(values) / len(values),
                "positive_fraction": sum(1 for v in values if v > 0) / len(values),
            }
        return out

    summary = {
        "model": args.model,
        "source_with_image": str(path_a),
        "source_masked": str(path_b),
        "matched_samples": len(per_row),
        "overall": {
            "pass_at_1_with_image": sum(overall_a) / len(overall_a) if overall_a else None,
            "pass_at_1_masked": sum(overall_b) / len(overall_b) if overall_b else None,
            "mean_delta": (
                sum(overall_a) - sum(overall_b)
            ) / len(overall_a) if overall_a else None,
        },
        "by_question_type": summarize(by_type),
        "by_image_type": summarize(by_image_type),
        "per_task": per_row,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[image_masking_delta] Wrote {args.out}")
    overall = summary["overall"]
    if overall["pass_at_1_with_image"] is not None:
        print(
            f"  overall  n={len(per_row):4d}  with_image={overall['pass_at_1_with_image']:.4f}"
            f"  masked={overall['pass_at_1_masked']:.4f}  delta={overall['mean_delta']:+.4f}"
        )
    for title, buckets in (("by_question_type", by_type), ("by_image_type", by_image_type)):
        print(f"  {title}:")
        for key, values in sorted(buckets.items()):
            if not values:
                continue
            mean_delta = sum(values) / len(values)
            print(f"    {key:22s}  n={len(values):4d}  mean_delta={mean_delta:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
