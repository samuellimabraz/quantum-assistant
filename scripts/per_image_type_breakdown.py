#!/usr/bin/env python3
"""Per-image-type Pass@1 breakdown on the synthetic multimodal subset (R1.1 redesigned).

Reads a saved synthetic evaluation JSON produced by ``SyntheticDatasetRunner``
and joins it with the source HuggingFace dataset to recover an image-type
label for every multimodal sample.

The synthetic dataset parquet does not ship an explicit ``image_type``
column, so the classifier reuses the ``_heuristic_classify`` rules of the
data-generation pipeline (``synthetic_data.extractors.transcriber``) applied
to the question + answer text. The rules are identical to those used during
dataset construction, so the labelling is consistent with the internal
pipeline taxonomy: ``circuit``, ``chart``, ``bloch_sphere``, ``formula``,
``table``, ``code_output``, ``diagram``, ``decorative``, ``unknown``.

Output JSON:
{
  "model": "...",
  "source_results": "...",
  "multimodal_samples": 581,
  "overall_mm_pass@1": 0.647,
  "by_image_type": {
    "circuit":      {"count": 312, "pass@1": 0.71, "type_breakdown": {"code": .., "qa": ..}},
    "chart":        {"count":  88, "pass@1": 0.66},
    ...
  }
}

Usage:
    python scripts/per_image_type_breakdown.py \
        --results outputs/evaluate/synthetic/<model>/<run>.json \
        --dataset /workspace/data/quantum-assistant/data \
        --model   <label> \
        --out     outputs/evaluate/image_type/<model>.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset, load_from_disk  # type: ignore
except ImportError:
    load_dataset = None
    load_from_disk = None


# Taxonomy identical to ``synthetic_data.parsers.base.ImageType`` values.
IMAGE_TYPES = (
    "circuit",
    "chart",
    "bloch_sphere",
    "formula",
    "table",
    "code_output",
    "diagram",
    "decorative",
    "unknown",
)


CIRCUIT_KW = [
    "quantum circuit",
    "circuit diagram",
    "hadamard gate",
    "cnot gate",
    "cx gate",
    "pauli",
    "rotation gate",
    "bell circuit",
    "ghz circuit",
    "qft circuit",
    "ansatz",
    "barrier",
]
CHART_KW = [
    "histogram",
    "bar chart",
    "measurement result",
    "measurement distribution",
    "probability distribution",
    "counts",
    "bar plot",
    "line plot",
    "y-axis",
    "x-axis",
    "error rate",
    "curve",
    "waveform",
    "pulse shape",
    "gaussian waveform",
    "gaussian‑like",
    "envelope",
    "bell-shaped",
    "modulation",
    "plotted",
    "y axis",
    "x axis",
]
BLOCH_KW = [
    "bloch sphere",
    "bloch vector",
    "qubit state vector",
    "spherical representation",
]
FORMULA_KW = [
    "equation",
    "formula",
    "integral",
    "summation",
    "hamiltonian",
    "operator expression",
    "derivation",
    "mathematical expression",
    r"h^{\otimes",
    r"\otimes",
    r"\sum",
    r"\sqrt",
    r"\frac",
    "latex",
]
TABLE_KW = [
    "table shown",
    "as shown in the table",
    "data table",
    "rows and columns",
]
CODE_OUT_KW = [
    "code output",
    "printed output",
    "traceback",
    "log output",
    "terminal output",
]
DIAGRAM_KW = [
    "diagram",
    "schematic",
    "architecture",
    "architectural",
    "experimental setup",
    "atomic model",
    "lattice",
    "cryostat",
    "cryogenic system",
    "photonic",
    "laser diode",
    "etching",
    "bus",
    "potential-well",
    "potential‑well",
    "energy level",
    "energy levels",
    "setup",
    "system shown",
    "device shown",
    "rubric",
    "flowchart",
    "block diagram",
    "stack",
    "topology",
    "panel",
    "components shown",
    "illustrated",
    "depicted in the image",
    "shown in the image",
    "shown in the figure",
    "scheme",
]


def _any_in(text: str, keywords) -> bool:
    return any(kw in text for kw in keywords)


def heuristic_image_type(question: str, answer: str, category: str, source: str) -> str:
    """Classify multimodal sample to an ImageType bucket using question+answer text.

    Mirrors ``synthetic_data.extractors.transcriber._heuristic_classify`` but
    operates on the Q&A text (the transcription itself is not stored in the
    published parquet). Priority order matches the original implementation.
    """

    q = (question or "").lower()
    a = (answer or "").lower()
    blob = f"{q}\n{a}"

    # Bloch first (specific, otherwise words like "state" get swallowed by chart).
    if _any_in(blob, BLOCH_KW):
        return "bloch_sphere"

    # Circuit (stricter than the pipeline's heuristic because "circuit"/"gate"/
    # "qubit" alone are ubiquitous in quantum text).
    if _any_in(blob, CIRCUIT_KW) or ("circuit" in q and "image" in q):
        return "circuit"

    # Chart / histogram / plots.
    if _any_in(blob, CHART_KW):
        return "chart"

    # Formula / equation / mathematical expression.
    if _any_in(blob, FORMULA_KW):
        return "formula"

    # Table.
    if _any_in(blob, TABLE_KW):
        return "table"

    # Code output.
    if _any_in(blob, CODE_OUT_KW):
        return "code_output"

    # General technical diagrams (schematics, setups, lattices). Checked after
    # the specific types so "circuit diagram" does not leak into this bucket.
    if _any_in(blob, DIAGRAM_KW):
        return "diagram"

    # Fallback by source hint.
    src = (source or "").lower()
    if "bloch" in src:
        return "bloch_sphere"
    if "histogram" in src or "counts" in src:
        return "chart"
    if "papers/" in src:
        # Science papers ingested here are physics/hardware diagrams.
        return "diagram"
    if re.search(r"\.ipynb\b", src):
        # Jupyter outputs in category-driven content are usually circuits.
        if category in ("circuits_and_gates", "quantum_info_and_operators"):
            return "circuit"

    return "unknown"


def load_hf_dataset(path: Path, split: str = "test"):
    if load_dataset is None:
        raise RuntimeError("datasets package not installed")

    # Support both save_to_disk and plain parquet directories.
    if (path / "dataset_info.json").exists():
        return load_from_disk(str(path))[split]

    if (path / f"{split}-00000-of-00001.parquet").exists():
        files = sorted(str(p) for p in path.glob(f"{split}-*.parquet"))
        return load_dataset("parquet", data_files=files, split="train")

    # Maybe path points directly at a parquet file or a different layout.
    files = sorted(str(p) for p in path.glob("**/test-*.parquet"))
    if files:
        return load_dataset("parquet", data_files=files, split="train")

    raise FileNotFoundError(f"No synthetic test split found under {path}")


def resolve_results_glob(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No results file matches: {pattern}")
    return Path(matches[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, help="Path or glob to the saved synthetic result JSON.")
    parser.add_argument("--dataset", required=True, type=Path, help="Synthetic dataset dir (parquet or save_to_disk).")
    parser.add_argument("--model", required=True, help="Label to write into the output summary.")
    parser.add_argument("--split", default="test", help="Dataset split used at eval time.")
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path.")
    args = parser.parse_args()

    results_path = resolve_results_glob(args.results)
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
    per_problem = results.get("results") or []
    if not per_problem:
        print(f"[image_type] No per-problem results in {results_path}", file=sys.stderr)
        return 1

    ds = load_hf_dataset(args.dataset, args.split)

    # Classify image_type per dataset row for the multimodal samples only.
    image_types_by_idx: dict[int, str] = {}
    is_mm_by_idx: dict[int, bool] = {}
    q_type_by_idx: dict[int, str] = {}
    category_by_idx: dict[int, str] = {}
    for idx, row in enumerate(ds):
        has_image = row.get("image") is not None
        is_mm_by_idx[idx] = has_image
        q_type_by_idx[idx] = row.get("type", "")
        category_by_idx[idx] = row.get("category", "")
        if has_image:
            image_types_by_idx[idx] = heuristic_image_type(
                row.get("question", ""),
                row.get("answer", ""),
                row.get("category", ""),
                row.get("source", ""),
            )

    # Aggregate by image_type over the MM subset.
    by_type: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "passed": 0.0,
            "by_type": defaultdict(lambda: {"count": 0, "passed": 0.0}),
            "by_category": defaultdict(lambda: {"count": 0, "passed": 0.0}),
        }
    )

    mm_count = 0
    mm_passed = 0.0
    for row in per_problem:
        task_id = str(row.get("task_id", ""))
        m = re.match(r"synthetic/[^/]+/(\d+)$", task_id)
        if not m:
            continue
        idx = int(m.group(1))
        if not is_mm_by_idx.get(idx, False):
            continue

        # Extract pass rate. For n=1 eval this is 0/1; supports n>1 too.
        passed = row.get("num_passed")
        total = row.get("num_samples") or row.get("num_total")
        if passed is None or total in (None, 0):
            # Fallback to the boolean ``success`` field.
            total = 1
            passed = 1 if row.get("success") else 0
        pass_rate = float(passed) / float(total)

        img_type = image_types_by_idx[idx]
        bucket = by_type[img_type]
        bucket["count"] += 1
        bucket["passed"] += pass_rate

        qt = q_type_by_idx.get(idx, "")
        if qt:
            bt = bucket["by_type"][qt]
            bt["count"] += 1
            bt["passed"] += pass_rate

        cat = category_by_idx.get(idx, "")
        if cat:
            bc = bucket["by_category"][cat]
            bc["count"] += 1
            bc["passed"] += pass_rate

        mm_count += 1
        mm_passed += pass_rate

    # Format output.
    summary_by_type: dict[str, Any] = {}
    for img_type in IMAGE_TYPES:
        bucket = by_type.get(img_type)
        if not bucket or bucket["count"] == 0:
            continue
        summary_by_type[img_type] = {
            "count": bucket["count"],
            "pass@1": bucket["passed"] / bucket["count"],
            "by_question_type": {
                qt: {"count": v["count"], "pass@1": v["passed"] / v["count"]}
                for qt, v in bucket["by_type"].items()
            },
            "by_category": {
                c: {"count": v["count"], "pass@1": v["passed"] / v["count"]}
                for c, v in bucket["by_category"].items()
            },
        }

    summary = {
        "model": args.model,
        "source_results": str(results_path),
        "multimodal_samples": mm_count,
        "overall_mm_pass@1": mm_passed / mm_count if mm_count else 0.0,
        "by_image_type": summary_by_type,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[image_type] wrote {args.out}")
    print(f"  multimodal samples: {mm_count}   overall MM Pass@1: {summary['overall_mm_pass@1']:.4f}")
    for img_type, stats in summary_by_type.items():
        print(f"  {img_type:14s}  n={stats['count']:4d}  pass@1={stats['pass@1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
