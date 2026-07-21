#!/usr/bin/env python3
"""Consolidate all review-experiment results into summary tables.

Reads the result JSONs produced by the review compute orchestrator and outputs:
  - paper/review_data/analysis.json   (machine-readable)
  - paper/review_data/analysis_tables.md  (human-readable markdown tables)

Usage:
    python scripts/analyze_review_results.py \
        --results-dir paper/review_data/evaluate \
        --out-dir paper/review_data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


MODELS_ORDER = [
    "qwen3-vl-ft-r32-2ep",
    "qwen3-vl-ft-r32-1ep",
    "qwen3-vl-ft-r64-1ep",
    "qwen3-vl-base",
    "granite-3.3-8b-qiskit",
    "qwen2.5-coder-14b-qiskit",
]

BENCHMARK_DIRS = {
    "qhe": "qiskit-humaneval",
    "qhe_passk": "qiskit-humaneval-passk",
    "qhe_hard": "qiskit-humaneval-hard",
    "synth": "synthetic",
}


def load_latest_json(directory: Path, slug: str) -> dict | None:
    """Find the most recent JSON matching *slug* under directory."""
    candidates = sorted(directory.glob(f"*_{slug}_*.json"), reverse=True)
    if not candidates:
        return None
    with open(candidates[0], encoding="utf-8") as f:
        return json.load(f)


def safe_round(v: Any, digits: int = 4) -> Any:
    return round(v, digits) if isinstance(v, float) else v


def fmt_pct(v: float | None) -> str:
    return f"{v*100:.2f}" if v is not None else "-"


def extract_metrics(data: dict) -> dict:
    """Pull the metrics we care about from a result JSON."""
    m = data.get("metrics", {})
    overall = m.get("overall", m)
    out: dict[str, Any] = {}
    for key in ("pass@1", "pass@5", "pass@10", "rouge_l", "bleu",
                "avg_pass_rate", "execution_accuracy", "success_rate"):
        if key in overall:
            out[key] = safe_round(overall[key])

    for prefix in ("by_type.", "by_category."):
        for k, v in m.items():
            if k.startswith(prefix) and isinstance(v, dict):
                out[k] = {kk: safe_round(vv) for kk, vv in v.items()}

    for mod_key in ("text_only", "multimodal"):
        if mod_key in m and isinstance(m[mod_key], dict):
            out[mod_key] = {kk: safe_round(vv) for kk, vv in m[mod_key].items()}

    return out


def build_model_results(results_dir: Path) -> dict[str, dict[str, Any]]:
    """For each model, load all benchmark results into a nested dict."""
    all_results: dict[str, dict[str, Any]] = {}

    for model_name in MODELS_ORDER:
        model_data: dict[str, Any] = {"model": model_name}

        for slug, bench_dir_name in BENCHMARK_DIRS.items():
            bench_path = results_dir / bench_dir_name / model_name
            if not bench_path.is_dir():
                continue
            data = load_latest_json(bench_path, slug)
            if data is None:
                continue
            model_data[slug] = extract_metrics(data)

        # Image masking (R1.1)
        mask_path = results_dir / "image_masking" / f"{model_name}.json"
        if mask_path.exists():
            with open(mask_path) as f:
                model_data["image_masking"] = json.load(f)

        # Stratified QHE-Hard (R2.7)
        strat_path = results_dir / "stratified" / f"{model_name}.json"
        if strat_path.exists():
            with open(strat_path) as f:
                model_data["stratified"] = json.load(f)

        all_results[model_name] = model_data

    return all_results


def render_main_table(results: dict) -> str:
    """Table 1: Main results (QHE, QHE-Hard, Synthetic overall)."""
    lines = [
        "## Table 1: Main Results (Pass@1)\n",
        "| Model | QHE | QHE-Hard | S-FC | S-CG | S-QA (ROUGE-L) | Text | MM |",
        "|-------|-----|----------|------|------|----------------|------|-----|",
    ]
    for name in MODELS_ORDER:
        d = results.get(name, {})
        qhe = d.get("qhe", {}).get("pass@1")
        qhe_h = d.get("qhe_hard", {}).get("pass@1")
        synth = d.get("synth", {})
        fc = synth.get("by_type.function_completion", {}).get("pass@1")
        cg = synth.get("by_type.code_generation", {}).get("pass@1")
        qa = synth.get("by_type.qa", {}).get("rouge_l")
        text = synth.get("text_only", {}).get("pass@1")
        mm = synth.get("multimodal", {}).get("pass@1")
        lines.append(
            f"| {name} | {fmt_pct(qhe)} | {fmt_pct(qhe_h)} | {fmt_pct(fc)} | "
            f"{fmt_pct(cg)} | {fmt_pct(qa)} | {fmt_pct(text)} | {fmt_pct(mm)} |"
        )
    return "\n".join(lines)


def render_passk_table(results: dict) -> str:
    """Table 2: Pass@k on QHE (R1.4)."""
    lines = [
        "\n## Table 2: Pass@k on Qiskit HumanEval (R1.4)\n",
        "| Model | Pass@1 | Pass@5 | Pass@10 |",
        "|-------|--------|--------|---------|",
    ]
    for name in MODELS_ORDER:
        pk = results.get(name, {}).get("qhe_passk", {})
        lines.append(
            f"| {name} | {fmt_pct(pk.get('pass@1'))} | "
            f"{fmt_pct(pk.get('pass@5'))} | {fmt_pct(pk.get('pass@10'))} |"
        )
    return "\n".join(lines)


def render_image_masking_table(results: dict) -> str:
    """Table 3: Image-masking ablation (R1.1)."""
    lines = [
        "\n## Table 3: Image-Masking Ablation on Multimodal Subset (R1.1)\n",
        "| Model | Pass@1 (with image) | Pass@1 (masked) | Delta |",
        "|-------|---------------------|-----------------|-------|",
    ]
    for name in MODELS_ORDER:
        im = results.get(name, {}).get("image_masking", {})
        ov = im.get("overall", {})
        wi = ov.get("pass_at_1_with_image")
        mk = ov.get("pass_at_1_masked")
        delta = ov.get("mean_delta")
        lines.append(
            f"| {name} | {fmt_pct(wi)} | {fmt_pct(mk)} | "
            f"{'+' if delta and delta > 0 else ''}{fmt_pct(delta)} |"
        )
    return "\n".join(lines)


def render_image_masking_by_type(results: dict) -> str:
    """Table 3b: Image-masking delta by question type."""
    lines = [
        "\n## Table 3b: Image-Masking Delta by Question Type\n",
        "| Model | FC Delta | CG Delta | QA Delta |",
        "|-------|----------|----------|----------|",
    ]
    for name in MODELS_ORDER:
        im = results.get(name, {}).get("image_masking", {})
        by_type = im.get("by_question_type", {})
        fc = by_type.get("function_completion", {}).get("mean_delta")
        cg = by_type.get("code_generation", {}).get("mean_delta")
        qa = by_type.get("qa", {}).get("mean_delta")
        if fc is None and cg is None and qa is None:
            continue
        lines.append(
            f"| {name} | {fmt_pct(fc)} | {fmt_pct(cg)} | {fmt_pct(qa)} |"
        )
    return "\n".join(lines)


def render_stratified_table(results: dict) -> str:
    """Table 4: QHE-Hard stratified (R2.7)."""
    all_diffs = defaultdict(dict)
    all_cats = set()
    for name in MODELS_ORDER:
        strat = results.get(name, {}).get("stratified", {})
        by_diff = strat.get("by_difficulty", {})
        by_cat = strat.get("by_category", {})
        for bucket, stats in by_diff.items():
            all_diffs[name][f"diff:{bucket}"] = stats.get("pass_at_1")
        for bucket, stats in by_cat.items():
            all_diffs[name][f"cat:{bucket}"] = stats.get("pass_at_1")
            all_cats.add(f"cat:{bucket}")

    diff_keys = sorted(k for k in set().union(*(d.keys() for d in all_diffs.values())) if k.startswith("diff:"))
    cat_keys = sorted(all_cats)

    lines = ["\n## Table 4: QHE-Hard Stratified Pass@1 (R2.7)\n"]

    if diff_keys:
        cols = " | ".join(k.replace("diff:", "") for k in diff_keys)
        lines.append(f"### By Difficulty\n\n| Model | {cols} |")
        lines.append("|-------|" + "|".join(["-------"] * len(diff_keys)) + "|")
        for name in MODELS_ORDER:
            vals = " | ".join(fmt_pct(all_diffs.get(name, {}).get(k)) for k in diff_keys)
            lines.append(f"| {name} | {vals} |")

    if cat_keys:
        cols = " | ".join(k.replace("cat:", "") for k in cat_keys)
        lines.append(f"\n### By Category\n\n| Model | {cols} |")
        lines.append("|-------|" + "|".join(["-------"] * len(cat_keys)) + "|")
        for name in MODELS_ORDER:
            vals = " | ".join(fmt_pct(all_diffs.get(name, {}).get(k)) for k in cat_keys)
            lines.append(f"| {name} | {vals} |")

    return "\n".join(lines)


def render_synth_by_category(results: dict) -> str:
    """Table 5: Synthetic dataset by category."""
    all_cats: set[str] = set()
    for name in MODELS_ORDER:
        synth = results.get(name, {}).get("synth", {})
        for k in synth:
            if k.startswith("by_category."):
                all_cats.add(k.replace("by_category.", ""))
    cats = sorted(all_cats)
    if not cats:
        return ""

    lines = [
        "\n## Table 5: Synthetic Pass@1 by Category\n",
        "| Model | " + " | ".join(c[:20] for c in cats) + " |",
        "|-------|" + "|".join(["-------"] * len(cats)) + "|",
    ]
    for name in MODELS_ORDER:
        synth = results.get(name, {}).get("synth", {})
        vals = []
        for c in cats:
            bucket = synth.get(f"by_category.{c}", {})
            vals.append(fmt_pct(bucket.get("success_rate")))
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def render_granite_comparison(results: dict) -> str:
    """Table 6: granite-3.3-8b-qiskit vs others (R1.3/R2.3)."""
    lines = [
        "\n## Table 6: Size-Matched Comparison — granite-3.3-8b-qiskit (R1.3/R2.3)\n",
        "| Model | Params | QHE P@1 | QHE-Hard P@1 | Synth Text P@1 |",
        "|-------|--------|---------|--------------|----------------|",
    ]
    param_map = {
        "qwen3-vl-ft-r32-2ep": "8B+90M",
        "qwen3-vl-ft-r32-1ep": "8B+90M",
        "qwen3-vl-ft-r64-1ep": "8B+180M",
        "qwen3-vl-base": "8B",
        "granite-3.3-8b-qiskit": "8B",
        "qwen2.5-coder-14b-qiskit": "14B",
    }
    for name in MODELS_ORDER:
        d = results.get(name, {})
        qhe = d.get("qhe", {}).get("pass@1")
        qhe_h = d.get("qhe_hard", {}).get("pass@1")
        text = d.get("synth", {}).get("text_only", {}).get("pass@1")
        lines.append(
            f"| {name} | {param_map.get(name, '?')} | {fmt_pct(qhe)} | "
            f"{fmt_pct(qhe_h)} | {fmt_pct(text)} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    results = build_model_results(args.results_dir)

    # Save machine-readable JSON
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "analysis.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Wrote {args.out_dir / 'analysis.json'}")

    # Render markdown tables
    sections = [
        "# Review Experiment Results — Consolidated Analysis\n",
        render_main_table(results),
        render_passk_table(results),
        render_image_masking_table(results),
        render_image_masking_by_type(results),
        render_stratified_table(results),
        render_synth_by_category(results),
        render_granite_comparison(results),
    ]
    md = "\n\n".join(s for s in sections if s)
    with open(args.out_dir / "analysis_tables.md", "w") as f:
        f.write(md + "\n")
    print(f"Wrote {args.out_dir / 'analysis_tables.md'}")

    # Print summary to stdout
    print("\n--- Quick Summary ---")
    for name in MODELS_ORDER:
        d = results.get(name, {})
        qhe = d.get("qhe", {}).get("pass@1")
        qhe_h = d.get("qhe_hard", {}).get("pass@1")
        im = d.get("image_masking", {}).get("overall", {}).get("mean_delta")
        print(f"  {name:30s}  QHE={fmt_pct(qhe):>6s}  QHE-H={fmt_pct(qhe_h):>6s}  "
              f"ImgDelta={fmt_pct(im):>7s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
