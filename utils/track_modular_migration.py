#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
"""Track top-50 Copied-from churn files and modular migration status (WO-017)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / ".ci" / "copied_from_catalog.json"
OUTPUT = REPO_ROOT / ".ci" / "modular_migration_progress.json"
DOC = REPO_ROOT / "docs" / "architecture" / "modular_migration_progress.md"


def _modular_path(modeling_path: str) -> Path | None:
    p = Path(modeling_path)
    if not p.name.startswith("modeling_"):
        return None
    model_dir = REPO_ROOT / p.parent
    mod = model_dir / f"modular_{model_dir.name}.py"
    return mod if mod.exists() else None


def build_progress(top_n: int = 50) -> dict:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    per_file: Counter[str] = Counter()
    for entry in catalog["entries"]:
        fp = entry["file_path"]
        if fp.startswith("src/transformers/models/") and "modeling_" in fp:
            per_file[fp] += 1

    entries = []
    before_total = 0
    after_notes = 0
    for fp, count in per_file.most_common(top_n):
        before_total += count
        mod = _modular_path(fp)
        is_auto = False
        full = (REPO_ROOT / fp)
        if full.exists():
            is_auto = "automatically generated from" in full.read_text(encoding="utf-8", errors="ignore")[:800]
        status = "modular_source" if mod else "pending_modular"
        if mod and is_auto:
            status = "modular_codegen"
        entries.append(
            {
                "file_path": fp,
                "annotations_before": count,
                "modular_file": str(mod.relative_to(REPO_ROOT)) if mod else None,
                "status": status,
            }
        )

    wave1_migrated = (REPO_ROOT / ".ci" / "modular_migration_wave1_files.txt").read_text(encoding="utf-8").strip().splitlines() if (
        REPO_ROOT / ".ci" / "modular_migration_wave1_files.txt"
    ).exists() else []

    return {
        "version": 1,
        "top_n": top_n,
        "entries": entries,
        "wave1_migrated_files": wave1_migrated,
        "summary": {
            "files_tracked": len(entries),
            "wave1_migrated_count": len(wave1_migrated),
            "annotations_in_top_n": before_total,
            "modular_source_count": sum(1 for e in entries if e["modular_file"]),
            "target_annotation_reduction": 150,
        },
    }


def write_doc(report: dict) -> None:
    lines = [
        "# Modular migration progress (WO-017)",
        "",
        f"Tracking top **{report['top_n']}** `modeling_*.py` files by `# Copied from` count.",
        "",
        "| File | Annotations (baseline) | Modular source | Status |",
        "|------|------------------------|----------------|--------|",
    ]
    for row in report["entries"]:
        mod = row["modular_file"] or "—"
        lines.append(f"| `{row['file_path']}` | {row['annotations_before']} | `{mod}` | {row['status']} |")
    lines.extend(
        [
            "",
            "Regenerate:",
            "",
            "```bash",
            "python utils/track_modular_migration.py --write",
            "python utils/catalog_copied_from.py  # refresh annotation counts",
            "```",
            "",
        ]
    )
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = build_progress()
    if args.write:
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        write_doc(report)
        print(f"Wrote {OUTPUT} and {DOC}")
    else:
        print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
