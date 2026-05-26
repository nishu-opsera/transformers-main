#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Catalog all '# Copied from' annotations (WO-015)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / ".ci" / "copied_from_catalog.json"
SCAN_ROOTS = (
    REPO_ROOT / "src" / "transformers",
    REPO_ROOT / "tests",
    REPO_ROOT / "examples",
    REPO_ROOT / "utils",
    REPO_ROOT / "benchmark",
)

COPIED_FROM_RE = re.compile(r"^\s*#\s*Copied from\s+(.+?)\s*$", re.IGNORECASE)
CLASS_RE = re.compile(r"^(\s*)class\s+(\w+)")
DEF_RE = re.compile(r"^(\s*)def\s+(\w+)")


@dataclass
class CopiedFromEntry:
    file_path: str
    line_number: int
    source_reference: str
    target_name: str | None
    annotation_type: str
    has_modifications: bool
    migration_complexity: str


def _estimate_complexity(source_reference: str, has_modifications: bool) -> str:
    if has_modifications:
        return "complex"
    lowered = source_reference.lower()
    if lowered.startswith("http") or "github.com" in lowered or "://" in lowered:
        return "complex"
    if "transformers." in lowered or lowered.startswith("loss.") or lowered.startswith("fairseq"):
        return "simple"
    return "medium"


def _infer_target(lines: list[str], start_index: int) -> tuple[str | None, str]:
    class_indent: int | None = None
    for back in range(start_index, max(-1, start_index - 20), -1):
        class_match = CLASS_RE.match(lines[back])
        if class_match:
            class_indent = len(class_match.group(1))
            break
    for offset in range(1, min(40, len(lines) - start_index)):
        line = lines[start_index + offset]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        class_match = CLASS_RE.match(line)
        if class_match:
            class_indent = len(class_match.group(1))
            return class_match.group(2), "class"
        def_match = DEF_RE.match(line)
        if def_match:
            indent = len(def_match.group(1))
            if class_indent is not None and indent > class_indent:
                return def_match.group(2), "method"
            return def_match.group(2), "function"
        if stripped.startswith(("@", "class ", "def ")):
            break
    return None, "unknown"


def scan_file(path: Path, *, repo_root: Path | None = None) -> list[CopiedFromEntry]:
    root = repo_root or REPO_ROOT
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.name
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    entries: list[CopiedFromEntry] = []
    for index, line in enumerate(lines, start=1):
        match = COPIED_FROM_RE.match(line)
        if not match:
            continue
        source = match.group(1).strip()
        has_modifications = "with changes" in line.lower() or "with change" in source.lower()
        target_name, annotation_type = _infer_target(lines, index - 1)
        entries.append(
            CopiedFromEntry(
                file_path=rel,
                line_number=index,
                source_reference=source,
                target_name=target_name,
                annotation_type=annotation_type,
                has_modifications=has_modifications,
                migration_complexity=_estimate_complexity(source, has_modifications),
            )
        )
    return entries


def _git_churn_by_file() -> dict[str, int]:
    result = subprocess.run(
        ["git", "log", "--pretty=format:", "--name-only", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {}
    counts: Counter[str] = Counter()
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.endswith(".py"):
            counts[line] += 1
    return dict(counts)


def build_catalog() -> dict:
    entries: list[CopiedFromEntry] = []
    files_scanned = 0
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            files_scanned += 1
            entries.extend(scan_file(path))

    churn = _git_churn_by_file()
    file_counts = Counter(entry.file_path for entry in entries)
    priority_files = [
        path
        for path, _count in sorted(
            ((path, churn.get(path, 0)) for path in file_counts),
            key=lambda item: item[1],
            reverse=True,
        )[:50]
    ]

    type_counts = Counter(entry.annotation_type for entry in entries)
    complexity_counts = Counter(entry.migration_complexity for entry in entries)

    return {
        "version": 1,
        "entries": [asdict(entry) for entry in entries],
        "summary": {
            "total_annotations": len(entries),
            "total_files": len(file_counts),
            "files_scanned": files_scanned,
            "annotation_types": dict(type_counts),
            "migration_complexity": dict(complexity_counts),
            "priority_migration_files": priority_files,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Write catalog JSON here (default: .ci/copied_from_catalog.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the catalog on disk differs from a fresh scan.",
    )
    args = parser.parse_args()

    catalog = build_catalog()
    fresh_json = json.dumps(catalog, indent=2) + "\n"

    if args.check:
        if not args.output.exists():
            print(f"Missing {args.output}. Run without --check to generate.", flush=True)
            return 1
        if args.output.read_text(encoding="utf-8") != fresh_json:
            print(f"Catalog drift detected in {args.output}. Regenerate with: python utils/catalog_copied_from.py")
            return 1
        print(f"Catalog OK ({catalog['summary']['total_annotations']} annotations).")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fresh_json, encoding="utf-8")
    summary = catalog["summary"]
    print(
        f"Wrote {args.output} — {summary['total_annotations']} annotations "
        f"in {summary['total_files']} files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
