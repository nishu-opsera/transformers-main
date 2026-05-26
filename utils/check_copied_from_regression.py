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
"""Warn on new '# Copied from' annotations in PR diffs (WO-016)."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULAR_DOCS_URL = "https://huggingface.co/docs/transformers/modular_transformers"
COPIED_FROM_ADDED_RE = re.compile(r"^\+\s*#\s*Copied from\s+(.+?)\s*$", re.IGNORECASE)
DIFF_FILE_HEADER_RE = re.compile(r"^\+\+\+ b/(.*)$")


@dataclass
class NewCopiedFromAnnotation:
    file_path: str
    source_reference: str
    diff_line: str


def parse_added_copied_from_lines(diff_text: str) -> list[NewCopiedFromAnnotation]:
    """Parse a unified diff and return newly added '# Copied from' comment lines."""
    findings: list[NewCopiedFromAnnotation] = []
    current_file: str | None = None
    for line in diff_text.splitlines():
        file_match = DIFF_FILE_HEADER_RE.match(line)
        if file_match:
            path = file_match.group(1)
            current_file = path if path != "/dev/null" else None
            continue
        if current_file is None or not current_file.endswith(".py"):
            continue
        copied_match = COPIED_FROM_ADDED_RE.match(line)
        if copied_match:
            findings.append(
                NewCopiedFromAnnotation(
                    file_path=current_file,
                    source_reference=copied_match.group(1).strip(),
                    diff_line=line,
                )
            )
    return findings


def _git_diff(base_ref: str, head_ref: str) -> str:
    result = subprocess.run(
        ["git", "diff", f"{base_ref}...{head_ref}", "-U0", "--", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git diff failed: {result.stderr.strip()}")
    return result.stdout


def _format_warning(findings: list[NewCopiedFromAnnotation]) -> str:
    lines = [
        f"WARNING: {len(findings)} new '# Copied from' annotation(s) detected in this change.",
        "New model contributions must use modular files as the canonical paradigm.",
        f"See: {MODULAR_DOCS_URL}",
        "",
    ]
    for item in findings[:25]:
        lines.append(f"  - {item.file_path}: {item.source_reference}")
    if len(findings) > 25:
        lines.append(f"  ... and {len(findings) - 25} more")
    lines.append("")
    lines.append(
        "Set COPIED_FROM_BLOCK_NEW=1 to treat new annotations as a blocking failure (WO-016)."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        default=os.environ.get("COPIED_FROM_BASE_REF", "origin/main"),
        help="Git ref for the merge base side (default: origin/main)",
    )
    parser.add_argument(
        "--head-ref",
        default=os.environ.get("COPIED_FROM_HEAD_REF", "HEAD"),
        help="Git ref for the PR branch (default: HEAD)",
    )
    parser.add_argument(
        "--diff-file",
        type=Path,
        help="Read diff from a file instead of invoking git (for tests).",
    )
    args = parser.parse_args()

    if args.diff_file is not None:
        diff_text = args.diff_file.read_text(encoding="utf-8")
    else:
        diff_text = _git_diff(args.base_ref, args.head_ref)

    findings = parse_added_copied_from_lines(diff_text)
    if not findings:
        print("No new '# Copied from' annotations detected.")
        return 0

    message = _format_warning(findings)
    print(message, file=sys.stderr)
    block = os.environ.get("COPIED_FROM_BLOCK_NEW", "0").lower() in {"1", "true", "yes"}
    return 1 if block else 0


if __name__ == "__main__":
    raise SystemExit(main())
