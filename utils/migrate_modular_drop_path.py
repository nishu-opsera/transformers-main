#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
"""Replace duplicated SwinDropPath copies with imports from modular_swin (WO-017)."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "transformers"

PATTERN = re.compile(
    r"# Copied from transformers\.models\.swin\.modular_swin\.SwinDropPath with SwinDropPath->(\w+)\n"
    r"class \1\(nn\.Module\):.*?"
    r"(?=\n\n(?:class |@|def [a-z]))",
    re.DOTALL,
)

IMPORT_LINE = "from transformers.models.swin.modular_swin import SwinDropPath as {alias}\n\n"


def migrate_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "modular_swin.SwinDropPath" not in text:
        return False

    def repl(match: re.Match[str]) -> str:
        alias = match.group(1)
        return IMPORT_LINE.format(alias=alias)

    new_text, count = PATTERN.subn(repl, text)
    if count == 0:
        return False
    if "from transformers.models.swin.modular_swin import" not in new_text:
        # ensure import appears once at first use site only
        pass
    path.write_text(new_text, encoding="utf-8")
    print(f"migrated {path.relative_to(REPO_ROOT)} ({count} class(es))")
    return True


def main() -> int:
    changed = 0
    for path in sorted(SRC.rglob("*.py")):
        if migrate_file(path):
            changed += 1
    print(f"Done. {changed} files updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
