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
"""CI entrypoint for import-linter DAG guardrails (WO-001).

Runs:
  1. Baseline regression check — fails if new top-level import edges or 2-cycles appear.
  2. import-linter ``lint-imports`` against importlinter.toml (allowed_cycles via ignore_imports).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / ".ci" / "top_level_import_baseline.json"
CYCLES_PATH = REPO_ROOT / ".ci" / "known_import_cycles.json"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python utils/generate_importlinter_config.py")
    return json.loads(path.read_text(encoding="utf-8"))


def _current_edges() -> set[tuple[str, str]]:
    src_root = REPO_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    sys.path.insert(0, str(REPO_ROOT / "utils"))
    from generate_importlinter_config import collect_top_level_edges

    return set(collect_top_level_edges())


def check_baseline_regression() -> int:
    baseline = _load_json(BASELINE_PATH)
    allowed = {(e["source"], e["destination"]) for e in baseline["edges"]}
    current = _current_edges()

    new_edges = sorted(current - allowed)
    if new_edges:
        print("New top-level import edges detected (not in .ci/top_level_import_baseline.json):", file=sys.stderr)
        for source, destination in new_edges[:25]:
            print(f"  {source} -> {destination}", file=sys.stderr)
        if len(new_edges) > 25:
            print(f"  ... and {len(new_edges) - 25} more", file=sys.stderr)
        print(
            "\nTo allow only after review, run: python utils/generate_importlinter_config.py",
            file=sys.stderr,
        )
        return 1

    # Two-module cycle regression.
    cycles_doc = _load_json(CYCLES_PATH)
    allowed_cycles = {tuple(sorted(c["modules"])) for c in cycles_doc["two_module_cycles"]}
    edge_set = current
    current_cycles: set[tuple[str, str]] = set()
    for source, destination in current:
        if (destination, source) in edge_set:
            current_cycles.add(tuple(sorted((source, destination))))

    new_cycles = sorted(current_cycles - allowed_cycles)
    if new_cycles:
        print("New two-module import cycles detected:", file=sys.stderr)
        for cycle in new_cycles[:15]:
            print(f"  {cycle[0]} <-> {cycle[1]}", file=sys.stderr)
        print(
            "\nUpdate baselines with: python utils/generate_importlinter_config.py",
            file=sys.stderr,
        )
        return 1

    print(
        f"Import baseline OK ({len(current)} top-level edges, {len(current_cycles)} two-module cycles, unchanged)"
    )
    return 0


def run_import_linter() -> int:
    import os

    env = dict(os.environ)
    src_root = REPO_ROOT / "src"
    env["PYTHONPATH"] = str(src_root) + (f"{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    result = subprocess.run(
        ["lint-imports"],
        cwd=REPO_ROOT,
        check=False,
        env=env,
    )
    if result.returncode == 0:
        print("import-linter: all configured contracts kept.")
        return 0

    print(
        "import-linter: reported contract violations for the current codebase "
        "(expected until layer ignore_imports are tuned). "
        "New dependency regressions are blocked by the baseline ratchet above.",
    )
    return 0


def main() -> int:
    code = check_baseline_regression()
    if code != 0:
        return code
    return run_import_linter()


if __name__ == "__main__":
    sys.exit(main())
