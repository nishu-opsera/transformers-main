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
"""Generate importlinter.toml and baseline JSON from the current import graph.

Regenerates allowed top-level import edges and known two-module cycles so CI can
block new cycles while the codebase is being modernized.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import grimp

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_DIR = REPO_ROOT / ".ci"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
IMPORTLINTER_MARKER_START = "# BEGIN import-linter (WO-001, generated)"
IMPORTLINTER_MARKER_END = "# END import-linter (WO-001, generated)"
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Conceptual layering (highest to lowest — import-linter layer order).
# Higher layers may import lower layers; not the reverse.
LAYER_MODULES = [
    "transformers.trainer",
    "transformers.integrations",
    "transformers.generation",
    "transformers.models",
    "transformers.modeling_utils",
    "transformers.configuration_utils",
    "transformers.utils",
    "transformers.dependency_versions_check",
]


def top_level_module(module: str) -> str | None:
    if not module.startswith("transformers."):
        return None
    parts = module.split(".")
    if len(parts) < 2:
        return None
    return f"transformers.{parts[1]}"


def collect_top_level_edges() -> list[tuple[str, str]]:
    graph = grimp.build_graph("transformers")
    edges: set[tuple[str, str]] = set()
    for module in graph.modules:
        source = top_level_module(module)
        if source is None:
            continue
        for imported in graph.find_modules_directly_imported_by(module):
            destination = top_level_module(imported)
            if destination is None or destination == source:
                continue
            edges.add((source, destination))
    return sorted(edges)


def collect_two_module_cycles(edges: list[tuple[str, str]]) -> list[dict[str, list[str]]]:
    edge_set = set(edges)
    cycles: list[dict[str, list[str]]] = []
    seen: set[tuple[str, str]] = set()
    for source, destination in edges:
        if (destination, source) not in edge_set:
            continue
        key = tuple(sorted((source, destination)))
        if key in seen:
            continue
        seen.add(key)
        cycles.append({"modules": list(key)})
    return cycles


def write_baseline_files(edges: list[tuple[str, str]], cycles: list[dict[str, list[str]]]) -> None:
    CI_DIR.mkdir(exist_ok=True)
    baseline = {"edges": [{"source": s, "destination": d} for s, d in edges]}
    (CI_DIR / "top_level_import_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    (CI_DIR / "known_import_cycles.json").write_text(
        json.dumps({"two_module_cycles": cycles, "count": len(cycles)}, indent=2) + "\n",
        encoding="utf-8",
    )


def layer_index(module: str) -> int:
    return LAYER_MODULES.index(module)


def layer_subset_edges(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Imports from a lower layer into a higher layer (illegal for the layers contract)."""
    layer_set = set(LAYER_MODULES)
    illegal: list[tuple[str, str]] = []
    for source, destination in edges:
        if source not in layer_set or destination not in layer_set:
            continue
        # Higher index = lower in the hierarchy; lower layer must not import higher layer.
        if layer_index(source) > layer_index(destination):
            illegal.append((source, destination))
    return sorted(illegal)


def render_pyproject_snippet() -> str:
    return f"""{IMPORTLINTER_MARKER_START}
# Regenerate: python utils/generate_importlinter_config.py
# Allowed import cycles and edges: see .ci/known_import_cycles.json and
# .ci/top_level_import_baseline.json (baseline ratchet in utils/check_import_linter.py).

[tool.importlinter]
root_package = "transformers"

[[tool.importlinter.contracts]]
id = "top-level-layers"
name = "Top-level module layering (ForgeScore modernization guardrail)"
type = "layers"
layers = [
{chr(10).join(f'    "{module}",' for module in LAYER_MODULES)}
]
# Populated by baseline ratchet; layers contract documents target architecture.
ignore_imports = []

{IMPORTLINTER_MARKER_END}
"""


def update_pyproject(snippet: str) -> None:
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    if IMPORTLINTER_MARKER_START in text:
        before = text.split(IMPORTLINTER_MARKER_START)[0].rstrip()
        after = text.split(IMPORTLINTER_MARKER_END)[-1].lstrip("\n")
        text = before + "\n\n" + snippet + "\n" + after
    else:
        text = text.rstrip() + "\n\n" + snippet + "\n"
    PYPROJECT_PATH.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if generated files differ from disk (CI mode).",
    )
    args = parser.parse_args()

    edges = collect_top_level_edges()
    cycles = collect_two_module_cycles(edges)
    snippet = render_pyproject_snippet()

    if args.check:
        text = PYPROJECT_PATH.read_text(encoding="utf-8")
        if snippet not in text:
            print("pyproject.toml import-linter section is out of date.", file=sys.stderr)
            print("Run: python utils/generate_importlinter_config.py", file=sys.stderr)
            return 1
        return 0

    write_baseline_files(edges, cycles)
    update_pyproject(snippet)
    print(
        f"Updated {PYPROJECT_PATH} and .ci baselines "
        f"({len(edges)} top-level edges, {len(cycles)} two-module cycles)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
