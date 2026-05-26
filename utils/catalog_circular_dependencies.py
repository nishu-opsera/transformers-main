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
"""Build the circular dependency catalog for WO-006 (documentation only).

Reproducible analysis:
  PYTHONPATH=src python utils/catalog_circular_dependencies.py
  PYTHONPATH=src python utils/generate_importlinter_config.py  # refresh .ci baselines first
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import grimp

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_CYCLES_PATH = REPO_ROOT / ".ci" / "known_import_cycles.json"
CI_BASELINE_PATH = REPO_ROOT / ".ci" / "top_level_import_baseline.json"
CATALOG_JSON_PATH = REPO_ROOT / ".ci" / "circular_dependency_catalog.json"
DOC_PATH = REPO_ROOT / "docs" / "architecture" / "circular_dependencies.md"
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def top_level_module(module: str) -> str | None:
    if not module.startswith("transformers."):
        return None
    parts = module.split(".")
    if len(parts) < 2:
        return None
    return f"transformers.{parts[1]}"


def collect_top_level_edges(graph: grimp.ImportGraph) -> set[tuple[str, str]]:
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
    return edges


def tarjan_scc(adj: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    onstack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    result: list[list[str]] = []

    def strongconnect(vertex: str) -> None:
        nonlocal index
        indices[vertex] = index
        lowlink[vertex] = index
        index += 1
        stack.append(vertex)
        onstack.add(vertex)
        for neighbor in adj.get(vertex, ()):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlink[vertex] = min(lowlink[vertex], lowlink[neighbor])
            elif neighbor in onstack:
                lowlink[vertex] = min(lowlink[vertex], indices[neighbor])
        if lowlink[vertex] == indices[vertex]:
            component: list[str] = []
            while True:
                node = stack.pop()
                onstack.remove(node)
                component.append(node)
                if node == vertex:
                    break
            if len(component) > 1:
                result.append(sorted(component))

    for node in sorted(adj):
        if node not in indices:
            strongconnect(node)
    return sorted(result, key=len, reverse=True)


def importers_of(edges: set[tuple[str, str]]) -> dict[str, set[str]]:
    importers: dict[str, set[str]] = defaultdict(set)
    for source, destination in edges:
        importers[destination].add(source)
    return importers


def severity_label(affected_count: int) -> str:
    if affected_count > 50:
        return "critical"
    if affected_count >= 10:
        return "high"
    return "medium"


def propose_strategy(module_a: str, module_b: str) -> str:
    pair = {module_a, module_b}
    if "transformers.modeling_utils" in pair and "transformers.configuration_utils" in pair:
        return "shared_abstraction"
    if "transformers.modeling_utils" in pair or "transformers.configuration_utils" in pair:
        return "shared_abstraction"
    if "transformers.integrations" in pair or "transformers.trainer" in pair:
        return "lazy_import"
    if "transformers.utils" in pair:
        return "interface_or_lazy_import"
    return "lazy_import"


STRATEGY_DESCRIPTIONS = {
    "shared_abstraction": (
        "Extract shared types/protocols into a neutral module (e.g. `modeling_protocols.py`) "
        "that both sides import instead of importing each other."
    ),
    "interface_or_lazy_import": (
        "Introduce a `typing.Protocol` or move the import under `if TYPE_CHECKING:` "
        "so runtime import order no longer forms a cycle."
    ),
    "lazy_import": (
        "Defer the import to function scope or use lazy module attributes so the cycle "
        "exists only for type checkers, not at runtime."
    ),
}


def example_chain(graph: grimp.ImportGraph, source_top: str, dest_top: str) -> list[str]:
    """Return a short submodule chain illustrating the top-level edge source_top → dest_top."""
    candidates_source = [m for m in graph.modules if m.startswith(source_top + ".") or m == source_top]
    candidates_dest = [m for m in graph.modules if m.startswith(dest_top + ".") or m == dest_top]
    for src in sorted(candidates_source)[:40]:
        for dst in sorted(candidates_dest)[:40]:
            try:
                chain = graph.find_shortest_chain(src, dst)
            except Exception:
                continue
            if chain:
                return list(chain)
    return [source_top, dest_top]


def build_catalog() -> dict:
    graph = grimp.build_graph("transformers")
    edges = collect_top_level_edges(graph)
    edge_set = edges
    importer_map = importers_of(edges)

    cycles_doc = json.loads(CI_CYCLES_PATH.read_text(encoding="utf-8"))
    two_cycles = cycles_doc["two_module_cycles"]

    adj: dict[str, set[str]] = defaultdict(set)
    for source, destination in edges:
        adj[source].add(destination)

    sccs = tarjan_scc(adj)

    cycle_entries = []
    for entry in two_cycles:
        module_a, module_b = entry["modules"]
        forward = (module_a, module_b)
        reverse = (module_b, module_a)
        if forward not in edge_set or reverse not in edge_set:
            continue
        affected = len(importer_map[module_a] | importer_map[module_b])
        strategy = propose_strategy(module_a, module_b)
        cycle_entries.append(
            {
                "modules": [module_a, module_b],
                "import_chains": {
                    "forward": example_chain(graph, module_a, module_b),
                    "reverse": example_chain(graph, module_b, module_a),
                },
                "top_level_edges": [list(forward), list(reverse)],
                "affected_top_level_importers": affected,
                "severity": severity_label(affected),
                "resolution_strategy": strategy,
            }
        )

    cycle_entries.sort(
        key=lambda item: (-item["affected_top_level_importers"], item["modules"][0], item["modules"][1])
    )

    severity_counts = defaultdict(int)
    for item in cycle_entries:
        severity_counts[item["severity"]] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tools": {
            "grimp": grimp.__version__ if hasattr(grimp, "__version__") else "unknown",
            "commands": [
                "PYTHONPATH=src python utils/generate_importlinter_config.py",
                "PYTHONPATH=src python utils/catalog_circular_dependencies.py",
            ],
        },
        "cross_reference": {
            "known_import_cycles_json": str(CI_CYCLES_PATH.relative_to(REPO_ROOT)),
            "two_module_cycle_count": len(two_cycles),
            "catalogued_cycles": len(cycle_entries),
        },
        "strongly_connected_components": [
            {"size": len(component), "modules": component} for component in sccs
        ],
        "severity_summary": dict(severity_counts),
        "cycles": cycle_entries,
    }


def render_markdown(catalog: dict) -> str:
    lines = [
        "# Circular dependency catalog",
        "",
        "> **WO-006** — authoritative map of top-level import cycles. "
        "Regenerate with `PYTHONPATH=src python utils/catalog_circular_dependencies.py`.",
        "",
        f"_Generated: {catalog['generated_at']}_",
        "",
        "## How this document is produced",
        "",
        "1. `utils/generate_importlinter_config.py` scans the live package with **grimp** and writes "
        "`.ci/known_import_cycles.json` (49 two-module cycles, WO-001 baseline).",
        "2. `utils/catalog_circular_dependencies.py` (this catalog) enriches each cycle with severity, "
        "example submodule chains, and resolution strategies.",
        "3. CI blocks **new** two-module cycles via `utils/check_import_linter.py` (baseline ratchet).",
        "",
        "## Strongly connected components (top-level)",
        "",
        "At the top-level `transformers.*` boundary, the import graph contains one large SCC "
        f"({catalog['strongly_connected_components'][0]['size']} modules) plus isolated two-node cycles "
        "documented below. Breaking the hub modules inside the large SCC (`modeling_utils`, "
        "`configuration_utils`, `models`, `integrations`, `utils`) is the critical path for WO-007/WO-008.",
        "",
        "| SCC size | Modules (first 12) |",
        "|---------:|-------------------|",
    ]
    for scc in catalog["strongly_connected_components"][:3]:
        preview = ", ".join(f"`{m.split('.')[-1]}`" for m in scc["modules"][:12])
        if len(scc["modules"]) > 12:
            preview += f", … (+{len(scc['modules']) - 12} more)"
        lines.append(f"| {scc['size']} | {preview} |")

    lines.extend(
        [
            "",
            "## Severity summary (two-module cycles)",
            "",
            "| Severity | Count | Definition |",
            "|----------|------:|------------|",
            f"| critical | {catalog['severity_summary'].get('critical', 0)} | >50 top-level importers affected |",
            f"| high | {catalog['severity_summary'].get('high', 0)} | 10–50 importers affected |",
            f"| medium | {catalog['severity_summary'].get('medium', 0)} | <10 importers affected |",
            "",
            "## Resolution strategies",
            "",
            "Three approaches used in subsequent work orders (WO-007, WO-008):",
            "",
        ]
    )
    for key, description in STRATEGY_DESCRIPTIONS.items():
        lines.append(f"### `{key}`")
        lines.append("")
        lines.append(description)
        lines.append("")

    lines.extend(
        [
            "**Example — shared abstraction (WO-007 target):** "
            "`configuration_utils` ↔ `modeling_utils` — extract `PreTrainedConfig` protocol surface "
            "and weight-loading hooks into `transformers.modeling_protocols` so configs never import "
            "`PreTrainedModel` at module import time.",
            "",
            "**Example — `typing.Protocol`:** "
            "`integrations` ↔ `trainer` — declare callback protocols in `integrations` without importing "
            "`Trainer` until runtime inside factory functions.",
            "",
            "**Example — `TYPE_CHECKING` lazy import:** "
            "`utils` ↔ `image_utils` — move annotation-only imports under `if TYPE_CHECKING:` "
            "and quote forward references in public APIs.",
            "",
            "## Two-module cycle catalog",
            "",
            "Cross-referenced with `.ci/known_import_cycles.json`. "
            f"**{catalog['cross_reference']['catalogued_cycles']}** cycles documented.",
            "",
            "| Severity | Modules | Importers | Strategy | Example chain (forward) |",
            "|----------|---------|----------:|----------|-------------------------|",
        ]
    )

    for item in catalog["cycles"]:
        mod_a, mod_b = item["modules"]
        short = f"`{mod_a.split('.')[-1]}` ↔ `{mod_b.split('.')[-1]}`"
        chain = " → ".join(m.split(".")[-1] for m in item["import_chains"]["forward"][:5])
        if len(item["import_chains"]["forward"]) > 5:
            chain += " → …"
        lines.append(
            f"| {item['severity']} | {short} | {item['affected_top_level_importers']} | "
            f"`{item['resolution_strategy']}` | {chain} |"
        )

    lines.extend(
        [
            "",
            "## Work order mapping",
            "",
            "| Work order | Scope |",
            "|------------|-------|",
            "| WO-001 | Baseline ratchet — block new top-level edges / 2-cycles |",
            "| WO-006 | This catalog |",
            "| WO-007 | Break **critical** cycles via shared abstractions |",
            "| WO-008 | Break **high/medium** cycles via lazy imports and interfaces |",
            "",
            "## Maintainer checklist",
            "",
            "- After intentional graph changes: run `generate_importlinter_config.py`, then this script.",
            "- Confirm every entry in `.ci/known_import_cycles.json` appears in the table above.",
            "- Review resolution strategy for any new **critical** cycle before merging.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify docs/architecture/circular_dependencies.md matches generated output.",
    )
    args = parser.parse_args()

    if not CI_CYCLES_PATH.exists():
        print(f"Missing {CI_CYCLES_PATH}. Run: python utils/generate_importlinter_config.py", file=sys.stderr)
        return 1

    catalog = build_catalog()
    markdown = render_markdown(catalog)

    CATALOG_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_JSON_PATH.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.check:
        if not DOC_PATH.exists():
            print(f"Missing {DOC_PATH}. Run without --check to generate.", file=sys.stderr)
            return 1
        if DOC_PATH.read_text(encoding="utf-8") != markdown:
            print(f"{DOC_PATH} is out of date. Regenerate with catalog_circular_dependencies.py", file=sys.stderr)
            return 1
        print("Circular dependency catalog is up to date.")
        return 0

    DOC_PATH.write_text(markdown, encoding="utf-8")
    print(f"Wrote {DOC_PATH}")
    print(f"Wrote {CATALOG_JSON_PATH}")
    print(
        f"Catalogued {catalog['cross_reference']['catalogued_cycles']} cycles "
        f"(critical={catalog['severity_summary'].get('critical', 0)}, "
        f"high={catalog['severity_summary'].get('high', 0)}, "
        f"medium={catalog['severity_summary'].get('medium', 0)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
