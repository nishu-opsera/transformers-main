# ADR 004: Modular file → generated modeling artifacts

| Field | Value |
|-------|-------|
| Status | Accepted (ForgeScore modernization WO-020) |
| Date | 2026-05-26 |
| Work order | WO-020 |

## Context

Transformers historically followed a **one model, one file** policy: each architecture duplicated
layers, attention blocks, and docstrings across hundreds of `modeling_*.py` files. Maintenance
cost scales with model count, and drift between similar architectures (Llama, Mistral, Qwen, etc.)
is common.

The **modular** workflow introduces `modular_<model>.py` as the editable source of truth.
Contributors subclass existing classes and import shared utilities; a converter emits the
user-facing single-file modules (`modeling_*.py`, `configuration_*.py`, …) that Hub users and
`Auto*` classes expect.

## Decision

1. **Author in modular, ship generated** — New or refactored models maintain logic in
   `src/transformers/models/<name>/modular_<name>.py`. Generated files carry the
   `AUTO_GENERATED_MESSAGE` header from `utils/modular_model_converter.py` and must not be edited
   by hand.

2. **LibCST-based conversion** — `utils/modular_model_converter.py` parses modular sources with
   LibCST, maps classes to target file types (`modeling`, `configuration`, tokenization, etc.),
   resolves inheritance and relative imports, and writes deterministic output modules.

3. **CI enforcement** — CI regenerates artifacts from modular sources and fails if generated files
   drift. Manual edits to generated files are rejected in review.

4. **Backward-compatible surface** — Generated `modeling_*.py` files remain the public module
   paths for `from_pretrained`, Hub snapshots, and external libraries that import concrete classes.
   Modular files are contributor-facing, not end-user API.

5. **Escape hatch** — Models with no sensible base class may still use hand-written
   `modeling_*.py` (documented in `docs/source/en/modular_transformers.md`). Modular is the
   default for derivative architectures, not a hard requirement for every file type.

6. **CLI integration** — `transformers-cli add-new-model-like` and related tooling scaffold
   modular files when copying from an existing model (`cli/add_new_model_like.py`).

## Alternatives considered

| Alternative | Why rejected |
|-------------|--------------|
| Continue manual copy-paste only | Does not scale; high drift risk across 300+ architectures |
| Modular-only API (no generated files) | Breaks Hub layout, docs, and downstream imports expecting `modeling_*.py` |
| External codegen repo | Adds release coupling; codegen must live with the models it emits |
| Macro/template strings without AST | Fragile across Python syntax changes; LibCST preserves structure safely |

## Consequences

- **Positive:** Less duplicated code; clearer inheritance chains; single edit point for shared
  fixes; aligns contributor workflow with design docs.
- **Negative:** Contributors must run the converter locally; merge conflicts in generated files;
  debugging requires tracing generated line numbers back to modular sources.
- **Follow-up:** Expand modular coverage for tokenizers/processors where inheritance applies;
  keep `AUTO_GENERATED_MESSAGE` and CI check in sync when converter behavior changes.

## Validation

- Regenerate from modular: `python utils/modular_model_converter.py <path/to/modular_*.py>` (or
  project Makefile target) and confirm clean git diff.
- CI modular check job — no drift between modular source and generated artifacts.
- Spot-check: generated header references the correct modular path.

## References

- `utils/modular_model_converter.py` — `convert_modular_file`, `AUTO_GENERATED_MESSAGE`
- `docs/source/en/modular_transformers.md`
- `src/transformers/cli/add_new_model_like.py` — `create_modular_file`
- Example: `src/transformers/models/*/modular_*.py`
