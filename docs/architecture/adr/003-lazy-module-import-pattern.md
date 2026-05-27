# ADR 003: Lazy public API via `_LazyModule`

| Field | Value |
|-------|-------|
| Status | Accepted (ForgeScore modernization WO-020) |
| Date | 2026-05-26 |
| Work order | WO-020 |

## Context

Transformers exposes 400+ public symbols from `transformers.__init__` and domain entry points.
Eagerly importing every model, tokenizer, and utility at package import time would pull in heavy
optional dependencies (Torch, vision/audio stacks, quantizers) and amplify circular-import risk
(WO-006–WO-008). ForgeScore reports extreme fan-in on the package root; any import-time work there
affects every downstream consumer.

The library therefore treats the public namespace as a **lazy facade**: names are registered up
front, but submodules load only when accessed.

## Decision

1. **`_LazyModule` as the package shell** — Replace eager `__init__` exports with
   `_LazyModule` (`src/transformers/utils/import_utils.py`). The module stores an
   `_import_structure` mapping optional backend keys → submodule → symbol sets.

2. **Attribute-driven loading** — `__getattr__` resolves a requested name by importing the
   backing submodule (`importlib.import_module`) and caching the object on the module instance.
   Subsequent access is O(1) without re-import.

3. **`TYPE_CHECKING` split** — Under `TYPE_CHECKING`, real imports and type stubs are available
   for static analysis. At runtime, `sys.modules[__name__]` is replaced with `_LazyModule` so
   importers pay no cost until first use.

4. **Backend-gated symbols** — Symbols that require optional backends (e.g. Torch, Flax) are
   registered under backend frozensets. Missing backends surface `OptionalDependencyNotAvailable`
   instead of failing the entire import.

5. **Composable structures** — Domain sub-registries (ADR 001) contribute fragments merged via
   `merge_import_structures`. Root and domain facades share the same lazy mechanism.

6. **Preservation of public paths** — Lazy loading is an implementation detail. All existing
   `from transformers import X` paths remain valid; no symbol moves to a breaking location.

## Alternatives considered

| Alternative | Why rejected |
|-------------|--------------|
| Eager imports in `__init__.py` | Unacceptable cold-start time and dependency surface; worsens circular imports |
| PEP 562 `__getattr__` on a normal module without structure map | No backend gating or submodule batching; harder to compose domain registries |
| Split into multiple PyPI packages | Breaks single-package install story and Hub `Auto*` ergonomics |
| Import hooks / meta-path finders | Opaque to debuggers and static tools; `_LazyModule` is explicit and battle-tested |

## Consequences

- **Positive:** Fast `import transformers`; optional deps loaded only when needed; enables domain
  decomposition (ADR 001) without eager root fan-out; aligns with circular-dependency guidance
  in `docs/architecture/circular_dependencies.md`.
- **Negative:** First access to a symbol pays import latency; stack traces can point through
  `__getattr__`; tooling must handle lazy modules (see `tests/test_init_imports.py`).
- **Follow-up:** Keep characterization tests green (WO-026); extend lazy facades for new domains
  (WO-010/WO-011); avoid new eager imports at module top level in model code.

## Validation

- `make test-init-imports` — root module is lazy; model submodules defer until accessed.
- `tests/downstream_compat/test_import_patterns.py` — public API registration intact.
- Manual: `import transformers` then `transformers.BertModel` triggers a single modeling import.

## References

- `src/transformers/utils/import_utils.py` — `_LazyModule`, `merge_import_structures`
- `src/transformers/__init__.py` — root lazy shell and domain aliases
- [ADR 001: init decomposition](./001-init-decomposition.md)
- `docs/architecture/circular_dependencies.md`
