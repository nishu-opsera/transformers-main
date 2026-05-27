# ADR 005: `Auto*` class registry pattern

| Field | Value |
|-------|-------|
| Status | Accepted (ForgeScore modernization WO-020) |
| Date | 2026-05-26 |
| Work order | WO-020 |

## Context

Users load thousands of Hub checkpoints without knowing concrete class names in advance. The
`Auto*` family (`AutoConfig`, `AutoModel`, `AutoTokenizer`, task-specific `AutoModelFor*`, etc.)
provides a stable entry point: read `config.json` → resolve `model_type` → instantiate the correct
class.

Registries must stay **complete** (every supported architecture), **lazy** (no import of all
models at startup), and **extensible** (community `register()` without forking). Domain
decomposition (ADR 001) must not fragment the mapping story Hub users rely on.

## Decision

1. **Central name maps** — `model_type` strings map to config/model class names via ordered dicts
   in `src/transformers/models/auto/auto_mappings.py` (e.g. `CONFIG_MAPPING_NAMES`,
   `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`). Maps are the single declarative source consumed by
   lazy registry objects.

2. **Lazy registry objects** — `_LazyConfigMapping` and `_LazyAutoMapping`
   (`configuration_auto.py`, `auto_factory.py`) defer `importlib.import_module` until a key is
   accessed. Module-level caches (`_modules`) avoid repeated imports per model family.

3. **`Auto*.from_pretrained` orchestration** — Auto classes load config first, select the
   concrete class from the registry, then delegate to `PreTrained*.from_pretrained`. Custom Hub
   code paths integrate via `trust_remote_code` (ADR 007).

4. **Runtime registration** — `AutoConfig.register`, `AutoModel.register`, etc. append to
   `_extra_content` on the lazy mapping for plugins and research forks. Registrations must keep
   `config_class.model_type` consistent with the registered key.

5. **CI guardrails** — `utils/check_auto.py` validates mapping completeness against the codebase
   and can refresh mapping files. New models must update auto mappings as part of contribution.

6. **Domain facades are additive** — Domain registries (ADR 001) re-export subsets of Auto
   symbols for discoverability; root Auto classes remain canonical for Hub compatibility.

## Alternatives considered

| Alternative | Why rejected |
|-------------|--------------|
| Plugin discovery via entry points only | Incomplete for in-tree models; Hub expects built-in mappings |
| Eager dict of imported classes | Import-time cost and circular dependencies scale with model count |
| String-based `eval` of class paths | Security and refactor fragility |
| One mega `AutoModel` without task variants | Loses type-safe task heads and clearer error messages |

## Consequences

- **Positive:** Stable user API across 300+ architectures; lazy loading preserves startup
  performance; `register()` supports extensions without editing core maps.
- **Negative:** Mapping tables are large and must stay synchronized; duplicate or ambiguous
  `model_type` entries cause subtle bugs; modernization splits require careful merge of fragments.
- **Follow-up:** WO-010/WO-011 domain registries must not duplicate conflicting mappings;
  characterization tests for common Auto import paths (WO-026).

## Validation

- `python utils/check_auto.py` (or CI equivalent) — mappings consistent with model modules.
- Manual: `AutoConfig.from_pretrained("bert-base-uncased")` and a task-specific
  `AutoModelForSequenceClassification.from_pretrained(...)` resolve without importing unrelated
  architectures first.
- New model PR checklist includes updated `auto_mappings.py` entries.

## References

- `src/transformers/models/auto/auto_mappings.py`
- `src/transformers/models/auto/configuration_auto.py` — `_LazyConfigMapping`, `AutoConfig`
- `src/transformers/models/auto/auto_factory.py` — `_LazyAutoMapping`, `_BaseAutoModelClass`
- `utils/check_auto.py`
- [ADR 001: init decomposition](./001-init-decomposition.md)
- [ADR 007: trust_remote_code gate](./007-trust-remote-code-gate.md)
