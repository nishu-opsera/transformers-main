# Config–model boundary protocols (WO-012)

Formal contracts live in `src/transformers/protocols.py`. They define how configuration
and modeling layers may interact without mutual imports.

## ConfigProtocol

All `PreTrainedConfig` subclasses should structurally satisfy `ConfigProtocol`:

| Capability | Purpose |
|------------|---------|
| `model_type` | Registry / AutoClass dispatch |
| `to_dict()` | Hub serialization, checkpoints |
| `to_json_string()` | JSON export |

Configuration objects remain **inert**: no calls into `modeling_*`, `Trainer`, or tensor ops.

## ModelConfigConsumer

Modeling code should treat configs through `ModelConfigConsumer` — only `model_type` and
`to_dict()` for structural checks. Deeper hyperparameter reads use plain attributes on the
concrete config instance, but must not invoke config methods defined in modeling modules.

## ModelWithConfigProtocol

Models expose `config: ConfigProtocol` and `save_pretrained(...)`. Shared helpers in
`base_abstractions.py` use these protocols to avoid `configuration_utils` ↔ `modeling_utils`
import cycles.

## Validation

- **Runtime:** `assert_config_protocol()` / `isinstance(..., ConfigProtocol)` in tests.
- **Static:** structural subtyping with `ty`, Pyright, or mypy on new code.
- **CI:** `tests/repo_utils/test_protocols.py` and `make test-protocols`.

## Remediation (WO-013+)

`modular_*.py` files are modular/codegen sources (not runtime configuration boundaries) and are
excluded from the layer linter scope. Violations in `configuration_*.py` and `configuration_utils.py`
are enforced.

Layer violations detected by `utils/check_layer_violations.py` should be fixed by:

1. Moving execution logic from config classes into modeling modules.
2. Replacing config imports of modeling symbols with `ConfigProtocol` / `ModelConfigConsumer`.
3. Using dependency injection for shared utilities instead of cross-layer calls.
