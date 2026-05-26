# Layer violation remediation (WO-013 / WO-014)

Config-to-model layer boundaries are enforced by `utils/check_layer_violations.py` (WO-002).
This document tracks remediation status and backlog priority for future releases.

## PRD interim target

**Fewer than 1,000** config→modeling AST violations in the **runtime configuration** scope.

## Current status

| Scope | Files | Violations | CI enforced | PRD target |
|-------|-------|------------|-------------|------------|
| **Runtime configuration** | `configuration_*.py`, `configuration_utils.py` | **0** | Yes | Met |
| **Modular codegen** | `modular_*.py` under `src/transformers/models/` | ~3,700 | No (tracked) | Future (WO-017) |

WO-013 cleared all violations in runtime configuration files (moves such as `rope_config_utils.py`,
`importlib` lazy loads, and protocol-based boundaries).

WO-014 confirms the interim PRD target for **enforced** scope and publishes the modular backlog.

## Approved remediation patterns (WO-013)

1. Move execution logic from config classes into modeling modules.
2. Replace config imports of modeling symbols with `ConfigProtocol` / `ModelConfigConsumer` (`protocols.py`).
3. Extract shared config-only types to neutral modules (e.g. `rope_config_utils.py`).
4. Use `importlib.import_module(...)` for unavoidable late bindings (same as import-cycle work).
5. Do **not** refactor `modular_*.py` for layer lint alone — migrate to modular codegen (WO-017).

## Modular backlog priority

Regenerate machine-readable tiers:

```bash
make report-layer-violations
```

Priority tiers (by violation count per `modular_*.py` file):

- **P0 (≥30 violations):** highest churn; migrate first in WO-017.
- **P1 (20–29):** medium priority.
- **P2 (<20):** address in later releases or when touching the model.

See `.ci/layer_violation_tracking.json` for the full per-file list.

## CI commands

```bash
make check-layer-violations          # fails on new runtime-config violations
make report-layer-violations         # refresh tracking JSON + doc inputs
python utils/check_layer_violations.py --write-baseline  # after reviewed ratchet
```

## Historical reference

ForgeScore reference snapshot: **3,958** violations before modernization (configuration + modular combined).
Runtime configuration is now **0**; remaining count is almost entirely modular codegen sources.
