# CI baselines (ForgeScore modernization)

## Import graph (WO-001)

| File | Purpose |
|------|---------|
| `top_level_import_baseline.json` | Allowed directed imports between top-level `transformers.*` modules (351 edges). CI fails if a **new** edge appears. |
| `known_import_cycles.json` | Documented two-module import cycles (49). CI fails if a **new** 2-cycle appears. |

Regenerate after an intentional, reviewed graph change:

```bash
PYTHONPATH=src python utils/generate_importlinter_config.py
```

`importlinter` contracts live in `pyproject.toml` (see markers `BEGIN import-linter`).

## Layer violations (WO-002)

| File | Purpose |
|------|---------|
| `layer_violation_baseline.json` | Baseline count and fingerprints for config→modeling boundary crossings. CI fails on **new** fingerprints or a count above baseline. |

Scans `configuration_*.py`, `configuration_utils.py`, and `modular_*.py` under `src/transformers/` for uses of symbols imported from modeling modules (ForgeScore reference ~3958; current AST baseline ~3840).

Regenerate after an intentional, reviewed change:

```bash
PYTHONPATH=src python utils/check_layer_violations.py --write-baseline
```
