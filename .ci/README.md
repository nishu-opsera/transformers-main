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

## Modular conversion (WO-003)

CI job `check_modular_conversion` verifies every `modular_*.py` under `src/transformers/models/` and `examples/modular-transformers/` matches committed generated files.

```bash
make check-modular-conversion
# equivalent:
python utils/check_modular_conversion.py --check
python utils/modular_model_converter.py --check
```

On drift, CI prints the modular source path, generated file path, and a unified diff.

## Import time benchmark (WO-004)

| File | Purpose |
|------|---------|
| `import_time_baseline.json` | Baseline mean/median/p95 per import scenario (5 samples). |
| `import_time_report.json` | Latest CI run output (also stored as a CI artifact). |

```bash
make benchmark-import-time
python utils/benchmark_import_time.py --write-baseline  # after reviewed change
```

Warns (non-blocking) when mean import time exceeds baseline by more than 10%.

## Downstream compatibility (WO-005)

Smoke tests in `tests/downstream_compat/` cover 15+ import patterns (Auto classes, pipelines, Trainer, submodules).

```bash
make check-downstream-compat
```

GitHub Actions workflow `downstream_compat.yml` runs weekly (Mondays) and on `workflow_dispatch`.

## Circular dependency catalog (WO-006)

| File | Purpose |
|------|---------|
| `circular_dependency_catalog.json` | Machine-readable cycle catalog (severity, chains, strategies). |
| `../docs/architecture/circular_dependencies.md` | Human-readable catalog cross-referenced with `known_import_cycles.json`. |

```bash
make catalog-circular-dependencies
```
