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
