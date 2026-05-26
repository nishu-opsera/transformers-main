# Rollback procedures for high-risk modernization changes

This runbook covers **WO-007**, **WO-008**, **WO-010**, and **WO-011** — structural changes that can affect
imports, checkpoints, and downstream packages. Use it when CI or canary validation fails after merging
modernization work.

Related safety nets:

- [WO-026 import characterization](../../tests/test_init_imports.py) — `make test-init-imports`
- [WO-027 config serialization](../../tests/test_config_serialization.py) — `make test-config-serialization`
- [ADR 001: init decomposition](./adr/001-init-decomposition.md)

---

## General response checklist

1. **Confirm the failure** using the detection signals for the work order (below).
2. **Stop further merges** of related modernization PRs until root cause is understood.
3. **Revert** the offending commit(s) on `main` (or release branch) using the revert steps below.
4. **Verify** post-revert CI locally before pushing.
5. **Communicate** using the communication plan template.
6. **Record** a short RCA using the template at the end of this document.

### Post-revert verification (run on the reverted branch)

```bash
make test-init-imports
make test-config-serialization
make check-downstream-compat
make check-import-linter
make benchmark-import-time
make measure-dependency-balance
```

These targets were verified on the modernization branch (`feature/forge-modernization`) during WO-028.

---

## WO-007 — Extract shared abstractions (critical cycle breaks)

**Scope:** Shared abstractions in `protocols.py`, `trainer_utils.py`, quantizers, `core_model_loading`, etc.

### Detection criteria

| Signal | Command / location |
|--------|-------------------|
| New import-linter failure | `make check-import-linter` |
| Increased known cycle count | `.ci/known_import_cycles.json` baseline drift in CI |
| Trainer / loading regressions | `pytest tests/trainer/` (subset) |
| Downstream import breakage | `make check-downstream-compat` |

### Revert steps

```bash
# Identify commits (example)
git log --oneline --grep='\[WO-007\]'

# Revert the merge commit or single WO commit
git revert <commit-sha> --no-edit

# Regenerate import baselines if the revert touches import graph tooling
python utils/generate_importlinter_config.py

make check-import-linter
make test-init-imports
```

### Communication plan

- Open a GitHub issue titled: `Rollback: WO-007 shared abstraction (import cycle work)`
- Label: `modernization`, `rollback`, `priority:high`
- Release note (if shipped): “Reverted shared abstraction refactor due to {symptom}; imports/training restored.”

---

## WO-008 — Break remaining import cycles

**Scope:** Lazy/`importlib` import edges across `configuration_utils`, `modeling_utils`, `integrations`, `utils`, etc.

### Detection criteria

| Signal | Command / location |
|--------|-------------------|
| Runtime `ImportError` / circular import on `import transformers` | Manual repro + `make test-init-imports` |
| Unexpected cycle baseline change | `make check-import-linter` |
| Model load / `from_pretrained` failure | `make test-config-serialization`, model smoke tests |
| Downstream packages fail import patterns | `make check-downstream-compat` |

### Revert steps

```bash
git log --oneline --grep='\[WO-008\]'

# Revert most recent WO-008 commit first; repeat if multiple commits
git revert <commit-sha> --no-edit

python utils/generate_importlinter_config.py
make check-import-linter
make test-init-imports
make test-config-serialization
```

If only one file is problematic, a targeted revert of that file’s commit is acceptable; still rerun the verification block above.

### Communication plan

- Issue title: `Rollback: WO-008 import cycle remediation`
- Mention affected subsystems (e.g. generation watermarking, domain registries) in the issue body.
- If users saw import-time regressions, note `make benchmark-import-time` delta in the RCA.

---

## WO-010 — NLP / Vision domain sub-registries

**Scope:** `src/transformers/_registries/`, composed root `__init__` import structure for NLP/Vision.

### Detection criteria

| Signal | Command / location |
|--------|-------------------|
| Missing symbol on `from transformers import X` | `make test-init-imports` |
| Auto* resolution failure for NLP/Vision models | `tests/test_init_imports.py` Auto mapping tests |
| Domain registry mismatch | `make test-domain-registries` |
| Import time regression > budget | `make benchmark-import-time` |

### Revert steps

```bash
git log --oneline --grep='\[WO-010\]'
git revert <commit-sha> --no-edit

make test-domain-registries
make test-init-imports
make check-downstream-compat
```

After revert, confirm `src/transformers/__init__.py` no longer depends on removed `_registries` composition (or restore prior flat `_import_structure`).

### Communication plan

- Issue title: `Rollback: WO-010 NLP/Vision domain sub-registries`
- Note whether public imports still work via root lazy module (expected yes after revert).

---

## WO-011 — Audio / Multimodal domain sub-registries

**Scope:** Audio/Multimodal registry modules merged into root `__init__`.

### Detection criteria

| Signal | Command / location |
|--------|-------------------|
| Audio/Multimodal `from transformers import …` failures | `make test-init-imports` |
| Registry orphan / unmapped model folders | `make test-domain-registries` |
| Processor / feature-extractor Auto class errors | Downstream compat tests |

### Revert steps

```bash
git log --oneline --grep='\[WO-011\]'
git revert <commit-sha> --no-edit

make test-domain-registries
make test-init-imports
make check-downstream-compat
```

### Communication plan

- Issue title: `Rollback: WO-011 Audio/Multimodal domain sub-registries`
- Cross-link WO-010 rollback issue if both domains were released together.

---

## Root cause analysis template

```markdown
## Summary
One paragraph: what broke, when detected, user impact.

## Timeline
- YYYY-MM-DD HH:MM — merged PR #
- YYYY-MM-DD HH:MM — CI / report failure
- YYYY-MM-DD HH:MM — revert PR #

## Detection
Which signal fired? Link CI job / test name.

## Root cause
Technical explanation (module, import edge, registry mapping, etc.).

## Revert
Commit SHA reverted; verification commands run (paste pass/fail).

## Follow-up
- [ ] Fix forward PR
- [ ] Add/adjust characterization test
- [ ] Update baseline (cycles, import time, goldens)
```

---

## Escalation

- **Two maintainer sign-off** required for re-landing WO-007 changes (per PRD).
- For release branches, cherry-pick revert to the release branch before patch tag.
- Do not force-push `main`; use revert commits to preserve history.
