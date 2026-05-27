# Architecture Decision Records (ADRs)

This directory captures **significant, durable architectural decisions** for the Hugging Face
Transformers modernization program (ForgeScore). ADRs explain *why* the codebase is structured the
way it is—not just *what* changed in a given PR.

## When to write an ADR

Create or update an ADR when a decision:

- Affects multiple modules, teams, or work orders
- Is hard to reverse without breaking public API or downstream packages
- Introduces a new pattern contributors are expected to follow
- Has meaningful trade-offs that future maintainers need to understand

Skip ADRs for routine bug fixes, single-file refactors, or decisions that are obvious from code
comments alone.

## Numbering

| Range | Purpose |
|-------|---------|
| `001`–`099` | Foundational / cross-cutting decisions (this modernization program) |
| `100+` | Domain- or feature-specific decisions (reserve as needed) |

`002` — device and accelerator context (WO-018).

Use the next free number in the series. Do not renumber accepted ADRs; mark superseded records
with status **Superseded by ADR NNN** and link forward.

## Lifecycle

1. **Proposed** — Draft in a PR; link the work order (e.g. WO-020).
2. **Accepted** — Merged after review; becomes the source of truth for that decision.
3. **Deprecated / Superseded** — Still listed for history; new work follows the replacement ADR.

Status values: `Proposed`, `Accepted`, `Deprecated`, `Superseded by ADR NNN`.

## How to add an ADR

1. Copy [`TEMPLATE.md`](./TEMPLATE.md) to `NNN-short-slug.md` (three-digit prefix, kebab-case slug).
2. Fill in **Context**, **Decision**, **Alternatives considered**, and **Consequences**.
3. Add **Validation** and **References** when they help reviewers or CI owners.
4. Link the new ADR from related docs (e.g. `docs/architecture/rollback_procedures.md`).
5. Update the index below.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [001](./001-init-decomposition.md) | Domain sub-registry architecture for `transformers.__init__` | Accepted (WO-009) |
| [002](./002-device-context.md) | DeviceContext for centralized device placement | Accepted (WO-018) |
| [003](./003-lazy-module-import-pattern.md) | Lazy public API via `_LazyModule` | Accepted (WO-020) |
| [004](./004-modular-file-code-generation.md) | Modular file → generated modeling artifacts | Accepted (WO-020) |
| [005](./005-auto-class-registry-pattern.md) | `Auto*` class registry pattern | Accepted (WO-020) |
| [006](./006-safetensors-over-pickle.md) | Prefer safetensors over pickle for weights | Accepted (WO-020) |
| [007](./007-trust-remote-code-gate.md) | `trust_remote_code` security gate | Accepted (WO-020) |

## Related documentation

- [`docs/architecture/circular_dependencies.md`](../circular_dependencies.md)
- [`docs/architecture/rollback_procedures.md`](../rollback_procedures.md)
