# ADR 002: DeviceContext for centralized device placement

| Field | Value |
|-------|-------|
| Status | Accepted (ForgeScore modernization WO-018) |
| Date | 2026-05-26 |
| Work order | WO-018 |

## Context

Transformers models contain thousands of scattered ``.to(device)`` calls. Implicit device
placement makes multi-GPU and dtype policies hard to audit and can cause redundant host/device
transfers. The PRD calls for a single boundary abstraction at model entry points.

## Decision

1. Introduce ``DeviceContext`` in ``src/transformers/utils/device_context.py`` with:
   - explicit device (and optional dtype),
   - ``move()`` for tensors, dicts, and lists,
   - ``move_module()`` for ``nn.Module``,
   - nested context support.

2. **New contributions** should wrap public ``forward`` / ``generate`` boundaries in
   ``DeviceContext`` (or ``device_context()`` helper).

3. **Existing models** may keep direct ``.to()`` during transition; no mass migration in WO-018.

4. Multi-GPU: one ``DeviceContext`` per replica/process; do not share a context across ranks.
   Use Accelerate/FSDP for sharding; DeviceContext only normalizes local device placement.

## Alternatives considered

| Alternative | Why not chosen |
|-------------|----------------|
| Global thread-local device | Hidden magic; breaks explicit debugging |
| Mandatory refactor of all `.to()` | Too large for one release |
| Only documentation | No enforceable pattern for new code |

## Consequences

- Positive: clearer boundaries; easier static review of device policy
- Negative: small overhead vs raw ``.to()`` (target &lt;1% in micro-benchmarks)
- Migration guide: WO-019; reference tests in ``tests/utils/test_device_context.py``

## Validation

- Unit tests without GPU
- Optional: Bert forward wrapped in DeviceContext in a follow-up PR
