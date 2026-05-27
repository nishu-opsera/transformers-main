# GitHub Actions disabled on this fork

Upstream Hugging Face workflows (GPU self-hosted runners, Docker image builds,
nightly benchmarks, doc builds, etc.) are **not runnable** on `nishu-opsera/transformers-main`
and were generating failure notification emails on every push to `main`.

Workflow definitions are preserved under [`.github/workflows-disabled/`](../workflows-disabled/)
for reference. Run modernization checks locally instead:

```bash
make check-import-linter
make check-layer-violations
make check-modular-conversion
make check-downstream-compat
```

To re-enable a workflow, move its YAML back into this directory and ensure triggers
and secrets match your environment.
