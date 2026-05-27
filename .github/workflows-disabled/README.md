# Disabled GitHub Actions (upstream HF CI)

These workflows were moved from `.github/workflows/` to stop automatic runs and
failure emails on the `nishu-opsera/transformers-main` fork.

They expect Hugging Face infrastructure (self-hosted A10/MI/GAUDI runners, internal
Docker registries, org secrets, etc.) and are unrelated to the Forge modernization
guardrails in this branch.

Restore individual files only if you configure matching runners and secrets.
