# Repository working instructions

- This repository contains the Phase 1 research design plus tested execution foundations in `src/vglab`. It is not a measured reconstruction stack: SHARP, Depth Pro 2, and official protocol runs have not been executed. Preserve the distinction between proposed, implemented, tested, and measured. `configs/phase1_protocol.json` stays `execution_enabled=false` until a real runner is verified.
- The active scope is Phase 1 in `docs/PHASE1_SPEC.md`. Do not silently add pose optimization, dynamic reconstruction, learned global tokens, product integration, or unrelated work.
- Primary language for project documentation is Chinese. Keep equations, units, experiment IDs, and source citations precise.
- Before changing a technical assumption, read `docs/DATA_CONTRACT.md`, `docs/EXPERIMENT_PROTOCOL.md`, and `docs/DECISIONS.md`.
- Use explicit dependency imports; do not introduce global library namespaces such as `THREE.*`.
- Store no user capture, credentials, model weights, prediction caches, or large generated scenes in Git. Keep provenance and hashes instead. Respect upstream license conditions.
- Separate source frames from validation/test frames before any network inference or cross-frame context construction. Never use a model's own prediction as independent geometric ground truth.
- Keep SHARP-only controls free of Depth Pro 2 outputs. Record failed runs and protocol changes. Never fabricate results or mark planned work complete.
- Design files are not executable experiments. Only add a launch command when its implementation exists and is verified for its stated purpose.
