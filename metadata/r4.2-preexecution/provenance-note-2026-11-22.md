# R4.2 pre-execution provenance note — 2026-11-22

This namespace materializes the approved R4.2 A/B/C/D feature contracts and a deterministic plan for 120 **unexecuted** C/D runs.

- Repository checkpoint: `7ad3284f4c45f793c3f7ee10a678c3f73109bf53` on `research/r4.1-scientific-freeze`.
- Frozen R4.1 aggregate: `sha256:cce473976d8ea079d5ad77031da21a2062a02aaae6121ce451ff0db49eacea34`.
- A and B are immutable references to the frozen R4.1 C1/C2 runs. They must not be retrained or overwritten.
- C and D are planned only. No estimator was constructed, fitted, or used for prediction while producing this evidence.
- The source audit was copied byte-for-byte from `/tmp/r4.2-feature-contract-audit/` into `audit/`; its reports were not rewritten.
- `semantic_timing_relation_count` is retained as F3 context with secondary tag `Timing-relation structural evidence`.
- This note and the accompanying manifests do not authorize ML execution.
