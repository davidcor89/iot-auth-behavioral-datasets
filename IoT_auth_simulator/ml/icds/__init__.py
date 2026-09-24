"""Leakage-safe ICDS evaluation infrastructure.

Phase 1 contains deterministic O03 trace feature extraction. Phase 2 freezes
trace splits and C0/C1/C2 cohort membership using privileged control metadata
without inspecting O03 values. Phase 3 provides the explicitly authorized,
train-only baseline execution path; importing the package never fits or runs a
model, and full scientific execution remains a separate reviewed checkpoint.
"""
