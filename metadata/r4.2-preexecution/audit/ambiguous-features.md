# Ambiguous features

**Status:** audit proposal, not an approved R4.2 execution manifest.

## `semantic_timing_relation_count`

- Source: `IoT_auth_simulator/ml/evaluation/trace_features.py:269` and helpers at lines 415–452; assignment at line 512.
- It reconstructs challenge-delimited attempts, selects nonce-received/response-sent pairs with equal nonces, computes one signed timestamp margin per pair, and returns only `len(margins)`.
- Its output does **not** depend on the sign, magnitude, threshold satisfaction, or boundary distance of those margins. Changing timestamps while retaining finite values and event/nonce structure cannot change this count.
- Nevertheless, its pair selection is specific to the modeled Timing relation and therefore is not wholly generic context.

The audit provisionally assigns it to **F3 (trace/event context)** with the secondary tags **direct Timing; contextual**. Treating it as F1 is an equally implementable alternative and changes candidate-list membership/counts. Approval requires a scientific classification decision; the audit does not make that decision.

No other feature has an unresolved primary assignment based on its implementation.
