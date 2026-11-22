# Timing equivalence and redundancy audit

**Status:** audit proposal, not an approved R4.2 execution manifest. No model results were inspected.

## Direct predicate computation

`semantic_timing_min_signed_causal_margin` is the sole feature that returns the modeled signed margin. `_attempt_segments` begins a segment at each `challenge_sent`; `_timing_relations` matches each `nonce_received` to the first later unused `response_sent` with the same nonce and computes

`response_sent.observed_timestamp - nonce_received.observed_timestamp`.

The feature is the minimum over those reconstructed relations. The extractor itself does not threshold the margin; it exposes sign and boundary distance directly.

## Other Timing-conditioned evidence

`semantic_timing_relation_count = len(margins)` uses the same reconstructed relations. It does not evaluate the sign or magnitude and is not computationally equivalent to the predicate, but its event-pair selection is Timing-specific. This is the only family-assignment ambiguity.

## Generic temporal overlap

The seven F2 features summarize total duration and all actual inter-event delays. They do not select the modeled nonce/response relation and are not generally algebraically equivalent to the minimum signed causal margin. They share timestamps or exported delays with F1.

Under the additional frozen-grammar condition that a matched `response_sent` immediately follows its `nonce_received`, the corresponding causal margin equals that row's `observed_delay_since_previous_event`. If it is also the smallest gap in the trace, `observed_inter_event_delay_min` equals the direct F1 value. These are conditional near-equivalences, not general code identities. The other delay summaries and duration can retain correlated boundary evidence.

## Context confounding

Event-type count/frequency pairs are algebraically linked by `frequency = count / trace_event_count`. Result count/frequency pairs have the same relation. Counts of challenge/nonce/response/retry events, trace length, and `semantic_timing_relation_count` encode generator-specific attempt/recovery structure. Removing F1 or F2 therefore does not remove all evidence associated with how Timing variants were generated.

## Nonce overlap

The five generic nonce multiplicity features ignore attempt boundaries. The two direct Nonce features reconstruct challenge-delimited attempts. Generic multiplicity is not generally equivalent to cross-attempt sharing because legitimate propagation/within-attempt recurrence also repeats values. In constrained traces with one occurrence per attempt and no legitimate within-attempt recurrence, repetition can become near-equivalent to cross-attempt reuse; that requires assumptions not enforced by these generic summaries.

## Causal interpretation

The proposed ablations are executable, but full causal attribution to independent evidence families is not warranted: F1/F2 share temporal evidence, F3 retains generator-structure proxies, and F4 generic/direct Nonce features overlap under restricted grammars.
