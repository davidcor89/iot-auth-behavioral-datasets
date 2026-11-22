# Source reference index

**Status:** R4.2 read-only audit proposal; not an approved execution manifest.  
**Checkpoint:** `7ad3284f4c45f793c3f7ee10a678c3f73109bf53`  
**R4.1 aggregate:** `sha256:cce473976d8ea079d5ad77031da21a2062a02aaae6121ce451ff0db49eacea34`

| Repository-relative path | SHA-256 | Relevant locations |
|---|---|---|
| `IoT_auth_simulator/ml/evaluation/trace_features.py` | `33de12151cd8b292d7d7dd7181bf10a58fb3766efdb64b3dd81bfe399888b8bf` | contract 25–42; definitions 51–310; spec/order 313–376; attempt reconstruction 415–429; Timing relation 432–452; extraction 455–524 |
| `IoT_auth_simulator/ml/evaluation/pipeline.py` | `0f972471678dcaed844d8e8cae016037b82e2b52cdd9cb6229c4b1cd3342539f` | canonical hash 139–141; profile selection/order/fingerprint 156–175; frozen-input reconstruction 182–228; run provenance 430–490 |
| `IoT_auth_simulator/ml/icds/feature_spec_icds_v1.json` | `f425fcc081a12f20335e21692b8156260793e02829159e54ec1f4f426c8f54bb` | serialized frozen feature specification; supporting evidence, not used alone |
| `artifacts/r4.1-device-disjoint-evaluation/run-provenance.json` | `b3794a80099732871a3553edb6e60ac6add0b73df79f3aace926289fb9807fd4` | persisted 180-run feature names, profile fingerprints, and preprocessing metadata |
| `IoT_auth_simulator/simulator/event_model.py` | `fdc85cfaca1a3dd08abff17a6c332db41b3e1d7da90987eec1956357318a8737` | `EventType` 39–106; `EventResult` 164–167 |
| `artifacts/r4.1-device-disjoint-evaluation/integrity-report.json` | `c2d336ba0b9f2fc7a5875dead6aee216f1feddda3f4c30f1dc951c2f0bf04d99` | persisted feature-order/fingerprint checks |
| `metadata/r4.1-freeze/freeze-manifest.json` | `71f39843a9afbee4b1a42c198808ca52969b247f6e70558e71b8f366b0f0636d` | archival checkpoint and aggregate fingerprint construction |

All paths are relative to the authorized repository root. Line references refer to checkpoint `7ad3284f4c45f793c3f7ee10a678c3f73109bf53`.
