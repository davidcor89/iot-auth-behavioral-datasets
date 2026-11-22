# R4.1 scientific freeze metadata

This directory binds the frozen R4.1 execution to portable input copies,
validation evidence, the exact executed implementation, and a complete
file-level freeze manifest.

`freeze-manifest.json` defines the aggregate fingerprint as SHA-256 over the
UTF-8 canonical JSON encoding of its sorted `authoritative_inventory` array,
using sorted object keys and separators `,` and `:` with no whitespace. Each
record contains exactly `path`, `bytes`, and `sha256`. The manifest itself,
the aggregate-fingerprint field, and any final archival Git commit identifier
are excluded from that digest input, preventing circularity.

The external evidence root is a local preservation locator, not a universally
portable path. Repository-relative mirrors in `inputs/` are authoritative
byte-identical archival copies for fresh-clone verification.
