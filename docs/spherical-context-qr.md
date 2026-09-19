# Spherical Context QR

By Prashant Jagtap. Private development specification, version scqr1.

Spherical Context QR is a compact multi-facet representation of a context item.
Each supplied facet vector is normalized onto its own unit sphere, then encoded
with Gaussian random hyperplanes. Comparing stamps gives a vector of bit-agreement
scores rather than a single undifferentiated similarity score. Applications can
inspect or weight those scores for a particular task.

For independent Gaussian hyperplanes, the expected disagreement fraction is
theta/pi for vectors separated by angle theta. Finite fingerprints introduce
sampling error. Scores are neither calibrated relevance probabilities nor direct
cosine similarities. This implementation uses uncentered Gaussian projections;
learned centered projections in the historical API have different semantics.

Mathematically the representation is a **product of unit spheres**, one per view,
followed by binary quantization. It is not a three-dimensional sphere, a new QR
error-correction standard, or lossless semantic compression. “QR” names the
portable context-stamp concept. The versioned payload can be passed to a QR
renderer subject to its capacity; no visual QR decoder is included.

## Distinction from a context graph

| Mechanism | Representation | Operation | Limitation |
|---|---|---|---|
| Spherical context stamp | Named compact angular views of one item | Per-facet comparison and task-weighted retrieval | Does not establish causality, truth or dependencies |
| Context graph | Items connected by typed, explicit edges | Traversal, dependency closure and invalidation | Requires accurate, maintained relationships |
| Combined workflow | Stamps retrieve roots; declared relationships expand evidence | Retrieve, validate, assemble and hand off | Both retrieval errors and missing edges remain possible |

An isolated stamp remains useful without a graph. Conversely, the graph can use
known root IDs without any stamp. The two must be evaluated separately, including
graph-only, stamp-only and combined configurations. A combined result does not
establish that stamps outperform graph retrieval or vector search generally.

Multiple independent projection seeds of the same embedding are **directions**.
Different supplied representations, such as intent, entity and task, are **views**.
Repeating the same vector under more seeds does not create new semantic facets.
Neither construction discovers every facet automatically. Missing facets and
incompatible encoder identities are errors, not silently imputed knowledge.

## Runtime boundaries

- Source content remains in an application-owned store. A stamp cannot reconstruct it.
- Exact SHA-256 identity and declared revisions govern reuse; similarity never establishes identity.
- Dependencies are explicit and bound to both endpoint digests and revisions.
- An incomplete, inaccessible, changed or over-budget dependency closure returns no text.
- Roles are supplied by a trusted host. The library does not authenticate users.
- Application access controls must cover the store, relationship queries and exported stamps.
- Inferred relationships need review before they become hard dependencies. No automatic inference is shipped.
- Context content is untrusted model input. These checks do not solve prompt injection.

## Evaluation

The new protocol separates equal-bit representation comparisons, learned facet
scoring, dependency traversal, and complete handoff measurements. Historical BEIR
and local-model results remain evidence for the historical implementation, not
for this new representation. See `evidence/spherical-v1/protocol.json` and the
failure ledger for the scope of the new experiments.
