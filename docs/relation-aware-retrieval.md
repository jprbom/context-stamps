# Relation-aware spherical routing

## Implemented boundary

The 256-bit spherical capsule is a lossy routing identity. It does not serialize a document, a context graph or a model state. Source evidence, exact identifiers, permissions, versions and typed relationships remain in an external resolver.

The current architecture has four stages:

1. Compile observed semantic, task, entity, relation, temporal, authority, policy and modality facets.
2. Convert declared typed links into a bounded relation vector.
3. Fit and quantize each facet independently, then concatenate exactly 256 bits.
4. Use the capsule to activate candidates and recover authorized evidence through an exact or precise retriever.

![Spherical context workflow](assets/spherical-context-cover.png)

The [animated SVG](assets/spherical-context-flow-animated.svg) shows the same path. The [interactive workflow](assets/spherical-context-workflow.html) provides guided views for capsule construction, retrieval and relationships.

## Relation map

For a directed weighted edge from node (i) to node (j), the implementation forms a row-normalized transition matrix (P). Starting at root (r), bounded personalized diffusion is

\[
p^{(0)} = e_r,\qquad
p^{(t+1)} = \rho e_r + (1-\rho)P^\top p^{(t)},
\]

where (0 < \rho < 1) is the restart probability and (t) is a caller-bounded hop count. The reference implementation limits the graph to 1,000 nodes, 4,096 edges, 32 hops and a 4,096-dimensional output.

Each reached node and each typed forward or reverse edge contributes a signed, deterministically hashed feature. The final vector is L2-normalized. Direction and relationship type therefore affect the feature vector, while collisions and diffusion make it lossy. `RelationMap.revision` binds the ordered canonical edge set; an edge update changes the revision.

This is distinct from `ContextGraph`. `ContextGraph` resolves complete declared dependency closures and provenance. `RelationMap` only creates a bounded vector for ranking. A stamp match cannot prove that a dependency exists or is current.

## Product-sphere quantization

For each facet (f), training rows are normalized to the unit sphere. The ITQ family learns a center \(\mu_f\) and an orthogonal rotation \(R_f\). A facet vector (x_f) produces

\[
b_f = \operatorname{sign}\left((\hat{x}_f-\mu_f)R_f\right),
\qquad
\hat{x}_f = \frac{x_f}{\lVert x_f\rVert_2}.
\]

The capsule is the concatenation

\[
B = b_{semantic}\,\Vert\,b_{task}\,\Vert\,b_{entity}\,\Vert\,b_{relation}\,\Vert
b_{temporal}\,\Vert\,b_{authority}\,\Vert\,b_{policy}\,\Vert\,b_{modality},
\]

subject to \(\sum_f |b_f|=256\). Independent rotations avoid treating unrelated facet coordinates as one Euclidean space. They do not make 256 bits lossless. The standard allocation is 96/32/32/32/16/16/16/16 bits and remains an unoptimized research profile.

ITQ, feature hashing and personalized diffusion are established mechanisms. The research contribution under evaluation is their integration with exact identity, evidence versioning, scope-bound validation and precise fallback.

## Precise residual path

The implemented precise hybrid score is

\[
s(d,q)=\alpha z(s_{dense}(d,q))+(1-\alpha)z(s_{BM25}(d,q)),
\]

where population z-scores are computed over the eligible corpus and \(\alpha=0.75\) was selected from the SciFact validation partition. This path is not contained in the 256-bit stamp.

The frozen blend improved nDCG@10 on SciFact, NFCorpus and ArguAna, then regressed on the prospective SciDocs check. A `ScopeCertificate` therefore enables the profile only when the paired bootstrap lower confidence bound on a disjoint validation set exceeds the declared minimum gain. Missing or failed certificates use dense scores.

This is a no-regression policy decision within a validated scope. It is not a theorem of universal superiority, and a certificate must not be copied to a different corpus, encoder revision, authorization filter or metric.

## Agent and edge use

The capsule can be useful when a workflow repeatedly asks which context region should activate before fetching current evidence:

- a coding agent can route among requirement, implementation, test, configuration and ownership facets;
- a research workflow can bind a claim to experiment, dataset, model revision and failure evidence;
- a local small model can exchange 32-byte resolver keys between steps while the evidence store remains on device;
- an enterprise workflow can reuse version-checked evidence packets while enforcing authorization outside the stamp.

The reference Python code is suitable for offline experiments and bounded local stores. It is not yet a distributed index, a secure capability token, a mobile energy result or a model-internal attention layer.

## Security controls

- Treat stamps as potentially linkable identifiers; do not use them as encryption or authorization.
- Supply the eligible source IDs after authentication and authorization.
- Bind certificates to corpus, encoder, schema, facet mask, weights, metric and retriever revision.
- Verify source and relation revisions before reusing a resolved packet.
- Treat retrieved text, code and metadata as untrusted model input.
- Limit graph size, hops, vector dimensions and score-array length before processing attacker-controlled requests.
