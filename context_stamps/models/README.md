# Experimental bundled context scorers

Author: Prashant Jagtap. Original fictional training data and coefficients: MIT.

These three four-view nonnegative pairwise ranking models use lexical hashing
dimension 64, 64 bits per view and projection seeds 17/41/83. Ordered views are
content, entity, intent, task; each successive view increments the seed by one.
They require exact family identity compatibility and do not accept arbitrary
semantic embeddings. They are not generative models or safety classifiers.

Training uses 120 procedural queries, eight candidates each, 840 pairs and 400
steps. A separate 40-query validation split selects regularization. Fresh testing
uses 200 queries and 200 shifted-task queries, with supplied exact metadata and
a shared grammar. The small mean improvement over ridge is not established by
the exploratory paired intervals. Exact-field matching is a perfect control on
these particular fixtures. Do not infer general language understanding.

Full provenance, failures, validation trials and weights are retained under
`evidence/pairwise-v1` in the source repository. Loading is explicitly opt-in via
`load_experimental_model(seed=17)`. Calibrate any activation threshold independently
and enforce exact metadata, authorization, freshness and dependency checks outside
the model.
