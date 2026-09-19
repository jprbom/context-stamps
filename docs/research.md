# Method, evidence and research scope

Author: Prashant Jagtap

## Implemented method

For an embedding `x`, each random bit is `sign(w_i · x)` with Gaussian projection rows. A family records the encoder identity, dimension, bit width, seed and method. Gaussian directions are rotationally symmetric: the expected disagreement probability is the angle divided by pi. Finite codes remain approximate.

Centered families subtract a fitted mean. ITQ fits PCA and an orthogonal rotation to reduce binary quantization error. Learned means and planes are serialized into the family. Bit agreement for centered or learned codes is a ranking score; the uncentered angular estimator law must not be assumed for them.

Gaussian generation is a Python reference implementation. Persist learned/explicit projection rows and validate golden vectors when porting to another runtime. Different quantization, floating-point behavior or near-zero projections can change bits.

The application layer uses an independent exact digest for identity and caller-provided versions for freshness. It does not establish task-relevant equivalence from proximity in embedding space.

## Reproduction

```bash
python -m pip install -e ".[learn]"
python benchmarks/retrieval.py --out results.json
```

The benchmark creates normalized synthetic vectors with a shared direction, uses independent training/index/query splits, and fits only on training data. It measures recall@10 relative to exact dense cosine neighbors, both stamps-only and after a top-50 shortlist with exact reranking. Results include the seed, split sizes, data digest, code digest, family IDs and runtime versions.

The synthetic distribution is intentionally narrow. Results do not establish semantic quality, code-edit safety, task completion, production memory footprint or universal speedups. Timing includes a small workload on the machine executing it; compare hardware and methodology before comparing numbers. Exact reranking retains access to dense vectors, so it does not deliver the raw-code-only memory footprint.

The development tests exercise exact duplicate handling, material edits even under identical fingerprints, dependency invalidation, strict supplied-version checks, serialization and budget accounting. These are correctness checks, not end-to-end agent evaluations.

## What is contributed

The release combines portable projection families, exact content identity, local version/dependency tracking, inspectable context packing and a small set of interfaces. Its contribution is the implementation and evaluation surface for these mechanisms together. Binary hashing, ITQ, semantic caching and agent memory are established work.

This standalone release is derived from the Cortex research direction. Historical private-corpus experiments are not bundled and their results are not presented as measurements of this new package. The package uses its own public fixtures and corrected projection/identity behavior.

## Next experiments

1. Compare exact deduplication, lexical/BM25 retrieval, dense retrieval, binary quantization and Context Stamps on public, versioned corpora.
2. Measure independently accepted tasks, harmful context omissions, stale evidence, all model/tool tokens, prefix-cache behavior and end-to-end latency. Include low-repetition workloads where the layer may add overhead.
3. Train a specialized encoder only after identifying a baseline failure. Separate semantic-relevance and material-change labels. Split by repository/document family before generating sibling examples.
4. Explore a small context-selection model using matched omission/restoration experiments. Do not label every response or context item positive because its overall run passed.
5. Evaluate progressive binary widths and edge quantization separately, against equal memory/latency baselines. Ordinary ITQ truncation is not an established nested code.

No generative SLM fine-tune or learned controller is shipped. Projection fitting is the supported training operation. Any future model release must include data provenance, model license, complete recipe, independent test outcomes and negative results. Use `program.md` for bounded experiments; use a fresh final test set after model selection.

## Prior work

- Charikar, *Similarity estimation techniques from rounding algorithms*, STOC 2002. [Author's paper](https://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarEstim.pdf).
- Gong and Lazebnik, *Iterative Quantization: A Procrustean Approach to Learning Binary Codes*, CVPR 2011. [Paper](https://slazebni.cs.illinois.edu/publications/cvpr11_small_code.pdf).
- [Sentence Transformers embedding quantization](https://huggingface.co/blog/embedding-quantization): binary/scalar embeddings and reranking.
- [LLMLingua](https://github.com/microsoft/LLMLingua): learned prompt compression.
- [Redis semantic caching](https://redis.io/docs/latest/develop/use-cases/semantic-cache/): approximate reuse with metadata boundaries.
- [Mem0](https://github.com/mem0ai/mem0), [Hindsight](https://github.com/vectorize-io/hindsight), and [Graphiti](https://github.com/getzep/graphiti): broader agent-memory and temporal-context systems.

These references identify overlapping techniques; no benchmark superiority over these systems is claimed.
