# Synthetic retrieval benchmark

Run `python benchmarks/retrieval.py --out results.json` after installing the `learn` extra. All inputs are generated from the recorded seed; there is no external corpus.

`reference.json` records one development run on Python 3.13 / NumPy 2.5.3 / Windows. Timings are machine-specific. It compares 16-bit Gaussian, centered and ITQ codes on independent query vectors against exact cosine neighbors. The top-50 hybrid retains access to dense vectors.

The reference run has stamps-only recall@10 of 0.1703, 0.1797 and 0.1922 respectively. Centered hashing has higher hybrid recall than ITQ in this run (0.5844 vs 0.5734). This negative comparison is retained. The small synthetic benchmark is not evidence of semantic retrieval, application savings or general ITQ superiority.

Code and data digests in each report allow comparisons to distinguish an algorithm change from a changed input or evaluator. Freeze this evaluator when running projection experiments.
