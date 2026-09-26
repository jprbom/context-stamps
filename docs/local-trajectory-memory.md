# Local trajectory memory and policy learning

Context Stamps can convert a recorded local workflow into three inspectable text views: individual observations, changes between adjacent observations, and recorded action order. These views let an application retain more than isolated text chunks. They do not establish that an action caused a change, that a task succeeded, or that a model's explanation is true.

The intended learning loop is local: collect authorized observations, measure outcomes, fit a small policy, evaluate it against the current policy on separate tasks and protected older tasks, then activate only a qualified candidate. A failed candidate remains inactive. Model weights do not have to change for local memory or context selection to improve.

## Use the adapter

Install from this research branch with `python -m pip install -e ".[learn]"`. The adapter itself has no third-party dependencies; fitting a linear policy uses NumPy on the CPU.

```python
from context_stamps.trajectory import ObservedEpisode, ObservedStep, trajectory_views

episode = ObservedEpisode("session-123", "Update a preference", (
    ObservedStep(0, "Notifications: enabled", "open(settings)", "local/settings"),
    ObservedStep(1, "Notifications: disabled", "toggle(notifications)", "local/settings"),
))
views = tuple(trajectory_views(episode))
for view in views:
    print(view.channel, view.steps, view.source_revision)
```

Each view binds to the SHA-256 revision of the complete accepted episode. A change to any accepted source field changes that revision. The hash is a reference; retain the source episode locally if later verification or reconstruction is needed. The adapter does not encode an episode into 32 bytes or replace the spherical routing capsule.

The adapter does not retire older views automatically. When an episode changes, the host must invalidate or supersede the affected nodes and update the index and current ACL. A changed hash alone does not revoke a previously stored observation.

Use `view.canonical_node(tenant="local", roles=("owner",), observed_at=...)` to enter the existing typed state API. The host supplies the ACL and the time at which it ingested the view. That time describes ingestion, not a guessed date for the original browser events. State views are `OBSERVATION`; change and path views are `DERIVED_RESULT`. No verified factual claims are created. The complete offline example is [local_trajectory_memory.py](../examples/local_trajectory_memory.py).

The adapter accepts at most 512 ordered steps per episode, 2 MiB per observation and 16 MiB of observation text per episode. It preserves original text in bounded state chunks. Derived differences are lossy retrieval aids. Long chunks may omit surrounding labels; retain and consult the source before using a fragment as decisive evidence. Screenshot paths, agent thoughts and reported success labels are not ingested by the benchmark adapter.

## Learn a context policy

`context_stamps.linear_policy` fits a regularized linear utility model from paired local outcomes. Each training task needs both the existing and candidate policy's measured result and cost. A positive prediction proposes the candidate; a nonpositive prediction keeps the baseline. This is a relative-utility estimate, not a calibrated confidence or proof of correctness.

Input features must be available before answering. The benchmark experiment uses question length and vocabulary, order/change wording, the distribution of retrieved sources, overlap between views and prompt size. Answer keys, question-category labels and holdout outcomes do not enter those features. Ridge regularization reduces sensitivity to correlated features; it does not guarantee generalization.

The policy is bound to an exact host-supplied revision. A different binding returns no action. The host must authorize the action independently and use the existing [local learning registry](local-domain-learning.md) for qualified activation and rollback. Neither the fitter nor the adapter runs a background training service or activates itself.

For a host that already records paired, independently scored local outcomes:

```python
import json
from pathlib import Path
from context_stamps.linear_policy import UtilityPair, fit_linear_policy
from context_stamps.local_learning import revision

# Each record contains cluster_id, features, baseline_correct,
# candidate_correct, baseline_cost and candidate_cost. Costs here are total
# model tokens. Features must use one frozen schema and stay within [-1, 1].
records = json.loads(Path("paired_local_outcomes.json").read_text())
rows = tuple(UtilityPair(**(r | {"features": tuple(r["features"])})) for r in records)
binding = revision({"feature_schema": "local-v1", "reader": "pinned-local-reader-v1"})
proposal = fit_linear_policy(rows, binding=binding, baseline_action="state",
                             candidate_action="linked", cost_cap=6400)
# Save this proposal for evaluation; do not replace the active policy here.
```

Do not derive `correct` from the reader's confidence or agreement with its own previous answer. Use an executable task check, trusted environment outcome or reviewed human label. Include all failures and both measured alternatives. The fitter cannot authenticate that evidence on its own.

## Local RTX evaluation protocol

The development experiment uses [LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2), a public benchmark of recorded workflows in customized web and ServiceNow environments. Its environment records are benchmark-generated, not private production histories.

The small tier supplies 100 trajectories per domain. There are 294 text-only questions with native deterministic scorers. Twelve questions from each of the six domain/ability groups form a 72-question development training split; the other 222 form the held-out comparison. Both splits share the benchmark's underlying histories. This is not an independent deployment-retention test. Another 128 text questions require a judge, and 29 require images; neither group is credited in this text-only run.

All three measured arms use the same installed Qwen2.5 1.5B Q4_K_M reader: no memory, SQLite FTS5 BM25 state retrieval, and rank-interleaved state/change/path retrieval. Complete prompts are limited to 6,144 tokens and output to 256 tokens. A fitted policy chooses between the latter two arms. Its held-out choices are saved before held-out generation and scoring. The learned result reuses the measured chosen arm; it is an offline policy evaluation, not an additional set of reader calls.

The current experiment precomputes prompts. Reported generation latency does not include index construction or all retrieval/packing work. Native phrase-match credit can accept an answer containing extra or contradictory text; it is not a general factuality or safety assessment. No native leaderboard comparison is justified by this smaller reader, custom prompt or restricted evaluation subset.

This experiment exercises trajectory views, canonical source binding and local context-policy fitting. It does not benchmark 256-bit capsule search, modify transformer attention or demonstrate a weight-training benefit.

Run from the repository root, with a local environment containing NumPy, `tokenizers` and the project:

```powershell
python -m pip install -e ".[learn]" tokenizers==0.23.2 huggingface_hub==1.33.0
# Download these public data files at the registered revision.
hf download xiaowu0162/longmemeval-v2 README.md DATA_CARD.md SCHEMA.md LICENSE checksums.sha256 questions.jsonl trajectories.jsonl haystacks/lme_v2_small.json --type dataset --revision f152293e235517d504809563c833d7190b8c713b --local-dir ../longmemeval-v2-data
git clone https://github.com/xiaowu0162/LongMemEval-V2 ../longmemeval-v2-source
git -C ../longmemeval-v2-source checkout 2cc8c540bdb87fe6761629b585e727e1c4704520

$env:PYTHONPATH='.'
python experiments/lme_memory.py prepare --data ../longmemeval-v2-data --tokenizer ../qwen25-15b-train-base --scorer ../longmemeval-v2-source/evaluation/qa_eval_metrics.py --output ../lme-memory-run
python experiments/lme_memory.py run --scorer ../longmemeval-v2-source/evaluation/qa_eval_metrics.py --output ../lme-memory-run
```

The tokenizer directory must contain the pinned Qwen2.5 1.5B tokenizer. The runner requires the registered local Ollama model digest and refuses remote-backed model entries. CPU preparation and fitting precede sequential local GPU inference. Output directories must be outside the repository and new; partial or uncertain runs are retained, not silently retried. Raw histories, prepared prompts and the index remain outside Git.

For the measured search refinement, replace `lme_memory.py` with `lme_memory_optimized.py` in both commands and use a new output directory. It keeps the original plan for sparse path searches and uses tie-preserving native ranking for state/change searches. All 882 tested returned fragment lists match the original. The second lookup comparison records median 279.39 to 98.80 ms and p95 357.41 to 311.84 ms. The original optimization's p95 regression is also retained. These are local lookup results, not a new reader-quality run or a whole-pipeline latency claim. Recheck other corpus shapes and devices before adopting it.

## What still needs qualification

Useful local self-improvement needs better decisions on future work, acceptable failures on older work, total resource savings after learning cost, and reversible updates. This adapter supplies source-bound observations and a measurable policy candidate. Broad workflow competence, live environment execution, adversarial-memory resistance, image/audio/video memory, smaller-device performance and independently qualified weight adaptation remain separate milestones.

Treat imported text as untrusted data. This adapter performs no recorded action and follows no URL. It is not a prompt-injection defense or a multi-tenant search service. A host must enforce source authorization before indexing, rebuild or isolate affected indexes after access changes, and recheck current permissions before returning evidence.
