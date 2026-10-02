# Coding-domain context and local learning: next evaluation

By Prashant Jagtap. Research protocol and current status, 2 October 2026.

The first MBPP adapter did not improve aggregate HumanEval+ correctness and
increased generation cost. A smaller second adapter improved that development
set but was tuned after reviewing its failures; neither is the Context Stamps
contribution. A [fresh repository-disjoint localization study](../evidence/repoqa-localization-v1/README.md)
then found **13/30 dense versus 5/30 stamp top-1** on held-out Python
repositories. That result tests a compact routing path only. It is not a
code-editing, authorized compiler, reader, or agent-workflow result. The
compact route remains inactive.

## What the completed localization gate changes

The study pinned the RepoQA source release and MiniLM encoder, indexed 11,164
functions from ten repositories, and held three repositories aside for final
measurement. It compared identical candidate sets under dense, TF–IDF and
256-bit multi-facet routing. The stamp stores 32 bytes per function but missed
more targets and was slower at this small-scale warm lookup in the measured
implementation. The runner and all 100 per-query rankings are published
without redistributing repository source code or raw descriptions.

The validation gate failed before any fit of a new CPU selector or reader
adapter. Fitting on this failed route would spend training budget without
evidence that the information needed by a coding reader survived retrieval.
No new model was activated. The inspected final cohort cannot be reused as an
untouched test for a revised stamp. The next candidate must address candidate
recall on train/validation, then use a new final group.

An exploratory [train-only ITQ quantizer follow-up](../evidence/repoqa-localization-v1/README.md)
raised validation top-1 to 9/30 for a 256-bit *single semantic* code, versus
14/30 dense. This is a quantization diagnostic, not the four-facet design or
a final-set gain. It was stopped before the inspected final repositories.

An additional 32-byte product-quantization diagnostic tied dense at **14/30
validation top-1**, with **21/30 versus 22/30 top-10** and lower MRR. It uses
a float query and a shared 393 KB codebook, so it is not the binary stamp and
may be unattractive for a small single-repository edge index. Repeated warm
Faiss lookups were slower than flat dense at this scale. The result merits a
new preregistered, independently held-out larger-corpus trial; it does not
qualify deployment or justify changing the default route.

That follow-up has now been run on a [new CodeSearchNet Python-derived cohort](../evidence/codesearchnet-quantization-v1/README.md).
Thirty held-out repositories supplied 90 exact-function queries, with
docstrings removed from indexed code. Dense reached **52/90 top-1**; ordinary
32-byte PQ reached **50/90**, OPQ **51/90**, and a validation-selected
query-weighted PQ **47/90**. All compact routes were slower in measured warm
CPU lookup. The query-weighted route also needed about **1.57 MB** of shared
codebook and transform storage, which exceeded the per-function vector saving
for every selected repository. The validation gain did not generalize, so no
compact semantic route qualifies. The cohort is now inspected and cannot be
reused as an untouched test for another tuned candidate.

A preregistered [matching-aware follow-up](../evidence/codesearchnet-matching-v1/README.md)
trained a rank-16 query adapter against quantized document reconstructions on
6,315 training pairs. It reached **59/90 top-1** on the existing validation
repositories, versus **63/90 dense**, and did not pass its quality gate. It
did not open the new 40-repository, 120-query cohort. Lower training loss
alone was not enough to repair quantization. The next useful design should
keep the 32-byte stamp as a handle and test a separately stored, more precise
residual or verified workflow path with full memory and latency accounting.

The [precision-residual test](../evidence/codesearchnet-residual-v1/README.md)
has since opened that cohort: SQ8 direct and dense both reached **81/120 top-1**,
with SQ8 using about one quarter of the pooled vector-index bytes. A 32-byte
PQ route with SQ8 reranking also tied top-1 but was much slower. An extended
CPU profile found no stable SQ8 latency advantage. This establishes a narrower
storage-versus-latency option for a host-owned index, not a self-contained
32-byte memory or a verified coding-agent outcome. The cohort is now inspected.

A subsequent [native coding-context development gate](../evidence/terminal-coding-development-v1/README.md)
used two selected Terminal-Bench 2.1 tasks. A broad vulnerability prompt caused
dense and 256-bit routes to disagree in the top three, so context injection
abstained before model calls. On a second task with an explicit source file and
two declared dependencies, Qwen2.5-Coder 7B failed the native verifier with
both direct-file and relation-closure context. The closure used more model input
tokens in both the 16-step shell-agent comparison and a one-shot code-generation
diagnostic. The former exposed repeated failed shell commands; the latter
exposed an incorrect input path even when the correct file was present in the
packet. These tasks are now inspected development data. Do not fit to their
reference solutions or treat their verifier failures as a final-set gain.

Before another quantizer fit, add a host-verified, concise file/path contract
and a fail-fast action loop. Compare it with the same contract available to a
strong direct-file baseline, so any observed gain can be attributed to context
routing or relationship activation rather than extra instructions. Freeze new
task families and a model of sufficient coding competence before inspecting
their native outcomes. Count full packet tokens, index construction, repeated
tool observations, agent wall time and native pass rate; retain regressions.

The path-contract implementation is `context_stamps.coding_contract`. It binds
an explicit source allowlist to on-disk SHA-256 revisions, renders exact runtime
input/output paths, and catches disallowed *literal* `/app/...` paths in Python
syntax before submission. It does not verify computed paths, file permissions,
program correctness or the contents of a 256-bit stamp. Native execution in an
isolated verifier remains the outcome check. The inspected-task replay script
`experiments/terminal_modern_contract.py` gives the same contract and the same
three source files to direct and stamp arms under one complete-prompt ceiling.

The subsequent [minimal-schema development test](../evidence/terminal-coding-schema-development-v1/README.md)
passed one already inspected native task with the exact CSV header in either
direct or stamp-bound form. The identical view gave identical model prompts and
responses; there was no stamp-specific quality gain. The [predeclared next task](../evidence/terminal-coding-schema-fresh-v1/README.md)
failed native grading in all four frozen arms, including direct full source and
both schema views. Later runtime repair and generic pattern tuning also failed.
This removes any basis for claiming a general coding gain from the development
pass. A successful baseline model/task pair is now a prerequisite for a useful
paired context comparison. The 30B local model did not fit the current GPU/host
memory; its attempted load is not a benchmark result.

Keep quantization in the next coding milestone as a **measured retrieval route**,
after a capable direct-context coding baseline has passed fresh native tasks.
The first candidate is a 32-byte multi-facet coarse key with a host-held SQ8
residual. Compare direct dense, direct SQ8, and stamp-to-SQ8, each with the same
eligibility, source revision, role filter, path contract and final prompt budget.
Use a deterministic retrieval-margin rule: if quantized scores cannot separate
the kth candidate from a competitor under a proven residual-error bound, fetch
the exact vectors or abstain. The existing `ResidualIndex` implements conservative
SQ8 bounds and exact refinement; the next experiment should log how often that
fallback fires. Try lower precision or learned bit allocation only on training
repositories, with full shared-codebook, backing-store, update and resolver
costs. Admit a smaller route only if fresh-task native pass rate is at least the
strong direct baseline and observed p95 end-to-end latency, full tokens or total
edge storage improve without a material regression. A 256-bit route must never
be described as storing the source, residual or authorization itself.

## Task unit and controls

Use public, redistributable repositories with pinned commits and independent
tests. Split by repository and task family before fitting a selector; keep a
sealed final partition and an older-task retention partition. A task should
include a user request, the repository state available before the change, an
authorized evidence scope, and an independently executable outcome check.
Do not train from an evaluated task's reference patch or hidden tests.

For each frozen local model, compare four paired arms in randomized order:

1. The same task prompt with no retrieved repository context.
2. A strong conventional retrieval baseline under the same complete-prompt
   token budget.
3. The authorized Context Stamps compiler resolving versioned source and
   dependency closure into that same budget.
4. An oracle-upper-bound diagnostic with the necessary source locations
   supplied by the host; this is not a deployable arm.

The 256-bit stamp is an identity and retrieval handle. It cannot contain an
entire codebase or prove the retrieved content is true. Measure index build,
storage, updates, resolution, authorization checks, model input/output tokens,
time to first token, total wall time, peak RAM/VRAM and native task pass rate.
Count abstentions, false positive evidence, stale source references, regressions
and incomplete edits as failures. Compare against full-context and simple
cache baselines when those fit the same device budget. Report cold and warm
paths separately.

## Candidate policy and retention

Train any retrieval or route policy only on the training repositories. Admit
only externally checked task outcomes; a model's self-reported success is not
a label. Fit a small CPU policy before another model-weight update. Retain the
frozen dense and direct baselines, and stop if the candidate cannot improve
quality or cost on independent validation. A low-rank model update may then be
compared both with and without the same compiled context. That factorial test
separates a weight effect from a context effect.

Before local activation, require the prespecified quality floor, no material
new-failure rate, complete cost accounting, unchanged older-domain outcomes,
and a reversible version in the local registry. The present MBPP/HumanEval+
and two inspected Terminal-Bench tasks are development controls; reusing them
does not create fresh validation evidence. Keep an explicit failure register
for import/entry-point errors, incorrect boundary cases, inefficient loops,
missing repository facts, stale evidence and unsafe tool requests.

## Device-task extension

Start with one OS and simulated, permission-scoped desktop tasks before
generalizing to Windows, macOS, Linux, Android and iOS. Distinguish planning,
observation, action and external verification. Never treat a screen caption,
successful command exit or generated action as proof of a completed user goal.
Publish device data and learned adapters only when rights and privacy checks
permit; private experience stays local by default. A useful edge result needs
measurements on the target class of hardware, including startup, memory,
battery/energy and recovery from interrupted workflows.

The [verified code-learning guide](verified-code-learning.md) gives the current
RTX training setup. The [Terminal-Bench development pilot](terminal-local-pilot.md)
records two already inspected coding-agent tasks. Neither establishes
cross-platform device competence or autonomous model self-improvement.
