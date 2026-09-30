# Coding-domain context and local learning: next evaluation

By Prashant Jagtap. Research protocol; the comparisons below have not run.

The first MBPP adapter did not improve aggregate HumanEval+ correctness and
increased generation cost. A smaller second adapter is a retention experiment,
not the Context Stamps contribution. The next useful test must isolate what the
context system contributes to a coding task with missing, changing repository
information.

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
