# MBPP training-source qualification

This directory records execution of the **published training split**, IDs
601–974. It is not a model evaluation. No test/validation solution enters a
training batch. The source and license inventory is in `source.json`.

The complete evidence contains 374 native reference checks and 374 empty-program
controls in fresh non-root offline containers. `records.json.gz` preserves raw
execution results, parsed assertions, source fingerprints, rejected examples
and cleanup. `summary.json` reports eligible IDs and failures; `plan.json`
records the protocol before execution. The recorded AST audit uses host Python
3.12; AST byte representations need not match another Python version.

`source-training-rows.json.gz` is the upstream training subset, reordered by
task ID and serialized as JSON. It retains prompts, code, setup and assertions.
Eligibility filtering is recorded separately; source records are not repaired
or silently replaced. Offline replay verifies the recorded evidence. Download
the pinned originals and rerun the container checks to independently reproduce
execution and the reserved-split overlap audit.

Dataset attribution: Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma,
Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le,
and Charles Sutton, *Program Synthesis with Large Language Models* (2021).
Source: [Google Research MBPP](https://github.com/google-research/google-research/tree/f46ca8374b4cddef97ca4208ad986049d74d296a/mbpp).
License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as identified
in the retained official Hugging Face dataset card. Filtering and serialization
are modifications by this experiment. No upstream endorsement is implied.

Context Stamps implementation: copyright Prashant Jagtap, MIT License. The
repository license does not relicense the dataset. Passing three published
assertions does not prove general correctness, adversarial verifier security,
model improvement or production readiness.
