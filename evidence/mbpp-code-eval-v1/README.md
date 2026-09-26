# Paired code-adapter evaluation

The retained development smoke covers HumanEval/0–3 with one greedy completion
per model/task: eight local generations, seven isolated native grades and one
base output without the required function, counted as a failure before execution.
The base passes 0/4 and the MBPP-trained adapter 2/4 on combined base and full
extended tests, gaining tasks 0 and 3. There are no execution errors or remaining
containers. Four tasks are insufficient for a reliable benefit claim.

`smoke.json.gz` contains both plans, prompts, raw generations, formatting
outcomes, token counts, batch timings, sandbox profiles and native grades.
These tasks are a development subset of the planned 164-task run, not an
additional independent holdout. Hyperparameters remain fixed after this smoke.
No adapter is activated, and no complete benchmark score is asserted here.

The comparison uses the same Qwen2.5-1.5B-Instruct BF16 base/backend with the
trained rank-eight adapter enabled or disabled. It measures a generic local
SFT control. It does not compare Context Stamps memory with another system,
establish edge performance, or test frontier models.

HumanEval prompts retain their MIT notice; EvalPlus retains its Apache-2.0 and
embedded MIT notices in [the grader evidence](../humaneval-grader-v1/README.md).
MBPP source attribution and checkpoint terms are in the
[model card](../mbpp-code-adapter-v1/MODEL_CARD.md). Implementation copyright
Prashant Jagtap, MIT License. See [methodology](../../docs/verified-code-learning.md).
