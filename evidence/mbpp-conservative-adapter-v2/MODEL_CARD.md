# Conservative local coding candidate

By Prashant Jagtap. Research candidate, inactive.

This is a 4,372,840-byte LoRA adapter for the pinned
[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
base. It is a second, deliberately smaller supervised update after the first
adapter tied the base on 164 HumanEval+ tasks while adding 19 regressions.
That inspected comparison is development evidence; the second candidate must
not be described as independently validated on HumanEval+.

The local RTX 5080 Laptop run used the same 368 qualified and decontaminated
MBPP training records as the first adapter. It changed the seed to 11, used
one epoch instead of two, and reduced the learning rate from 0.0001 to
0.00002. Rank 8, scaling 16, BF16, SDPA, batch size 4, assistant-only loss,
and the frozen base are unchanged. The run completed 92 optimizer steps in
39.09 seconds with 7,879,181,312 peak allocated CUDA bytes. It did not alter
the base fingerprint. The adapter SHA-256 is
`1d1db2050272d1067cbc25cdd883a23573db5327e503b4b666f9d0503a65f8eb`.
This combined recipe changes three choices at once; a score difference cannot
be attributed to any one of them.

Training loss and an unchanged base are not evidence of improved coding.
The [paired development comparison](../mbpp-conservative-comparison-v2/README.md)
reports 64/164 native passes versus 49/164 for the contemporaneous base, but
also 12 new failures and higher token/time cost. The candidate remains disabled.
It has no Context Stamps memory treatment, independent retention result,
device-control qualification, edge runtime measurement or production approval.

The source MBPP examples are CC BY 4.0, credited to Austin et al.,
*Program Synthesis with Large Language Models* (2021). The dataset revision,
license card, exact exclusions and independent assertion qualification are in
[the training-source evidence](../mbpp-training-v1/README.md). The Qwen base
and adapter weights retain the Qwen Apache-2.0 notice in this directory.
Original experiment code and this report are copyright Prashant Jagtap under
the repository MIT license. Upstream attribution is not an endorsement.

For a reproducible local fit, use the command in
[the code-learning guide](../../docs/verified-code-learning.md), replacing
`experiments/mbpp_adapter.py` with `experiments/mbpp_conservative_adapter.py`
and selecting a fresh output directory. The model, source and qualification
files must match the pinned hashes in `plan.json`; the adapter is not a
standalone base model. Do not load arbitrary local checkpoints or execute
model-generated programs outside the isolated grader.
