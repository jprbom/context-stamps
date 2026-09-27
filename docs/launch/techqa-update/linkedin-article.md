# Context Stamps: testing whether a small local model can learn to use evidence better

By Prashant Jagtap

A small language model can give a convincing answer without enough evidence. It can also quote a real passage that does not answer the question. If a local learning system treats either output as a successful example, its next update may reinforce the problem.

That is the practical question behind my current work on Context Stamps: how should a local model acquire context, use it, recognise missing information and improve from independently checked outcomes?

The aim is a useful domain system that can run locally with a small model. It needs to complete tasks reliably, preserve older skills and operate within its device's latency and memory limits. A smaller prompt is valuable when it preserves the information needed for the task.

## Where the spherical context stamp fits

Context Stamps uses a 256-bit, or 32-byte, reference across eight facets: semantic, task, entity, relation, temporal, authority, policy and modality.

The reference helps route a request to external evidence and relationship information. The document does not live inside those 32 bytes. The runtime still has to resolve the reference, check permissions and versions, preserve dependencies, and supply the evidence the model needs.

For local learning, this gives me a way to track which evidence supported an outcome and whether that evidence is still valid. It does not make a model's answer true.

## What I trained and tested on my local RTX setup

I used the public IBM TechQA data for a controlled technical-question-answering experiment with a local Qwen3.5:4b reader.

The experiment used 400 fitting questions, 177 calibration questions and all 310 questions in the native public development split. Normalized duplicate questions were grouped or withheld before inference. Documents still overlap across the partitions, so this is not a source-disjoint generalisation test.

For each question, BM25 selected up to eight overlapping source windows within the declared prompt budget. Both reader interfaces received the same selected text:

• A direct interface returned an answer span.
• A quoted interface first returned source quotations, then its answer.

The runtime checked whether the quotations matched current authorised source text and whether the answer occurred within them. I compared raw answers, filtered answers and an always-abstain control.

I also fitted three small ridge policies on CPU. Each has twelve observable features and an intercept. The target represents the utility of answering rather than abstaining, using independently supplied native answerability and span labels. Three regularisation strengths and six thresholds were assessed on calibration data only.

A policy had to preserve positive-answer F1, reduce false positives and improve the combined native score. None met that rule. The experiment therefore retained the fixed source check; no learned update was activated and no language-model weights changed.

The full study recorded 1,774 measured reader requests and three separate warmups.

## What the development results actually show

There are 160 answerable and 150 unanswerable development questions. The F1 values below measure character-span overlap in the correct document, on a scale from 0 to 100. They are not percentages of answers judged correct.

DIRECT ANSWER
Answerable F1: 13.64
False positives on unanswerable questions: 145 of 150

DIRECT ANSWER WITH EXACT-SPAN FILTER
Answerable F1: 13.64
False positives: 95 of 150

QUOTED ANSWER
Answerable F1: 11.21
False positives: 137 of 150

QUOTED ANSWER WITH SOURCE CHECK
Answerable F1: 7.10
False positives: 31 of 150

The simpler exact-span filter removed 50 false positives without reducing the recorded positive F1. That is a useful engineering observation, but 95 remaining false positives is still a serious failure.

The quoted source check rejected more unsupported answers. It also lost all native overlap on 19 answerable questions that the direct exact-span control had received credit for, while gaining overlap on four.

The combined native score rises from 24.78 for the direct exact-span control to 42.05 for the checked quotations. But always abstaining scores 48.39 because nearly half this development set is unanswerable. Reporting only the combined increase would give the wrong impression of useful answering.

## The latency and token cost matter too

Across the 310 requests for each interface:

DIRECT INTERFACE
Input plus output tokens: 1,184,538
Summed request time: 505.73 seconds
Median / p95 request time: 1.53 / 2.38 seconds

QUOTED INTERFACE
Input plus output tokens: 1,260,277
Summed request time: 1,177.99 seconds
Median / p95 request time: 3.71 / 6.30 seconds

The quoted interface used 6.39% more total model tokens and 2.33 times the summed request time. The checks run after generation, so rejecting an answer cannot recover the generation cost already incurred.

Shared context compilation, warmups and fitting are reported separately. CPU preparation and tests also ran on the host. These are local request measurements, not an isolated throughput study or proof of embedded-device efficiency.

## What needs to change next

There are two distinct problems to address.

First, the selected context contains a complete native reference span for only 81 of the 160 answerable development questions. A better answer policy cannot recover evidence it never receives.

Second, the reader often fails even when the relevant span is present. The training-only audit found many altered quotations and quotations assigned to the wrong source. My next text experiment will test selecting existing span identifiers, which the runtime can resolve exactly, instead of asking the model to copy long passages.

The wider local-learning design remains: collect authorised observations, obtain independent outcomes, fit a candidate, evaluate fresh tasks and retained skills, measure the complete cost, then either keep the current version or promote with monitoring and rollback.

I am also preparing a visual-memory experiment that separates image extraction from bounded arithmetic. After correcting numeric-answer handling and lookup instructions, the latest check gets 15/15 direct answers, 15/15 memory answers and 14/15 program answers across three authored charts. The remaining program returns a valid calculation for a category absent from the chart. Charging extraction also makes the memory treatment more expensive on these small examples. These are interface checks, not ChartQA benchmark scores or model training.

Every supported local deployment should be able to improve from its own verified experience: memory and small CPU policies on modest devices, with optional adapter training on capable hardware. The failed chart program explains why the model's own output cannot be its training judge. An update must preserve older skills and justify its complete operating cost before activation.

## What is available now

The research branch contains the code, fitted policies, native target-offset projections, all scored outcomes, cost records, failure review, source fingerprints and an independent replay. The original corpus, complete questions and raw prompts remain external under their source terms.

This iteration has not established autonomous model improvement, domain AGI or universal superiority. It has established specific failure modes that the next implementation must solve without hiding quality losses behind abstention or token savings.

Repository and current research branch:
https://github.com/jprbom/context-stamps/tree/research/enterprise-context

Detailed results and local reproduction:
https://github.com/jprbom/context-stamps/blob/research/enterprise-context/evidence/techqa-local-v1/README.md

Original TechQA project:
https://github.com/IBM/techqa

[Insert the repository's TechQA research figure after the results section.]
