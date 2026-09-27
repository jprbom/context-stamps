# Local relation learning: better selection, unresolved answering

By Prashant Jagtap

A 13-parameter context selector trained on the local RTX retains every required passage in **38/64 reserved answerable cases**, versus **17/64 for BM25** and **30/64 for a learned pointwise control**, at six passages per compact context. This is a retrieval result on one filtered MuSiQue development subset. It does not establish a useful complete-system improvement. Neither local reader qualifies the candidate for activation.

![Selection, answer quality and request cost](assets/multihop-v1.png)

## Question and method

Can a small locally trained policy follow useful links between candidate passages without a new language model? Multi-hop questions are useful here because finding one relevant paragraph is often insufficient. We compare full context, lexical BM25, a learned pointwise selector, and a learned selector that propagates query relevance through literal title mentions. All compact arms select six paragraphs; their actual token costs differ.

Each paragraph receives twelve input-only lexical and structural features. A linear score produces a normalized query restart distribution `q`. The transition matrix `P` follows a title mentioned in another paragraph and includes a self-loop; each row sums to one. Four iterations compute:

```text
h(0) = q
h(t+1) = (1 - alpha) * q + alpha * transpose(P) * h(t)
0 <= alpha <= 0.9
```

For fixed `q` and `P`, the mapping contracts L1 distance by at most `alpha`. That stabilizes the numerical recurrence; it does not prove relevant retrieval or correct reasoning. Links are lexical observations, not verified dependencies or discovered causes. The propagated mass is not a calibrated probability that an answer is correct. This changes context allocation outside the reader, not its internal transformer attention.

The pointwise control has twelve learned weights and `alpha=0`. The relation candidate learns thirteen parameters, including `alpha`; its selected value is **0.53216**. Both minimize cross entropy against the distribution over native supporting paragraphs. Native answer strings are not training features. The 256-bit stamp format is unchanged; graph edges, paragraph text, permissions and policy weights remain external.

## Data separation and training

The [MuSiQue source](https://github.com/StonyBrookNLP/musique) is pinned to revision `922ac98f19a201998dbdae6d7f2887a5258dbdeb`. We use the Full v1.0 train/development files, including paired answerable and unanswerable variants. We do not use its test split. MuSiQue material retains its [CC BY 4.0 license](MUSIQUE-LICENSE.txt) and [attribution](THIRD_PARTY_NOTICES.md).

An audit excludes exact normalized title, paragraph and question overlap with our previously downloaded SQuAD and HotpotQA corpora. Training, calibration and evaluation also share no normalized titles, exact paragraph texts or source-question IDs. Calibration and evaluation groups are source-disjoint from one another within each partition. Training groups may share sources with other training groups. These checks do not establish semantic non-overlap or absence from a base model's pretraining.

The partition contains 400 training question pairs, 64 calibration pairs and 64 evaluation pairs. Only the 400 answerable training variants supervise the rankers. Epoch selection uses support loss on the 64 answerable calibration variants; candidate selection uses complete-support coverage then support F1. This split is for model selection, not probability calibration. Policies are frozen before opening evaluation answers/support labels. Reader input allowlists exclude answers, answerability, decomposition and supporting flags. Paragraphs have a deterministic hash order independent of the original support positions.

Both rankers run 240 epochs with Adam, learning rate 0.03, L2 coefficient 0.001, batch size 64 and seed 71. CUDA FP32 uses deterministic algorithms with TF32 disabled. Selected epochs are 120 for pointwise and 200 for diffusion. On the RTX 5080 Laptop GPU, the fits take **4.89 s** and **4.79 s** after **1.05 s** of feature preparation. Peak Torch allocated memory is about **67.3 MiB** per fit; this excludes driver reservations and other process/device memory. Python inference agrees with retained CUDA scores within **1.77e-7** maximum absolute error. Tiny parameter count does not make the text feature/index storage free. This is not language-model fine-tuning.

## Frozen local reader comparison

Each reader completes 512 requests: 128 variants times four arms, plus one separate warmup. Qwen2.5 1.5B uses Q4_K_M. Qwen2.5-Coder 7B is a **post-hoc diagnostic reference**, added after the small-reader analysis with the same frozen prompts and policies. It is not a second fresh holdout. Both use pinned Ollama model digests, temperature 0, seed 71, context 4,096 and a 128-token output cap. The largest complete input is 3,712 tokens. Every recorded prompt count matches the local server; no input is truncated.

### Qwen2.5 1.5B

| Context | Answer F1 | Exact / 64 | All supports / 64 | Answers on unanswerable cases / 64 | Complete pairs / 64 | Total model tokens | Summed request seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 0.2549 | 12 | 64 | 35 | 4 | 293,249 | 53.36 |
| BM25, six passages | 0.1932 | 8 | 17 | 29 | 5 | 95,791 | 37.45 |
| Pointwise, six passages | 0.1744 | 7 | 30 | 30 | 3 | 109,460 | 36.19 |
| Diffusion, six passages | 0.2213 | 10 | 38 | 27 | 5 | 108,717 | 36.41 |

### Qwen2.5-Coder 7B

| Context | Answer F1 | Exact / 64 | All supports / 64 | Answers on unanswerable cases / 64 | Complete pairs / 64 | Total model tokens | Summed request seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 0.2042 | 8 | 64 | 11 | 5 | 292,841 | 107.19 |
| BM25, six passages | 0.0929 | 4 | 17 | 4 | 4 | 95,346 | 52.21 |
| Pointwise, six passages | 0.0661 | 1 | 30 | 5 | 1 | 109,178 | 49.99 |
| Diffusion, six passages | 0.1077 | 3 | 38 | 4 | 3 | 108,382 | 50.26 |

F1 and exact answer scores apply to the 64 answerable cases; false positives apply to the 64 unanswerable counterparts. A complete pair requires an exact answer to the answerable variant and abstention on its unanswerable partner. Output exactly equal to `UNKNOWN` after trimming/case conversion is an abstention. Other outputs are retained and scored unchanged.

The native support metric is applied to selected context, not to supports cited by a model. These numbers are **not directly comparable to a complete native leaderboard submission**. Nine controlled examples check the independent replay against upstream scoring, including false positives, missing support, duplicate supports and normalization. One BM25 response from the small reader reaches its output cap and stays in the denominator; all other outputs complete. Both runs have zero recorded protocol errors.

For the small reader, diffusion minus BM25 answer F1 is **+0.0281**, with a descriptive paired 95% bootstrap interval **[-0.0745, +0.1317]**. The complete-support difference is **+0.3281 [0.1719, 0.4844]**. Intervals use 5,000 resamples of the 64 source-separated question pairs, seed 71; they are not multiplicity-adjusted promotion tests. The seven exact-answer gains and five regressions against BM25 are all retained. Diffusion answers only **9 of the 38** cases with every required passage exactly. The larger coding reader does not resolve this bottleneck and is more likely to abstain; this does not establish that larger models are generally worse.

Diffusion uses **62.9% fewer total model tokens** and **31.8% less summed request time** than full context for the small reader, while exact answers fall from 12 to 10. Against BM25 it uses **13.5% more tokens**, with only a **2.8%** reduction in summed request time in this one run. The reference reader similarly saves about 63.0% tokens and 53.1% request time against full context, while exact answers fall from 8 to 3. No speed-for-equal-quality claim is supported.

Request time includes the local HTTP generation call; context compilation is separately recorded (about 0.29–0.51 s per arm across 128 requests). It excludes download, data auditing, feature preparation, training and cold warmup. Full records retain median/p95, tokens, server timings and compilation. This is a serial laptop run, not throughput, streamed time-to-first-token, energy, complete process-tree peak memory or target-edge qualification. Training cost must be amortized in a deployment decision.

## Reproduce on a local RTX

For a quick offline audit, no dataset, Ollama or GPU is needed:

```powershell
git clone --branch research/enterprise-context https://github.com/jprbom/context-stamps.git
cd context-stamps
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -e '.[dev]'
& .venv/Scripts/python.exe experiments/verify_multihop.py
& .venv/Scripts/python.exe experiments/test_relation_ranker.py -v
& .venv/Scripts/python.exe experiments/test_multihop_evidence.py -v
```

The audit reconstructs selections, answer/support metrics, all denominators, token/time totals, paired intervals and known partition exclusions. It checks recorded bytes; it does not re-attest dataset truth or reproduce GPU output bit-for-bit.

For new training, use a separate CUDA environment. The measured stack was Python 3.12.10, Torch 2.11.0+cu128, NumPy 2.4.6, Transformers 5.17.0 and Tokenizers 0.23.2. Ollama was 0.34.2. Install a CUDA-enabled Torch build compatible with your driver; the CPU development environment alone cannot run this trainer. Data/run paths below are siblings of the clone, never inside the publishing checkout. Use unused directories; failed attempts must not be overwritten.

```powershell
# In the CUDA-enabled Python environment:
python -m pip install -e '.[dev]' tokenizers==0.23.2
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'

# Public download, approximately 272 MB compressed; allow about 1 GB locally.
# Existing pinned source/data can be reused instead of downloading again.
python experiments/fetch_multihop.py --source ../musique-source-new --data ../musique-data-new

# The earlier RULER download provides the pinned SQuAD/HotpotQA overlap corpora.
# For a clean machine, prepare these with docs/ruler-local-development.md first.
python experiments/multihop_data.py --data ../musique-data-new --source ../musique-source-new --prior ../ruler-generator-source-v1/scripts/data/synthetic/json --output ../multihop-prepared-new
python experiments/train_relation_ranker.py --data ../multihop-prepared-new --output ../relation-ranker-new
```

Use the already installed pinned local models and tokenizers. Exact digests and tokenizer hashes are in the [small](../evidence/multihop-v1/small/registration.json) and [reference](../evidence/multihop-v1/reference/registration.json) registrations. `docs/ruler-local-learning.md` and `docs/page-relational-memory.md` describe their earlier setup. A moved model tag or different tokenizer is a new experimental configuration, not an exact replay. The runner refuses a changed identity and refuses to start while another Ollama model is resident; coordinate with your own workloads instead of terminating unrelated processes.

```powershell
python experiments/multihop_reader.py prepare --data ../multihop-prepared-new --trained ../relation-ranker-new --tokenizer ../qwen25-15b-tokenizer --output ../multihop-reader-new
python experiments/multihop_reader.py run --output ../multihop-reader-new
python experiments/eval_multihop.py --data ../multihop-prepared-new --run ../multihop-reader-new --source ../musique-source-new --output ../multihop-analysis-new

# Run sequentially after the small reader has released its own model.
python experiments/multihop_reference.py prepare --data ../multihop-prepared-new --trained ../relation-ranker-new --tokenizer ../qwen25-coder-7b-tokenizer --previous-analysis ../multihop-analysis-new --output ../multihop-reference-new
python experiments/multihop_reference.py run --output ../multihop-reference-new
python experiments/eval_multihop.py --data ../multihop-prepared-new --run ../multihop-reference-new --source ../musique-source-new --output ../multihop-reference-analysis-new
```

Review the seven pinned upstream scoring files before executing the scorer. The fetcher never executes upstream installers. Download hash/size checks do not replace code review. Generation uses loopback only and performs no paid provider calls. Corpus text is untrusted evidence and receives no authority to run tools. No credentials, personal local outcomes, raw paragraphs, raw prompts or pretrained model weights are published in this experiment.

## Next acceptance gates

The current 64 pairs are now development evidence. Do not retune on them and describe them as a fresh test. Next candidates need separately reserved question families and older-task retention cohorts. Compare a general-purpose reader and an explicit evidence-check/abstention stage, charging every extra call, token and millisecond. Keep the pointwise and lexical controls so recurrence earns its cost. Any new learned sufficiency signal needs its own scope calibration; ranking mass cannot substitute for it.

Use the [local learning registry and monitor](local-domain-learning.md) only after a real domain candidate passes prospective quality, retention and device/resource gates. This study does not activate a background learner, supply trustworthy labels for arbitrary local interactions or demonstrate autonomous weight improvement. The [complete failure review](../evidence/multihop-v1/failure-review.md) records the remaining gaps and retained preparation failures.
