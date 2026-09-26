# Experimental residual context rankers

Author: Prashant Jagtap. Trained locally on 26 September 2026.

**Status: not qualified for default deployment.** Ranking public evidence candidates is the trained task. These are not language models, semantic encoders, fact verifiers, action policies or new versions of Qwen. Their attention is inside the optional controller, not inside a downstream reader.

## Included checkpoints

| File | Training selection | Parameters | Format |
|---|---|---:|---|
| `checkpoints/selected-fp32.safetensors` | MLP seed 29, epoch 4 | 66,497 | FP32 |
| `checkpoints/selected-int8.safetensors` | Same checkpoint | 66,497 | Per-row int8 matrix storage, reconstructed FP32 |
| `checkpoints/recurrent-29.safetensors` | Best recurrent tuning run, seed 29, epoch 10 | 83,137 | FP32, four attention heads, two shared passes |

All six run histories and checkpoint hashes are in `training.json`. Runtime hashes and serialized sizes for the selected exports are in `runtime.json`. Other seed checkpoints remain reproducible locally and are not bundled. Weights are excluded from Python wheels; obtain these research artifacts from the reviewed repository clone.

## Inputs and outputs

Pinned normalized 384D MiniLM query/document embeddings and six retrieval features in the documented order, with a Boolean eligibility mask. Up to 128 candidates were used in training; the API bounds inputs to 256 candidates, but larger sets are unvalidated. Output is a relative ranking score, not probability, confidence, sufficiency or authority. The host must filter permissions before inference and validate source state again before exposure.

SciFact/NFCorpus train/tune/calibration partitions and every exclusion appear in `split-manifest.json`. Only relevance judgments are used. The complete train/evaluation procedure is in the [RTX guide](../../docs/local-rtx-controller.md). No online self-training, parameter rewriting, base-model fine-tuning or paid endpoint was used.

## Evaluation and limitations

The selected learned model scored 0.7190/0.3506/0.5294/0.1934 nDCG@10 on SciFact/NFCorpus/ArguAna/SciDocs. It did not beat the hybrid reference reliably. Independent calibration did not enable the learned route in either training domain. SciDocs regressed below dense, while four-step recurrent inference worsened transfer sharply. The serving method enforces trained depth; direct `forward` step overrides are diagnostic research interfaces.

All public test sets were inspected in earlier work. Reported query bootstrap intervals are descriptive, not multiplicity adjusted. These are English retrieval benchmarks, with incomplete relevance judgments and shared document corpora; no frontier-model, autonomous-agent, multilingual, medical-use or enterprise-scale claim follows from them.

Int8 serialization is lossy and does not supply int8 compute kernels. Its maximum score drift and metric changes are recorded in `supplementary.json` and `summary.json`. There is no claim of runtime acceleration from that storage format.

Terms and source acknowledgements: [ATTRIBUTION.md](ATTRIBUTION.md). Authored source remains MIT; these dataset-derived experimental weights and evidence are CC BY-SA 4.0. Preserve the author and dataset notices. Citation of Prashant Jagtap and the upstream datasets is requested; no endorsement is implied.
