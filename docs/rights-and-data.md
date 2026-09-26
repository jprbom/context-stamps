# License, attribution and data boundaries

Context Stamps remains **MIT licensed**, copyright 2026 Prashant Jagtap. This allows use, modification, redistribution and commercial use while requiring preservation of the copyright and permission notice. The warranty disclaimer is retained. See [the complete license](../LICENSE) and [MIT's published terms](https://opensource.org/license/mit).

Please credit Prashant Jagtap and cite [CITATION.cff](../CITATION.cff) in research. Citation is requested, not an extra license restriction. Forks and competing products are permitted. No claim of exclusive control over the underlying algorithms is made. No permission to imply endorsement by Prashant Jagtap is provided. This document does not add a noncommercial restriction or a mandatory service dependency.

## Included

- Authored source, tests, documentation and examples under MIT.
- Original fictional fixtures and their generated training records under MIT.
- Small selector coefficients trained in the disclosed experiments (synthetic: MIT; SciFact-derived exports and evidence: CC-BY-SA-4.0, with attribution in that directory). Model cards describe their data and limitations; they are experimental, not general-purpose language models.
- Public benchmark document/query IDs, rankings, aggregate metrics, file hashes and run configuration. The SciFact corpus and claims are not bundled.
- Third-party diagram viewer code with its original MIT notice.

## Excluded

Private Cortex research, chat transcripts, private corpus content, user databases, downloaded embedding weights, API keys, absolute local paths, personal documents and cached embeddings are not release inputs. A user's runtime database remains that user's data; installing this package does not upload it.

SciFact is a third-party dataset, identified as CC-BY-SA-4.0 by its [BEIR dataset card](https://huggingface.co/datasets/BeIR/scifact). Users must obtain it separately and respect its terms. MiniLM's [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) specifies Apache-2.0; its pretrained weights are not republished. Third-party data and model terms are not replaced by this repository's MIT license.

The standalone repository is public. Publication covers only this reviewed component and does not include the wider private Cortex workspace. Release packages provide the source, tests and reproducible evidence rather than attempting to hide Python source in a wheel.

The follow-up NFCorpus/ArguAna benchmark has [separate attribution and CC-BY-SA-4.0 terms](../evidence/replication-v1/ATTRIBUTION.md). Original fictional local-SLM fixtures and their generated outputs remain MIT; the public Qwen model is identified in the run manifest and its weights are not redistributed. No external participant responses are included. The usability kit asks for separate publication permission before sharing participant feedback.

## Controller experiment

The 26 September controller update adds locally trained public-data checkpoints and retrieval evidence under separate [CC BY-SA 4.0 attribution](../evidence/controller-v1/ATTRIBUTION.md). They are optional research artifacts, not part of the dependency-free MIT core or bundled Python wheel. The authored computation-cache fixtures and local Qwen pilot records remain MIT; Qwen weights are separately installed and not redistributed. The controller never trained on private Cortex records or model-generated image/audio/video media.

The follow-up controller-v2 adds FiQA calibration/evaluation and a separately obtained frozen cross-encoder teacher. Its [artifact attribution](../evidence/controller-v2/ATTRIBUTION.md) distinguishes source code, derived controller weights and third-party model/data terms. FiQA training data is unused. External teacher weights and raw public text remain outside Git. Dynamic-int8 execution is constructed locally from the experimental student; no pickle checkpoint is distributed.

## Multimodal evaluation models

The multimodal pilot uses pinned external model revisions for inference only. SD-Turbo retains its model-specific Stability AI license; MMS English TTS uses CC-BY-NC-4.0. The ModelScope video card contains inconsistent NC and NC-ND labels; treat redistribution and commercial use as unresolved until its terms are clarified. No weights or generated media from these models are included in this repository. Their use in a local research test does not make them part of the MIT-licensed core.

Model cards: [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo), [MMS English](https://huggingface.co/facebook/mms-tts-eng), [ModelScope video](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b). The model metadata in spherical manifests has its generated Modelfile removed to avoid publishing a local cache path; the redaction is recorded.
