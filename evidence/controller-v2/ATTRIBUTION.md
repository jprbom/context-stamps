# Attribution and artifact terms

Original implementation and documentation: copyright 2026 Prashant Jagtap, MIT License. Keep the owner's notice with source redistribution. The source license does not replace third-party dataset or pretrained-model terms.

This directory's public-dataset-derived rankings, IDs, evaluation results and experimental controller weights are distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), retaining the acknowledgements below and in [controller-v1 attribution](../controller-v1/ATTRIBUTION.md). No raw documents/queries, private Cortex research, credentials or external pretrained weights are republished.

The corresponding `docs/assets/controller-v2-*` result figures carry the same attribution. Original explanatory article text and implementation documentation remain authored by Prashant Jagtap under the repository's source/documentation terms.

- **SciFact**, David Wadden and colleagues, *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020. [Source](https://github.com/allenai/scifact). Training, tuning, calibration and regression evaluation use separate query assignments.
- **NFCorpus**, Vera Boteva, Demian Gholipour, Artem Sokolov and Stefan Riezler, *A Full-Text Learning to Rank Dataset for Medical Information Retrieval*, ECIR 2016. [Source](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/). Training, tuning, calibration and regression evaluation.
- **ArguAna**, Henning Wachsmuth and colleagues, *Retrieval of the Best Counterargument without Prior Topic Knowledge*, ACL 2018. [BEIR card](https://huggingface.co/datasets/BeIR/arguana). Regression evaluation only.
- **SciDocs**, Arman Cohan and colleagues, *SPECTER: Document-level Representation Learning using Citation-informed Transformers*, ACL 2020. [Source](https://github.com/allenai/scidocs). Regression evaluation only.
- **FiQA**, financial question-answering collection, FiQA 2018 / BEIR. [BEIR dataset card and CC BY-SA 4.0 terms](https://huggingface.co/datasets/BeIR/fiqa), [dataset registry](https://github.com/beir-cellar/beir/wiki/Datasets-available). Development split for calibration, test split for fresh local transfer evaluation; no FiQA training split used.
- **BEIR**, Nandan Thakur and colleagues, *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, 2021. [Source and dataset rights guidance](https://github.com/beir-cellar/beir).
- **all-MiniLM-L6-v2**, frozen Sentence Transformers encoder. [Model card / Apache-2.0](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- **ms-marco-MiniLM-L6-v2**, frozen cross-encoder teacher, trained upstream on MS MARCO. [Model card / Apache-2.0](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). Predictions are auxiliary supervision, not newly collected human judgments. Its base weights are obtained separately.

Model revisions and dataset fingerprints are recorded in the protocol and data manifest. The small student/controller checkpoints were trained locally from random initialization on the declared public training partitions. Their size excludes the encoder, teacher, corpus and index. Attribution is not a claim of ownership of these third-party datasets, pretrained models or established learning methods.
