# Controller experiment: attribution and terms

Original implementation and documentation: copyright 2026 Prashant Jagtap, MIT License. The core package's MIT terms do not replace third-party dataset/model terms.

This directory's public-dataset-derived results, IDs, rankings and exported experimental weights are provided under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), with these dataset acknowledgements retained. No raw documents, queries, private Cortex data or base-model weights are redistributed. This choice is not a determination that every possible use or redistribution of the original datasets is permitted; check the original sources and the BEIR notices.

- **SciFact**: David Wadden and colleagues, *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020. [Dataset](https://github.com/allenai/scifact), [BEIR card](https://huggingface.co/datasets/BeIR/scifact).
- **NFCorpus**: Vera Boteva, Demian Gholipour, Artem Sokolov and Stefan Riezler, *A Full-Text Learning to Rank Dataset for Medical Information Retrieval*, ECIR 2016. [Dataset](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/), [BEIR card](https://huggingface.co/datasets/BeIR/nfcorpus).
- **ArguAna**: Henning Wachsmuth and colleagues, *Retrieval of the Best Counterargument without Prior Topic Knowledge*, ACL 2018. [BEIR card](https://huggingface.co/datasets/BeIR/arguana). Evaluation only.
- **SciDocs**: Arman Cohan and colleagues, *SPECTER: Document-level Representation Learning using Citation-informed Transformers*, ACL 2020. [Source](https://github.com/allenai/scidocs), [BEIR card](https://huggingface.co/datasets/BeIR/scidocs). Evaluation only.
- **BEIR**: Nandan Thakur and colleagues, *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, 2021. [Repository and dataset rights guidance](https://github.com/beir-cellar/beir).
- **all-MiniLM-L6-v2**: Sentence Transformers pretrained encoder, [model card and Apache-2.0 terms](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Frozen, separately obtained, not republished.

The MLP and recurrent controller checkpoints were trained locally from random initialization on the stated training partitions. Their small size excludes the frozen encoder, documents and search index. They are optional research artifacts, with calibration failures and transfer regressions recorded. See the [model card](MODEL_CARD.md).
