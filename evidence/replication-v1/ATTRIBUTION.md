# Data attribution

The NFCorpus and ArguAna test data were obtained from the official BEIR archives linked in the [BEIR dataset catalogue](https://github.com/beir-cellar/beir/wiki/Datasets-available). Archive MD5 values were checked against that catalogue: NFCorpus `a89dba18a62ef92f7d323ec890a0d38d`; ArguAna `8ad3e3c2a5867cdced806d6503f29b99`. SHA-256 hashes of the extracted inputs are in the manifest.

- NFCorpus: Vera Boteva, Demian Gholipour, Artem Sokolov and Stefan Riezler, *A Full-Text Learning to Rank Dataset for Medical Information Retrieval*, 2016. [Original dataset](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) · [BEIR dataset card](https://huggingface.co/datasets/BeIR/nfcorpus).
- ArguAna: Henning Wachsmuth, Shahbaz Syed and Benno Stein, *Retrieval of the Best Counterargument without Prior Topic Knowledge*, 2018. [Original dataset](https://webis.de/data/arguana-counterargs.html) · [BEIR dataset card](https://huggingface.co/datasets/BeIR/arguana).
- BEIR: Nandan Thakur and colleagues, *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, 2021. [Project](https://github.com/beir-cellar/beir).

Both BEIR dataset cards identify their distributions as CC-BY-SA-4.0. The generated benchmark records in this directory are distributed under [CC-BY-SA-4.0](CC-BY-SA-4.0.txt) with these attributions. Changes consist of computed rankings, metrics, stratification and run metadata. No corpus text or query text is redistributed. This does not relicense the original source datasets or imply endorsement. Authored runner code remains MIT.

The pinned MiniLM encoder is Apache-2.0; no encoder weights are bundled. The frozen linear weights originate from the separately attributed SciFact experiment.
