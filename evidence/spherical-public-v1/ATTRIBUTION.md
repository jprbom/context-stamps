# Attribution and interpretation

SciFact: David Wadden and colleagues, *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020. [Original project](https://github.com/allenai/scifact).

NFCorpus: Vera Boteva, Demian Gholipour, Artem Sokolov and Stefan Riezler, *A Full-Text Learning to Rank Dataset for Medical Information Retrieval*, 2016. [Dataset](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/).

ArguAna: Henning Wachsmuth, Shahbaz Syed and Benno Stein, *Retrieval of the Best Counterargument without Prior Topic Knowledge*, 2018. [Dataset](https://webis.de/data/arguana-counterargs.html).

Inputs are the BEIR distributions recorded by SHA-256 in the manifest. See the original [SciFact attribution](../scifact-v1/ATTRIBUTION.md) and [replication attribution](../replication-v1/ATTRIBUTION.md). Computed rankings, metrics and run metadata in this directory are provided under [CC-BY-SA-4.0](../replication-v1/CC-BY-SA-4.0.txt). No raw query/corpus text is redistributed. Runner code is MIT. MiniLM weights remain external under their own Apache-2.0 license.

This run excludes self-document IDs before ranking and retains all positive qrels in the denominator. Its NFCorpus and ArguAna dense values differ slightly from historical runs; use the controls from this run for comparisons. These public test sets have been inspected before. Only the semantic/lexical weight was selected on SciFact validation; there was no test-set retuning or encoder fine-tuning.
