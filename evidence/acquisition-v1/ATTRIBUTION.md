# Attribution and artifact terms

Original implementation, documentation and engineering fixtures: copyright 2026 Prashant Jagtap, MIT License.

Public-dataset-derived coefficients, trajectories and result figures (`docs/assets/acquisition-v1-tradeoff.*`) are released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). These terms are separate from the MIT core. Preserve the dataset acknowledgements in [controller-v2 attribution](../controller-v2/ATTRIBUTION.md): SciFact (Wadden et al.), NFCorpus (Boteva et al.), ArguAna (Wachsmuth et al.), SciDocs (Cohan et al.), FiQA and BEIR (Thakur et al.).

The original frozen MiniLM retrieval features are reused under the provenance of that experiment. External pretrained weights, raw queries/documents, private Cortex data and teacher outputs are not published here. The new model learns only from SciFact/NFCorpus training judgments; FiQA and other regression collections are not used for gradient updates. Existing calibration and regression sets have been used previously by the project.

See the [model card](MODEL_CARD.md), [frozen protocol](protocol.json), and [implementation/results](../../docs/adaptive-acquisition.md). Attribution does not claim ownership of prior methods, datasets or pretrained encoders.
