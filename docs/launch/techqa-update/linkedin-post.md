I have written up the latest local experiment behind Context Stamps: can a small model learn when its evidence is sufficient to answer?

The study used public TechQA data, a local Qwen3.5:4b reader and 1,774 measured requests. I fitted three small policies; none passed the calibration rule.

Source checks reduced false positives, but also lost useful answers and increased request time. The article explains why a higher combined score was not enough, where context selection failed, and what I am changing next.

The code, fitted policies, failed cases and independent replay are available. I would particularly value feedback from people building local domain assistants or evaluating selective prediction and evidence use.

Article: [add the published LinkedIn article link]

Current research:
https://github.com/jprbom/context-stamps/tree/research/enterprise-context

#MachineLearning #SmallLanguageModels #OpenSource
