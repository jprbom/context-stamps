# Local learning needs more than a successful training run

By Prashant Jagtap

I have completed a local TechQA experiment for Context Stamps. The question was whether a small policy could reduce unsupported answers without sacrificing useful answering.

The setup: Qwen3.5:4b, 400 fitting questions, 177 calibration questions and all 310 native public development questions. Two interfaces received identical selected source text: direct answer generation and quotation-first generation. The full study recorded 1,774 measured requests plus three warmups.

Context Stamps uses a 256-bit reference to route to external evidence and relationship information. The source material stays outside the stamp. This experiment tests what happens after evidence has been selected, including whether a model's quotations actually match it.

I fitted three ridge policies with twelve observable features and an intercept. Their target was answer utility from native labels. All three failed calibration: none preserved positive F1 while reducing false positives and improving the combined score. No learned update was activated.

The development set contains 160 answerable and 150 unanswerable questions. Here is the useful comparison, with F1 measured as native character-span overlap on a 0–100 scale:

Direct + exact-span filter
Answerable F1: 13.64
False positives: 95/150

Quoted + source check
Answerable F1: 7.10
False positives: 31/150

The second method looks much better if we only count rejected unsupported answers. It also loses all native overlap on 19 answerable cases that previously received credit, while gaining overlap on four.

Its combined native F1 is 42.05 versus 24.78 for the direct exact-span control. Always abstaining scores 48.39. That control prevents a selective-answering metric from being mistaken for useful competence.

There is a cost problem as well. Quotation-first generation uses 6.39% more total model tokens and 2.33 times the summed request time. Post-generation checking cannot recover those tokens. Context compilation and warmups are reported separately; these are local measurements with other CPU work on the host, not isolated throughput or edge-device evidence.

The failure analysis separates two problems. Only 81/160 answerable questions have a complete reference span in the selected context. Even when evidence is present, copying exact quotations and assigning the right source ID often fails.

The next text experiment will improve evidence selection and test choosing existing span IDs instead of generating copied quotations. I will keep the recorded experiment intact and qualify a new candidate separately.

The broader aim is a locally improving domain system: verified memory, small routing policies and optional weight updates, with independent outcomes, retention tests, resource accounting and rollback. Training completion is only one step in that process.

The visual interface work reinforces that point. After fixing numeric-answer handling and cell-lookup instructions, the latest authored check gets 15/15 direct answers, 15/15 memory answers and 14/15 numerical programs. The failed program calculates a number for a category absent from the chart. Execution is valid; the answer is unsupported. Charging extraction also makes memory more expensive on these small fixtures. These are development checks, not a ChartQA result or learned improvement. They show why a locally improving system needs an independent outcome check before turning experience into training labels.

Code, native target-offset projections, fitted policies, all scored outcomes, cost records and an independent replay are in the research branch. Raw corpus and prompts remain external. No autonomous improvement or universal superiority is claimed from this result.

https://github.com/jprbom/context-stamps/tree/research/enterprise-context

[Attach the TechQA quality-and-cost research figure.]
