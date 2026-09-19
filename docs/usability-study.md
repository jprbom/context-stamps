# Developer usability study

Prepared by Prashant Jagtap. No external participants have completed this study yet.

A [clean Windows installation smoke test](../evidence/usability-smoke-v1/manifest.json) passed using the public v0.2.0 wheel. It confirms those commands work on one maintainer host; it does not replace participant testing.

The purpose is to learn whether developers can install Context Stamps, understand its version checks, and fit it into an existing workflow. This is a usability study, not a model-quality benchmark. The repository owner will invite participants; this kit does not send invitations or collect telemetry.

## Invitation to copy

> Would you try three short Context Stamps tasks and tell me where the instructions or API are unclear? Allow about 25 minutes. Python 3.10 or newer is needed; no model, API key or account is required. Participation is optional. Please use only the fictional examples, and do not share company files, credentials or personal data. You can stop at any time. Send feedback privately to the person who invited you. I will ask separately before quoting or publishing your feedback.

## Participant instructions

Work from a fresh virtual environment. Record your Python version and operating system; omit usernames and directory paths. Start a timer before each task and stop when you think it is complete. Record failures as useful findings. Do not spend more than eight minutes on one task. You may consult the linked documentation. Record any help from a person or an assistant, so unaided and assisted results can be distinguished.

### Task 1: first working example

Install the published wheel:

```bash
python -m venv .venv
```

Activate on Windows PowerShell with `.venv\Scripts\Activate.ps1`, or on macOS/Linux with `source .venv/bin/activate`. If activation is blocked, invoke `.venv\Scripts\python.exe` directly on Windows. Then:

```bash
python -m pip install https://github.com/jprbom/context-stamps/releases/download/v0.2.0/cortex_context_stamps-0.2.0-py3-none-any.whl
python -m context_stamps --help
```

Copy the first Python example from the [README](../README.md#python-remember-retrieve-and-pack) into a file and run it. Explain what the returned text contains and whether a binary stamp can reconstruct the original text. Record whether installation and the example worked, time taken, and the first confusing instruction.

### Task 2: change a source

Read [the changing-evidence example](../examples/changing_evidence.py). Copy it into a local file and run it with the installed package. Change the current specification once more and rerun it. Explain why an old derived plan must not be selected. Identify which source versions the caller must supply and what happens if a required source is missing or cannot fit the budget.

Success means you can demonstrate a current original source in the packet, a rejected stale plan, and an explicit insufficient-evidence result for an absent required source. Use the API and documentation; do not edit library internals.

### Task 3: use it in your preferred form

Choose one: Python library, command line, [agent instructions](../skills/context-stamps/SKILL.md), or [MCP integration](integrations.md). Describe a small workflow you already have and build a fictional two-record example for it. If your chosen integration needs extra setup, record that setup and where you stopped. No model call is required.

Explain whether you would use this package, a few local functions, or an existing retrieval system for that workflow, and why. We want the reason even if you would not adopt the package.

Return the [feedback template](usability-feedback-template.md) to your inviter. Do not open a public issue containing private feedback or identifying information.

## Facilitator guide

Invite developers with a mix of Python and agent-tool experience; a useful first round is five people, not a representative statistical sample. Do not lead them toward positive answers. Let them attempt each task before offering help. Record the exact point where help was needed. Ask “What did you expect?” rather than explaining the intended design first.

For each task report the participant count, unassisted completions, assisted completions, failures, time distribution and recurring problems. Keep participants anonymous using random IDs. Report the number invited and the number who responded when known; do not treat silence as success. Separate installation smoke tests performed by the maintainer from outside-user results.

Ask for explicit permission before publishing a quote or individual response. The default is a short aggregate report containing no identifying details. Keep private feedback only as long as needed for this study and honor deletion requests. Prefer privately supplied feedback over a public issue when confidentiality is uncertain.

Prioritize fixes that unblock task completion. Repeat affected tasks with new participants after changes. Do not claim “easy to use” from the maintainer's own installation test alone.
