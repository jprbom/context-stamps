# Integration examples

## Recommended Python entry point

Run `examples/progressive_context.py` for exact-first routing, precise fallback and selective packet reuse; `examples/partial_facets.py` shows queries with missing facets; `examples/automatic_facets.py` shows bounded extraction and provenance. Use `ProgressiveRouter`, `ContextSession`, `FacetQuery` and `FacetCompiler` from the package exports. The host supplies authorization, versions and its existing precise retriever. No public compact policy qualifies by default. [Current guide](selective-context.md) and [adaptive capsules](adaptive-capsules.md).

The CLI/MCP/skill interfaces below keep their existing SQLite evidence scope; installing them does not automatically expose the new Python APIs. A 32-byte receipt is process-local and distinct from a 32-byte spherical stamp.

## Library and independent embeddings

```python
from context_stamps import ContextMemory


class MyEncoder:
    identity = "my-embedding-model@immutable-revision:pooling=mean:normalized"
    dim = 3

    def encode(self, text):
        # Replace this demonstration vector with your encoder's actual output.
        return [1.0, 0.0, 0.0]


with ContextMemory(encoder=MyEncoder()) as memory:
    memory.add("A demonstration document.", source="demo")
    print(memory.recall("document"))
```

The constant vector above demonstrates the interface only; it has no retrieval quality. Low-level `stamp_vector` supports precomputed vectors without implementing a text adapter.

## MCP clients

Install from the checkout:

```bash
python -m pip install -e ".[mcp]"
```

A typical client configuration is:

```json
{
  "mcpServers": {
    "context-stamps": {
      "command": "/absolute/path/to/venv/bin/cstamps",
      "args": ["--db", "/absolute/path/to/project/context.sqlite", "serve"]
    }
  }
}
```

On Windows, use the full path to `venv\Scripts\cstamps.exe` and JSON-escape backslashes. Match your client's configuration format. The server provides `recall`, `pack`, `inspect_source`, `select` and `explain` by default. Explicit `--allow-writes` adds `remember`, `invalidate` and `forget`. `pack` uses a byte budget. No HTTP port or hosted account is required.

Use the CLI's `--model` and `--revision` before `serve` if using a neural encoder. These must match any existing store. First-time neural initialization may download weights; the default lexical server does not.

## Agent skills

Copy `skills/context-stamps` into the skill directory supported by your agent. For a Claude Code project, the destination is `.claude/skills/context-stamps/`. Install `cstamps` into an environment visible to that agent. The skill is self-contained and calls the executable, so it does not depend on a relative path back to this repository.

If skill loading is unavailable, supply `CONTEXT_STAMPS.md` as instructions. Neither method intercepts the host's internal context. Automatic transcript processing requires a separate, host-specific integration and separate testing.

## Local SLM

`examples/local_slm.py` provides a complete request to a local Ollama server. It stores two short reference chunks, packs relevant evidence, and sends actual text with source identifiers. Run `--dry-run` first to inspect all content before using a model.

The example uses a lexical encoder to avoid downloading another model. Change the `ContextMemory` encoder to the pinned semantic adapter for a separate retrieval-quality experiment. The generator model does not determine the stamp family; the encoder does.

## Edge deployment

The standard-library core requires a Python runtime. It has no accelerator requirement. A neural encoder contributes most of the model memory and may dominate latency. Reuse cached content digests to avoid re-embedding unchanged sources.

This release does not include browser/WASM, mobile-native, microcontroller, ONNX or quantized weight artifacts. A future export must benchmark memory, cold/warm latency and retrieval quality on named hardware and assign an appropriate encoder identity. Smaller fingerprints alone do not make a neural encoder smaller.

MCP exposes read tools by default. Add `--allow-writes` after `serve` only to enable remember, invalidate and forget for a trusted client. See [security guidance](../SECURITY.md).

## Spherical candidate integration

`scqr encode examples/facets.json` produces portable and compact payloads plus a shared schema. The spherical Python API is exported from `context_stamps`; run `examples/spherical_workflow.py` for a full local handoff. The existing MCP and historical `cstamps` tools retain their documented scope. For agent instructions, require the host to authenticate roles, supply current exact facets, resolve evidence after activation, and abstain on incomplete packets. Never instruct a model to treat a hash as reconstructed source knowledge.
