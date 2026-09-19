import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from context_stamps.encoders import SentenceTransformerEncoder


class EncoderTests(unittest.TestCase):
    def test_pinned_revision_required_before_import(self):
        with self.assertRaises(ValueError):
            SentenceTransformerEncoder("example", revision="main")

    def test_adapter_contract_without_model_download(self):
        class Vector:
            def tolist(self):
                return [0.5, 0.5]

        class Model:
            max_seq_length = 256

            def __init__(self, name, **kwargs):
                self.kwargs = kwargs

            def get_sentence_embedding_dimension(self):
                return 2

            def encode(self, text, **kwargs):
                return Vector()

        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = Model
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            encoder = SentenceTransformerEncoder("example", revision="a" * 40)
            self.assertEqual(encoder.encode("hello"), [0.5, 0.5])
            self.assertFalse(encoder.model.kwargs["trust_remote_code"])
            self.assertIn("a" * 40, encoder.identity)


@unittest.skipUnless(importlib.util.find_spec("mcp"), "optional MCP SDK not installed")
class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_server_has_no_write_tools(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        config = StdioServerParameters(
            command=sys.executable, args=["-m", "context_stamps", "--db", ":memory:", "serve"]
        )
        async with stdio_client(config) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listing = await session.list_tools()
                self.assertEqual({tool.name for tool in listing.tools}, {"recall", "pack", "inspect_source"})
                result = await session.call_tool("forget", {"source": "manual"})
                self.assertTrue(result.isError)

    async def test_real_stdio_roundtrip(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        with tempfile.TemporaryDirectory() as folder:
            config = StdioServerParameters(
                command=sys.executable,
                args=[
                    "-m",
                    "context_stamps",
                    "--db",
                    str(Path(folder) / "m.sqlite"),
                    "serve",
                    "--allow-writes",
                ],
            )
            async with stdio_client(config) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listing = await session.list_tools()
                    self.assertEqual(len(listing.tools), 6)
                    result = await session.call_tool("remember", {"source": "manual", "text": "pump filter"})
                    self.assertFalse(result.isError)
                    result = await session.call_tool("pack", {"query": "filter", "byte_budget": 1000})
                    self.assertFalse(result.isError)
                    payload = json.loads(result.content[0].text)
                    self.assertIn("pump filter", payload["text"])
                    await session.call_tool("invalidate", {"source": "manual"})
                    result = await session.call_tool("pack", {"query": "filter"})
                    self.assertEqual(json.loads(result.content[0].text)["text"], "")


if __name__ == "__main__":
    unittest.main()
