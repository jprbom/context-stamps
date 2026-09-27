"""Canary lifecycle and failure retention; mocked inference, no GPU calls."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chartqa_canary import capture, run
from chartqa_protocol import request_body
from test_chartqa_protocol import response
from test_chartqa_quant import table


class CanaryTests(unittest.TestCase):
    def test_existing_workload_prevents_generation_and_output_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"trial"
            with patch("chartqa_canary.local_api", return_value=dict(models=[dict(name="busy")])), patch("chartqa_canary.local_chat") as chat:
                with self.assertRaisesRegex(ValueError, "resident local workload"):
                    run(output, Path(temp)/"font.ttf")
                chat.assert_not_called()
                self.assertFalse(output.exists())

    def test_transport_failure_is_retained_without_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            body = request_body("direct", question="Authored?", image=b"fixture")
            with patch("chartqa_canary.local_chat", side_effect=TimeoutError("authored timeout")) as chat:
                with self.assertRaises(TimeoutError):
                    capture(output, "direct", body, "direct")
                self.assertEqual(chat.call_count, 1)
                failure = json.loads((output/"direct.transport-failure.json").read_bytes())
                self.assertFalse(failure["retry_allowed"])
                self.assertTrue((output/"direct.started.json").is_file())
                with self.assertRaises(FileExistsError):
                    capture(output, "direct", body, "direct")
                self.assertEqual(chat.call_count, 1)

    def test_complete_fixture_retains_raw_records_and_is_not_a_benchmark(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            font = folder/"font.txt"
            font.write_bytes(b"mock font; drawing is replaced in this lifecycle test")
            data = table(("20", "40", "60"), ("tasks", "tasks", "tasks"))
            replies = [response(dict(answer="60")), response(data), response(dict(answer="60")),
                       response(dict(program=dict(op="sum", cells=["c0", "c1"])))]
            def local_api(path, body=None):
                return {"/api/ps": dict(models=[]), "/api/show": dict(capabilities=["vision"]),
                        "/api/version": dict(version="mock")}[path]
            with patch("chartqa_canary.local_api", side_effect=local_api), patch("chartqa_canary.model_identity", return_value=dict(digest="a"*64)), \
                    patch("chartqa_canary.create_chart", side_effect=lambda path, font: path.write_bytes(b"authored bytes")), \
                    patch("chartqa_canary.local_chat", side_effect=replies) as chat:
                run(folder/"trial", font)
            self.assertEqual(chat.call_count, 4)
            complete = json.loads((folder/"trial"/"complete.json").read_bytes())
            self.assertTrue(complete["passed"])
            self.assertFalse(complete["candidate_active"])
            self.assertEqual(len(list((folder/"trial").glob("*.response.json"))), 4)
            registration = json.loads((folder/"trial"/"registration.json").read_bytes())
            self.assertIn("excluded", registration["purpose"])


if __name__ == "__main__":
    unittest.main()
