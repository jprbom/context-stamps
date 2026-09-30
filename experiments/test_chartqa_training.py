"""Public-training runner boundaries and label-independent routing features."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chartqa_protocol import digest
from chartqa_training_policy import environment, fit, input_cell
from chartqa_training_reader import prepared_training, run
from train_chartqa_policy import plan_metrics


class TrainingTests(unittest.TestCase):
    def test_mixed_routes_pay_the_full_extraction_once(self):
        rows = [dict(id=key, mode=mode, image_sha256="image", correct=True, stripped_exact=True,
            error=None, answer="1", origin=0, query_cost=dict(tokens=10, wall_seconds=1.0, processing_seconds=0.1, calls=1))
            for key in ("a", "b") for mode in ("direct", "memory", "program")]
        shared = {"image": dict(tokens=100, wall_seconds=5.0, processing_seconds=0.5, calls=1)}
        direct = plan_metrics(rows, shared, dict(a="direct", b="direct"))
        mixed = plan_metrics(rows, shared, dict(a="direct", b="program"))
        memory = plan_metrics(rows, shared, dict(a="memory", b="program"))
        self.assertEqual(direct["costs"]["tokens"], 20)
        self.assertEqual(mixed["costs"]["tokens"], 120)
        self.assertEqual(memory["costs"]["tokens"], 120)
        self.assertEqual(mixed["extractions_charged"], 1)
        self.assertEqual(mixed["costs"]["calls"], 3)
        with self.assertRaises(ValueError):
            plan_metrics(rows, shared, dict(a="direct"))

    def test_features_use_only_the_complete_question(self):
        expected = {"What is the ratio?": "relative", "What is the average?": "arithmetic",
                    "Which is lowest?": "extreme", "What is the value in 2015?": "other"}
        for question, cell in expected.items():
            self.assertEqual(input_cell(question), cell)
        for question in (None, "", "x"*16385):
            with self.assertRaises(ValueError):
                input_cell(question)

    def test_policy_requires_complete_actions_and_preserves_unknown_cells(self):
        binding = environment("a"*64)
        rows = [dict(id=f"case-{i}", question="What is the sum?", mode=mode,
                     correct=mode != "memory", tokens=10 if mode == "program" else 100)
                for i in range(10) for mode in ("direct", "memory", "program")]
        policy = fit(rows, binding=binding, penalty=10.0)
        self.assertEqual(policy.choose("other", binding=binding, allowed_actions=("direct", "memory", "program")), "direct")
        self.assertEqual(policy.choose("arithmetic", binding=binding, allowed_actions=("direct", "memory", "program")), "program")
        with self.assertRaises(ValueError):
            fit(rows[:-1], binding=binding, penalty=10.0)

    def test_reader_refuses_live_model_before_any_call_or_output(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)/"out"
            with patch("chartqa_training_reader.prepared_training", return_value={}), \
                    patch("chartqa_training_reader.local_api", return_value={"models": [dict(name="other")]}), \
                    patch("chartqa_training_reader.capture") as capture:
                with self.assertRaisesRegex(ValueError, "resident"):
                    run(Path(name), output)
                capture.assert_not_called()
                self.assertFalse(output.exists())

    def test_preparation_reads_no_keys_and_checks_paths_bytes_and_metadata(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            (folder/"images").mkdir()
            rows, files, identities = [], {}, {}
            for i in range(64):
                raw = str(i).encode()
                filename = f"images/{digest(raw)}.png"
                (folder/filename).write_bytes(raw)
                files[filename] = digest(raw)
                identity = dict(image_sha256=digest(raw), pixel_group=digest(raw), width=3, height=4,
                                byte_length=len(raw), media_type="image/png")
                identities[raw] = identity
                for _ in range(3 if i < 15 else 2):
                    origin = "h" if i % 2 else "a"
                    rows.append(dict(identity, split="train", id=f"train-{origin}-{len(rows):05d}", image_file=filename))
            payload = json.dumps(rows).encode()
            (folder/"train.inputs.json").write_bytes(payload)
            files["train.inputs.json"] = digest(payload)
            (folder/"manifest.json").write_text(json.dumps(dict(files=files)))
            # Deliberately invalid targets and unused splits must not be decoded.
            for key in ("train.keys.json", "val.inputs.json", "val.keys.json", "test.inputs.json", "test.keys.json"):
                (folder/key).write_text("not JSON")
            with patch("chartqa_training_reader.image_identity", side_effect=lambda raw: identities[raw]):
                self.assertEqual(len(prepared_training(folder)), 64)
                (folder/rows[0]["image_file"]).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "bytes"):
                    prepared_training(folder)


if __name__ == "__main__":
    unittest.main()
