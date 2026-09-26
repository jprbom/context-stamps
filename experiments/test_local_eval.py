"""CPU-only checks for the local pilot's spending/provenance boundaries."""

import copy
import tempfile
import unittest
from pathlib import Path

from local_eval import CONFIG, digest, validate_model, validate_plan, write_new


class LocalEvaluationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.name = "fixture:small"
        self.tags = {"models": [{"name": self.name, "size": 1024, "digest": "a" * 64}]}
        self.details = {"details": {"format": "gguf"}, "template": "prompt-v1"}

    def test_exact_local_identity_binds_template(self):
        first = validate_model(self.name, self.tags, self.details)
        changed = dict(self.details, template="prompt-v2")
        self.assertNotEqual(first, validate_model(self.name, self.tags, changed))

    def test_cloud_alias_rejected_even_without_remote_fields(self):
        self.tags["models"][0]["name"] = "fixture:cloud"
        with self.assertRaises(ValueError):
            validate_model("fixture:cloud", self.tags, self.details)

    def test_remote_indirection_rejected_in_both_api_responses(self):
        for field in ("remote_host", "remote_model"):
            for target in ("tags", "details"):
                tags, details = copy.deepcopy(self.tags), copy.deepcopy(self.details)
                (tags["models"][0] if target == "tags" else details)[field] = "remote"
                with self.subTest(field=field, target=target), self.assertRaises(ValueError):
                    validate_model(self.name, tags, details)

    def test_missing_weights_digest_or_exact_tag_refused(self):
        for patch in ({"size": 0}, {"digest": "bad"}, {"name": "fixture:other"}):
            tags = copy.deepcopy(self.tags)
            tags["models"][0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_model(self.name, tags, self.details)

    def test_prior_evidence_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            write_new(path, {"status": "failed"})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_new(path, {"status": "success"})
            self.assertEqual(before, path.read_bytes())

    def test_plan_cannot_add_provider_fallback_or_expand_sample_limit(self):
        import local_eval

        plan = {"schema": 1, "runner_sha256": digest(Path(local_eval.__file__).read_bytes()),
                "config": dict(CONFIG), "samples": [{"id": "a"}]}
        validate_plan(plan)
        for patch in ({"fallback_models": ["remote/model"]}, {"max_connections": 10}):
            changed = copy.deepcopy(plan)
            changed["config"].update(patch)
            with self.assertRaises(ValueError):
                validate_plan(changed)
        for samples in ([], [{"id": str(i)} for i in range(21)], [{"id": "a"}, {"id": "a"}]):
            with self.assertRaises(ValueError):
                validate_plan(dict(plan, samples=samples))

    def test_changed_runner_rejected(self):
        with self.assertRaises(ValueError):
            validate_plan({"schema": 1, "runner_sha256": "0" * 64})


if __name__ == "__main__":
    unittest.main()
