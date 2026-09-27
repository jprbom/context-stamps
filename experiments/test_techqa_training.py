"""Boundary tests for label isolation and admission of local reader outcomes."""

import json
import tempfile
import unittest
from pathlib import Path

from ruler_native import sha
from train_techqa_policy import accepted, load_phase


class TechQATrainingTests(unittest.TestCase):
    def test_admission_requires_source_binding_scope_and_complete_output(self):
        row = dict(error=None, final_scope_check=True, check=dict(status="source_bound"),
                   bound_prediction=dict(doc_id="doc", start_offset=1, end_offset=2, score=1), truncated=False)
        self.assertTrue(accepted(row))
        for change in (dict(error="parse failed"), dict(final_scope_check=False),
                       dict(check=dict(status="rejected")), dict(bound_prediction=None), dict(truncated=True)):
            self.assertFalse(accepted(row | change))
        self.assertFalse(accepted({}))

    def test_training_cannot_open_development_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("manifest.json", "registration.json", "prepared.json"):
                (root/name).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot open development"):
                load_phase(root, root, "development", "modern")

    def test_run_and_labels_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, runs = root/"data", root/"runs"
            data.mkdir()
            runs.mkdir()
            run = runs/"fit-modern"
            run.mkdir()
            def put(path, value):
                path.write_text(json.dumps(value), encoding="utf-8")
            put(data/"fit.keys.json", {"q": dict(ANSWERABLE="N")})
            put(data/"manifest.json", dict(files={"fit.keys.json": sha(data/"fit.keys.json")}))
            put(runs/"fit.compiled.json", [dict(id="q", question="fixture")])
            put(runs/"registration.json", dict(data_manifest=sha(data/"manifest.json"), source_hashes={}))
            put(runs/"prepared.json", dict(registration=sha(runs/"registration.json"),
                                            compiled=dict(fit=sha(runs/"fit.compiled.json"))))
            put(run/"fixture-cited.json", dict(id="q", mode="cited", answer=None))
            put(run/"fixture-direct.json", dict(id="q", mode="direct", answer=None))
            put(run/"complete.json", dict(model_unchanged=True, records=2, source_hashes={},
                                            files={p.name: sha(p) for p in run.glob("*.json")}))
            keys, _, _ = load_phase(data, runs, "fit", "modern")
            self.assertEqual(set(keys), {"q"})
            put(runs/"fit.compiled.json", [dict(id="q", question="modified")])
            with self.assertRaisesRegex(ValueError, "preparation changed"):
                load_phase(data, runs, "fit", "modern")


if __name__ == "__main__":
    unittest.main()
