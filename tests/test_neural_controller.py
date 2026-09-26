import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from context_stamps.neural_controller import RecurrentEvidenceRanker, export_ranker, load_ranker


@unittest.skipUnless(HAS_TORCH, "optional controller dependency")
class NeuralControllerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.model = RecurrentEvidenceRanker(embedding_dim=8, width=16, heads=4).eval()
        self.inputs = (torch.randn(2, 5, 8)[:, 0], torch.randn(2, 5, 8),
                       torch.randn(2, 5, 6), torch.ones(2, 5, dtype=torch.bool))

    def test_zero_residual_preserves_baseline(self):
        self.assertTrue(torch.equal(self.model(*self.inputs), self.inputs[2][..., 2]))

    def test_permutation_and_mask_isolation(self):
        torch.nn.init.normal_(self.model.output.weight)
        q, d, f, mask = self.inputs
        mask[:, -1] = False
        original = self.model(q, d, f, mask)
        d[:, -1] = 9000
        f[:, -1] = -9000
        self.assertTrue(torch.allclose(original, self.model(q, d, f, mask), atol=2e-5))
        order = torch.tensor([3, 1, 4, 0, 2])
        changed = self.model(q, d[:, order], f[:, order], mask[:, order])
        self.assertTrue(torch.allclose(original[:, order], changed, atol=2e-5))

    def test_bounded_recurrence_and_invalid_data(self):
        for steps in (0, 5, True):
            with self.assertRaises(ValueError):
                self.model(*self.inputs, steps=steps)
        q, d, f, mask = self.inputs
        for steps in (1, 4, True):
            with self.assertRaises(ValueError):
                self.model.score_candidates(q, d, f, mask, steps=steps)
        with self.assertRaises(ValueError):
            self.model.score_candidates(q, d, f, torch.zeros_like(mask))
        q[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.model.score_candidates(q, d, f, mask)

    def test_serving_masks_remain_below_extreme_valid_scores(self):
        q, d, f, mask = self.inputs
        f[:, :, 2] = -20000
        mask[:, -1] = False
        scores = self.model.score_candidates(q, d, f, mask)
        self.assertTrue(torch.isneginf(scores[:, -1]).all())
        self.assertTrue(torch.isfinite(scores[:, :-1]).all())

    def test_gradient_and_safe_quantized_roundtrip(self):
        torch.nn.init.normal_(self.model.output.weight, std=.05)
        self.model(*self.inputs).sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
        with tempfile.TemporaryDirectory() as folder:
            full, quant = Path(folder) / "full.safetensors", Path(folder) / "int8.safetensors"
            export_ranker(self.model, full)
            export_ranker(self.model, quant, int8=True)
            self.assertTrue(torch.equal(load_ranker(full)(*self.inputs), self.model(*self.inputs)))
            drift = (load_ranker(quant)(*self.inputs) - self.model(*self.inputs)).abs().max().item()
            self.assertLess(drift, .05)
            self.assertLess(quant.stat().st_size, full.stat().st_size)

    def test_wrong_shapes_and_unknown_checkpoints(self):
        q, d, f, mask = self.inputs
        with self.assertRaises(ValueError):
            self.model(q, d, f[:, :, :3], mask)
        with self.assertRaises(ValueError):
            RecurrentEvidenceRanker(width=17)
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.safetensors"
            save_file({"x": torch.ones(1)}, path)
            with self.assertRaises(ValueError):
                load_ranker(path)
