"""Numerical, masking and serialization boundaries of the optional v2 ranker."""

import tempfile
import unittest
import warnings
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional torch dependency")
class ContractiveTests(unittest.TestCase):
    def setUp(self):
        from context_stamps.contractive_controller import ContractiveRanker
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.model = ContractiveRanker(embedding_dim=8, width=8).eval()
        with torch.no_grad():
            self.model.output.weight.normal_()
            self.model.diagonal.fill_(2)
        self.inputs = (torch.randn(2, 8), torch.randn(2, 7, 8), torch.randn(2, 7, 6),
                       torch.tensor([[True] * 5 + [False] * 2, [True] * 7]))

    def test_contraction_and_extra_depth_convergence(self):
        u, p = self.model.initial_state(*self.inputs)
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1))))
        a, b = torch.randn_like(u) * 20, torch.randn_like(u) * 20
        def distance(x, y):
            return (x - y).norm(dim=-1).max().item()
        self.assertLessEqual(distance(self.model.advance(u, p, a), self.model.advance(u, p, b)),
                             .5 * distance(a, b) + 1e-6)
        reference = self.model(*self.inputs, steps=32)
        errors = [(self.model(*self.inputs, steps=n) - reference).abs().max().item() for n in (2, 4, 8)]
        self.assertLess(errors[2], errors[0])
        self.assertLess(errors[2], .02)

    def test_permutation_equivariance_and_padding_isolation(self):
        q, d, f, m = self.inputs
        expected = self.model(q, d, f, m)
        order = torch.tensor([4, 2, 6, 0, 1, 5, 3])
        actual = self.model(q, d[:, order], f[:, order], m[:, order])
        self.assertTrue(torch.allclose(expected[:, order], actual, atol=1e-6))
        d, f = d.clone(), f.clone()
        d[~m], f[~m] = 1e4, -1e4
        changed = self.model(q, d, f, m)
        self.assertTrue(torch.allclose(expected[m], changed[m], atol=1e-6))

    def test_bounded_residual_serving_and_bad_inputs(self):
        q, d, f, m = self.inputs
        score = self.model.score_candidates(q, d, f, m)
        self.assertLessEqual((score[m] - f[..., 2][m]).abs().max().item(), 1.000001)
        self.assertTrue(torch.isneginf(score[~m]).all())
        for steps in (0, 3, 32, True):
            with self.assertRaises(ValueError):
                self.model.score_candidates(q, d, f, m, steps=steps)
        with self.assertRaises(ValueError):
            self.model.score_candidates(q * float("nan"), d, f, m)
        with self.assertRaises(ValueError):
            self.model.score_candidates(q, d, f, torch.zeros_like(m))

    def test_pair_loss_improves_when_positive_moves_up(self):
        from context_stamps.contractive_controller import ndcg_pair_loss
        y, m = torch.tensor([[1., 0., 0.]]), torch.ones(1, 3, dtype=torch.bool)
        bad = ndcg_pair_loss(torch.tensor([[-1., 1., 0.]]), y, m)
        good = ndcg_pair_loss(torch.tensor([[1., 0., -1.]]), y, m)
        self.assertLess(good.item(), bad.item())
        scores = torch.randn(1, 3, requires_grad=True)
        loss = ndcg_pair_loss(scores, y * 0, m)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_distillation_offset_and_padding_invariance(self):
        from context_stamps.contractive_controller import centered_distillation
        a = torch.tensor([[2., -1., 100.]])
        b = torch.tensor([[1., -2., -100.]])
        m = torch.tensor([[True, True, False]])
        loss = centered_distillation(a, b, m)
        shifted = centered_distillation(a + 17, b - 31, m)
        self.assertTrue(torch.allclose(loss, shifted, atol=1e-6))

    def test_checkpoint_round_trip_and_corruption(self):
        from context_stamps.contractive_controller import load_contractive, save_contractive
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_contractive(self.model, path)
            restored = load_contractive(path)
            self.assertTrue(torch.equal(self.model(*self.inputs), restored(*self.inputs)))
            with torch.no_grad():
                self.model.diagonal[0] = float("nan")
            save_contractive(self.model, path)
            with self.assertRaises(ValueError):
                load_contractive(path)

    def test_dynamic_int8_masked_outliers_do_not_change_valid_scores(self):
        engines = torch.backends.quantized.supported_engines
        engine = next((v for v in ("x86", "onednn", "fbgemm", "qnnpack") if v in engines), None)
        if engine is None:
            self.skipTest("no dynamic-int8 backend")
        previous = torch.backends.quantized.engine
        try:
            torch.backends.quantized.engine = engine
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                quantized = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            q, d, f, m = self.inputs
            expected = quantized.score_candidates(q, d, f, m)
            d, f = d.clone(), f.clone()
            d[~m], f[~m] = 1e4, -1e4
            changed = quantized.score_candidates(q, d, f, m)
            self.assertTrue(torch.equal(expected[m], changed[m]))
        finally:
            torch.backends.quantized.engine = previous

    def test_split_projection_preserves_fp32_and_feature_precision(self):
        from context_stamps.controller_quantization import split_precision_controller
        equivalent = split_precision_controller(self.model, quantize=False)
        self.assertTrue(torch.allclose(self.model(*self.inputs), equivalent(*self.inputs), atol=1e-6))
        available = torch.backends.quantized.supported_engines
        engine = next((v for v in ("x86", "onednn", "fbgemm", "qnnpack") if v in available), None)
        if engine is None:
            self.skipTest("no dynamic-int8 backend")
        previous = torch.backends.quantized.engine
        try:
            torch.backends.quantized.engine = engine
            converted = split_precision_controller(self.model)
            self.assertIsInstance(converted.input.retrieval, torch.nn.Linear)
            self.assertTrue(torch.equal(converted.input.retrieval.weight, self.model.input.weight[:, -6:]))
            q, d, f, m = self.inputs
            expected = converted.score_candidates(q, d, f, m)
            d, f = d.clone(), f.clone()
            d[~m], f[~m] = 1e4, -1e4
            changed = converted.score_candidates(q, d, f, m)
            self.assertTrue(torch.equal(expected[m], changed[m]))
        finally:
            torch.backends.quantized.engine = previous


if __name__ == "__main__":
    unittest.main()
