"""Synthetic CPU tests; neither MLX nor real weights are imported."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from compare_layers import compare, metrics
from layer_dump import CallHooks, Writer, expected_points, golden_ids


class LayerTests(unittest.TestCase):
    def make_dump(self, directory, change=None, magnitude=0.5):
        writer = Writer(directory, [1, 2], 2, "synthetic")
        for layer, point in expected_points(2):
            rows = 1 if point in ("logits", "final_norm") else 2
            value = np.ones((rows, 4))
            if point == "router_indices":
                value = np.tile([0, 2], (rows, 1))
            if point == "router_weights":
                value = np.full((rows, 2), 0.5)
            if (layer, point) == change:
                value = value.astype(float)
                value[-1, -1] += magnitude
            writer.save(layer, point, value)
        writer.finish()

    def test_equal_and_first_substep(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root) / "a", Path(root) / "b"
            self.make_dump(a)
            self.make_dump(b, (1, "attention_raw"))
            self.assertIsNone(compare(a, a)["first_divergence"])
            report = compare(a, b)
            self.assertEqual(report["first_divergence"], "layers.001.attention_raw")
            result = next(p for p in report["points"] if p["name"] == report["first_divergence"])
            self.assertEqual(result["max_abs"], 0.5)
            self.assertEqual(result["first_bad_row"], 1)
            self.assertEqual(result["max_location"], [1, 3])

    def test_experts_exact_even_with_loose_threshold(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root) / "a", Path(root) / "b"
            self.make_dump(a)
            self.make_dump(b, (0, "router_indices"), 1)
            report = compare(a, b, atol=100, rtol=100, min_cosine=-1)
            self.assertEqual(report["first_divergence"], "layers.000.router_indices")
            point = next(p for p in report["points"] if p["name"] == report["first_divergence"])
            self.assertEqual(point["changed_expert_rows"], [1])
            self.assertEqual(point["routing"][0]["boundary_margins"], [0, 0])

    def test_reject_incomplete_nonfinite_shape_and_prompt(self):
        for defect in ("missing", "nan", "truncated", "prompt", "shape"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as root:
                a, b = Path(root) / "a", Path(root) / "b"
                self.make_dump(a)
                self.make_dump(b)
                path = b / "index.json"
                index = json.loads(path.read_text())
                if defect == "missing":
                    index["entries"].pop()
                elif defect == "prompt":
                    index["prompt_ids"][0] = 9
                elif defect == "shape":
                    index["entries"][0]["shape"][1] = 5
                elif defect == "truncated":
                    (b / "embeddings.f32").write_bytes(b"x")
                else:
                    (b / "embeddings.f32").write_bytes(np.full((2, 4), np.nan, dtype="<f4").tobytes())
                path.write_text(json.dumps(index))
                with self.assertRaises(ValueError):
                    compare(a, b)

    def test_zero_cosine_and_tolerances(self):
        z = np.zeros((1, 3))
        self.assertEqual(metrics(z, z, 0, 0, 1)["cosine"], 1)
        self.assertEqual(metrics(z, z+1, 0, 0, 1)["cosine"], 0)
        self.assertFalse(metrics(z+1, z+1.001, 0.01, 0, 0)["significant"])

    def test_hook_restores_methods_and_preserves_output(self):
        class Block:
            def __call__(self, x, scale=2):
                return x * scale
        first, second, untouched = Block(), Block(), Block()
        original = Block.__call__
        captured = []
        with self.assertRaisesRegex(ValueError, "intentional"):
            with CallHooks() as hooks:
                hooks.add(first, after=captured.append)
                hooks.add(second, after=lambda out: captured.append(-out))
                self.assertEqual(first(3, scale=4), 12)
                self.assertEqual(second(3), 6)
                self.assertEqual(untouched(5), 10)
                raise ValueError("intentional")
        self.assertEqual(captured, [12, -6])
        self.assertIs(Block.__call__, original)

    def test_golden_payload_integrity(self):
        import hashlib
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "gemma4.json"
            payload = json.dumps(dict(model="gemma4", cases=[dict(prompt_ids=[4, 5])])).encode()
            path.with_suffix(".bin").write_bytes(payload)
            path.write_text(json.dumps(dict(payload_sha256=hashlib.sha256(payload).hexdigest())))
            self.assertEqual(golden_ids(path), [4, 5])
            path.with_suffix(".bin").write_bytes(payload + b" ")
            with self.assertRaises(ValueError):
                golden_ids(path)


if __name__ == "__main__":
    unittest.main()
