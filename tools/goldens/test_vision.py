"""Vérifie les fixtures versionnées sans poids locaux, réseau ni GPU."""

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/vision"
spec = importlib.util.spec_from_file_location("vision_reference", Path(__file__).with_name("vision.py"))
vision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vision)


def load(directory, entry):
    payload = (directory / entry["file"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise AssertionError(f"empreinte : {entry['file']}")
    if len(payload) != entry["size_bytes"]:
        raise AssertionError("taille du payload")
    array = np.frombuffer(payload, dtype=np.dtype(entry["dtype"]).newbyteorder("<"))
    if array.size != np.prod(entry["shape"]):
        raise AssertionError("forme du payload")
    if not np.isfinite(array).all():
        raise AssertionError("payload non fini")
    return array.reshape(entry["shape"])


class VisionFixtures(unittest.TestCase):
    def test_inventory_complete_and_source_pinned(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text())
        self.assertEqual(len(manifest["checkpoints"]), 4)
        self.assertEqual(manifest["environment"]["reference_script_sha256"],
                         vision.digest(Path(vision.__file__).read_bytes()))
        total = 0
        for model in manifest["checkpoints"]:
            tensors = model["tensors"]
            expected = vision.expected_shapes(model["vision_config"], model["conv_layout"] == "OTHWI")
            self.assertEqual(model["unknown_vision_tensors"], [])
            self.assertEqual({t["canonical_name"] for t in tensors}, expected.keys())
            self.assertEqual(len(tensors), model["expected_vision_tensors"])
            for entry in tensors:
                self.assertEqual(entry["shape"], expected[entry["canonical_name"]])
                self.assertEqual(entry["dtype"], "BF16")
                self.assertEqual(len(entry["sha256"]), 64)
            total += len(tensors)
        self.assertEqual(total, 1296)

    def test_images_and_patch_order(self):
        directory = FIXTURES / "images"
        images = json.loads((directory / "images.json").read_text())
        generated = dict(vision.synthetic_images())
        for case in images:
            rgb, patches = load(directory, case["rgb"]), load(directory, case["patches"])
            np.testing.assert_array_equal(rgb, generated[case["image"]])
            h, w, _ = rgb.shape
            self.assertEqual(case["grid_thw"], [[1, h // 16, w // 16]])
            # Oracle par coordonnées : aucune reshape/permute partagée avec la capture.
            for row in range(len(patches)):
                block, inside = divmod(row, 4)
                bh, bw = divmod(block, w // 32)
                ih, iw = divmod(inside, 2)
                for channel in range(3):
                    tile = rgb[bh * 32 + ih * 16:bh * 32 + ih * 16 + 16,
                               bw * 32 + iw * 16:bw * 32 + iw * 16 + 16, channel]
                    expected = (tile.astype(np.float64) / 255 - 0.5) / 0.5
                    for time in range(2):
                        offset = (channel * 2 + time) * 256
                        np.testing.assert_allclose(patches[row, offset:offset + 256].reshape(16, 16),
                                                   expected, atol=1e-6, rtol=0)

    def test_embeddings_tables_and_positions(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text())
        tables_by_image = {}
        for model in manifest["checkpoints"]:
            directory = FIXTURES / model["checkpoint"]
            metadata = json.loads((directory / "fixtures.json").read_text())
            self.assertEqual(metadata["environment"], manifest["environment"])
            self.assertEqual(metadata["vision_payload_sha256"], model["vision_payload_sha256"])
            self.assertEqual(len(metadata["cases"]), 3)
            for case in metadata["cases"]:
                payloads = {k: load(directory, v) for k, v in case["payloads"].items()}
                _, h, w = case["grid_thw"][0]
                embeddings = payloads["embeddings"]
                self.assertEqual(embeddings.shape, (h * w // 4, model["vision_config"]["out_hidden_size"]))
                indices, weights = payloads["interpolation_indices"], payloads["interpolation_weights"]
                self.assertEqual(indices.shape, (4, h * w))
                self.assertEqual(weights.shape, indices.shape)
                self.assertTrue(np.all((indices >= 0) & (indices < 2304)))
                np.testing.assert_allclose(weights.sum(axis=0), 1, atol=1e-7, rtol=0)
                self.assertTrue(np.all((weights >= 0) & (weights <= 1)))
                if case["image"] in tables_by_image:
                    previous = tables_by_image[case["image"]]
                    np.testing.assert_array_equal(indices, previous[0])
                    np.testing.assert_array_equal(weights, previous[1])
                tables_by_image[case["image"]] = (indices, weights)
            prefixes = json.loads((directory / "mrope-inputs.json").read_text())["cases"]
            self.assertEqual(len(prefixes), 6)
            for prefix in prefixes:
                ids = prefix["input_ids"]
                pos = np.array(prefix["position_ids"])
                self.assertEqual(pos.shape, (3, 1, len(ids)))
                self.assertEqual(prefix["rope_delta"], int(pos.max()) + 1 - len(ids))
                self.assertEqual(len(prefix["forced_decode_ids"]), 16)
                self.assertEqual(prefix["decode_kv_positions"], list(range(len(ids), len(ids) + 16)))
                self.assertEqual(prefix["decode_rope_positions"],
                                 [p + prefix["rope_delta"] for p in prefix["decode_kv_positions"]])
                for span, grid in zip(prefix["image_spans"], prefix["image_grid_thw"]):
                    start, length = span["start"], span["length"]
                    self.assertEqual(length, np.prod(grid) // 4)
                    self.assertEqual(prefix["mm_token_type_ids"][start:start + length], [1] * length)
                    self.assertEqual(len(set(ids[start:start + length])), 1)
                    self.assertTrue((directory / span["embedding_file"]).is_file())

    def test_rejects_resize_and_unknown_profiles(self):
        for shape in ((255, 256, 3), (32, 32, 3), (2048, 2048, 3)):
            with self.assertRaises(ValueError):
                vision.patches_without_resize(np.zeros(shape, dtype=np.uint8))
        with self.assertRaises(ValueError):
            vision.expected_shapes(dict(depth=1, hidden_size=2, intermediate_size=3, out_hidden_size=4))
        with self.assertRaises(ValueError):
            vision.unique_object([("a", 1), ("a", 2)])


if __name__ == "__main__":
    unittest.main()
