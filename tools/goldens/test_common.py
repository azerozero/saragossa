"""Garde des captures : stdlib seule, aucun import de moteur ni modèle."""
import importlib
import importlib.metadata
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


class CaptureVersionsTest(unittest.TestCase):
    def test_pinned_versions_are_accepted(self):
        with patch("common.importlib.metadata.version", side_effect=common.PINS.__getitem__):
            for packages in (common.TEMPLATE_PACKAGES, common.LOGITS_PACKAGES,
                             common.EMBEDDER_PACKAGES, common.WHISPER_PACKAGES):
                common.require_versions(packages)

    def test_each_script_refuses_version_drift_before_loading(self):
        for name in ("chat_templates", "logits", "embedder", "whisper"):
            module = importlib.import_module(name)
            with self.subTest(script=name), patch.object(module, "local_model") as load:
                with patch("common.importlib.metadata.version", return_value="unexpected"):
                    with self.assertRaisesRegex(RuntimeError, "Versions de capture incompatibles"):
                        module.main()
                load.assert_not_called()

    def test_missing_distribution_is_an_error(self):
        with patch("common.importlib.metadata.version",
                   side_effect=importlib.metadata.PackageNotFoundError("mlx")):
            with self.assertRaisesRegex(RuntimeError, "mlx==0.32.4 requis, trouvé absent"):
                common.require_versions(("mlx",))

    def test_each_pin_is_enforced(self):
        for package in common.PINS:
            with self.subTest(package=package):
                def installed(name):
                    return "unexpected" if name == package else common.PINS[name]
                with patch("common.importlib.metadata.version", side_effect=installed):
                    with self.assertRaises(RuntimeError):
                        common.require_versions(tuple(common.PINS))


class LogitsPrecisionTest(unittest.TestCase):
    def test_float32_is_opt_in_and_writes_a_distinct_reference(self):
        import logits

        for float32 in (False, True):
            with self.subTest(float32=float32):
                mx = MagicMock()
                mlx = MagicMock(core=mx)
                model = MagicMock()
                mlx_lm = MagicMock()
                mlx_lm.load.return_value = (model, MagicMock())
                modules = {"mlx": mlx, "mlx.core": mx, "mlx_lm": mlx_lm,
                           "mlx_lm.models.cache": MagicMock()}
                argv = ["logits.py", "--model", "gemma4"] + (["--float32"] if float32 else [])
                source = {"model": "fixture", "versions": {}}
                # Empty prompts isolate loading, dtype selection and output routing.
                # All engine modules are stubs: this test never imports MLX.
                with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
                        patch.object(logits, "require_versions"), \
                        patch.object(logits, "local_model", return_value=Path("fixture")), \
                        patch.object(logits, "PROMPTS", []), \
                        patch.object(logits, "provenance", return_value=source), \
                        patch.object(logits, "write_json") as write:
                    logits.main()
                args = write.call_args.args
                self.assertEqual(args[0:2], ("logits", "gemma4_f32" if float32 else "gemma4"))
                self.assertEqual(args[2], {"model": "gemma4", "cases": []})
                self.assertEqual(args[3].get("float32_control"), True if float32 else None)
                if float32:
                    model.set_dtype.assert_called_once_with(mx.float32)
                    mx.eval.assert_called_once_with(model.parameters.return_value)
                else:
                    model.set_dtype.assert_not_called()
                    mx.eval.assert_not_called()


if __name__ == "__main__":
    unittest.main()
