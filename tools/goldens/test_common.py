"""Garde des captures : stdlib seule, aucun import de moteur ni modèle."""
import importlib
import importlib.metadata
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
