"""Vérifie le diagnostic du bench sans modèle ni GPU."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class BenchFailureTest(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "binaire synthétique POSIX")
    def test_binary_stderr_is_shown_and_preserved_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "failure"
            binary.write_text("#!/bin/sh\nprintf 'échec synthétique SSD\\n' >&2\nexit 7\n")
            binary.chmod(0o700)
            output = root / "reports"
            result = subprocess.run(
                [sys.executable, "-I", "-B", str(Path(__file__).with_name("bench_prefix_ssd.py")),
                 "--model", str(root), "--binary", str(binary), "--tokens", "32768",
                 "--output", str(output)],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("échec synthétique SSD", result.stderr)
            self.assertIn("32768 tokens (baseline)", result.stderr)
            log = output / "32768-baseline.log"
            self.assertIn(str(log), result.stderr)
            self.assertEqual(log.read_text(), "échec synthétique SSD\n")
            self.assertFalse((output / "32768-cold.log").exists())


if __name__ == "__main__":
    unittest.main()
