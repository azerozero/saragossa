"""Contrats CPU du protocole de routage, sans charger de modèle."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "bench", Path(__file__).with_name("bench_spec_route_pairwise.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class ProtocolTests(unittest.TestCase):
    def test_flags_isolated(self):
        with patch.dict(bench.os.environ, {"SARAGOSSA_RUST_SPEC_ROUTE": "1"}):
            env = bench.environment(Path("/d"))
        self.assertEqual(env["SARAGOSSA_RUST_MTP_LOOKUP_DRAFT"], "0")
        self.assertEqual(env["SARAGOSSA_RUST_DFLASH_DRAFT_DIR"], "/d")
        self.assertNotIn("SARAGOSSA_RUST_SPEC_ROUTE", env)

    def test_every_pair_has_abba_order(self):
        modes = bench.mode_order(0)
        schedule = modes + bench.mode_order(1)
        for index, first in enumerate(modes):
            for second in modes[index + 1:]:
                self.assertEqual([m for m in schedule if m in (first, second)],
                                 [first, second, second, first])

    def test_route_is_paired_with_best(self):
        rows = []
        for repeat, base in enumerate([10, 100]):
            for mode, rate in [("mtp", base), ("dflash", base * 2), ("route", base * 2.2)]:
                rows.append(dict(case="C", mode=mode, repeat=repeat, warmup=False,
                    decode_tok_s=rate, ttft_ms=20, total_ms=40, accepted=1,
                    proposed=2, ids_equal=True, routed="dflash"))
        rows.append(dict(rows[0], warmup=True, decode_tok_s=10000))
        result = bench.summarize(rows)[0]
        self.assertEqual(result["best"], "dflash")
        self.assertAlmostEqual(result["route_vs_best_pct"], 10)


if __name__ == "__main__":
    unittest.main()
