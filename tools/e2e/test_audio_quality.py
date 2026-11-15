"""Vérifie les oracles avec des erreurs de transcription connues."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_quality import minimum_seconds, transcription_scores


class AudioQuality(unittest.TestCase):
    def test_exact_and_unicode(self):
        self.assertEqual(transcription_scores("Été, bonjour !", "E\u0301TE\u0301 bonjour"),
                         {"word_similarity": 1.0, "wer": 0.0})

    def test_edit_operations(self):
        for text in ("a b", "a b c d", "a x c"):
            self.assertEqual(transcription_scores("a b c", text)["wer"], 1 / 3)
        self.assertEqual(transcription_scores("a b c", "")["wer"], 1.0)
        with self.assertRaises(ValueError):
            transcription_scores("", "a")

    def test_duration_scales_with_words(self):
        self.assertEqual(minimum_seconds("bonjour"), 1.0)
        self.assertEqual(minimum_seconds("mot " * 50), 10.0)


if __name__ == "__main__":
    unittest.main()
