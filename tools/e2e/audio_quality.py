"""Oracles audio sans SDK, poids ni réseau."""

from difflib import SequenceMatcher
import re
import unicodedata


def words(text):
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def minimum_seconds(text):
    # Plancher conservateur : au plus cinq mots par seconde.
    return max(1.0, len(words(text)) / 5.0)


def transcription_scores(expected, actual):
    reference, hypothesis = words(expected), words(actual)
    if not reference:
        raise ValueError("référence WER vide")
    row = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        next_row = [i]
        for j, hyp in enumerate(hypothesis, 1):
            next_row.append(min(row[j] + 1, next_row[-1] + 1,
                                row[j - 1] + (ref != hyp)))
        row = next_row
    return {"word_similarity": SequenceMatcher(None, reference, hypothesis, autojunk=False).ratio(),
            "wer": row[-1] / len(reference)}
