# Voix de référence (fixtures de test TTS)

Copie embarquée de `voices/` du dépôt reti, pour que les tests de parité
TTS (clone, mimi, speaker, golden) tournent aussi dans le miroir public
`saragossa` : `scripts/publish-saragossa.sh` ne synchronise que
`crates/saragossa`, donc un chemin `../../voices/` n'y arrive jamais.

- `reti-fr.wav` : extrait FR studio, voix de référence pour le clonage.
- `reti-fr.txt` : transcript de la référence (requis par le clonage ICL).
- Source : CML-TTS (locuteur 12080_11650), via `kyutai/tts-voices`, version
  *enhanced* (débruitée).
- Licence : **CC-BY 4.0**. Attribution : CML-TTS / Kyutai tts-voices.

Résolution : `test_support::voice_fixture` cherche ce répertoire d'abord,
puis retombe sur `../../voices/`. Les deux copies doivent rester
identiques tant qu'elles coexistent (`cmp` dans les tests).
