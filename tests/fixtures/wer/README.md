# Emplacement réservé au corpus WER Whisper

Point 7 différé : aucun téléchargement FLEURS n’est autorisé ni effectué.
Ce répertoire ne contient ni audio nouveau, ni transcription inventée.

Après autorisation distincte, définir un manifeste avec version du corpus,
langue et partition, licence, identifiant de chaque extrait, chemin relatif,
SHA-256 du WAV, fréquence, durée et transcription de référence. Conserver la
provenance et une sélection fixe, indépendante des résultats du moteur.

Le futur test utilisera l’API publique `WhisperModel::transcribe`, rapportera
substitutions, suppressions, insertions et WER agrégé pondéré par le nombre de
mots. Définir le seuil d’acceptation avant la première mesure. Ne pas présenter
la similarité sur une seule voix embarquée comme une mesure FLEURS.
