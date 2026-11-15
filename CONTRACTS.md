# Contrats comportementaux du moteur

Ces contrats décrivent le comportement attendu, pas une adaptation aux sorties
actuelles. Toute modification intentionnelle doit préciser son motif et son
oracle dans la même revue. Les goldens existants et leurs tolérances restent figés.

## LLM et échantillonnage

- **LLM-1 — déterminisme** : mêmes logits, paramètres et état du sampler donnent
  les mêmes tokens. L’état avance exactement du nombre de tirages consommés.
- **LLM-2 — distribution** : les probabilités retenues sont finies, positives,
  normalisées ; top-k borne le support et top-p garde le token qui atteint le
  seuil. Les égalités sont départagées par indice.
- **LLM-3 — templates** : rôles, contenu et appels d’outils conservent leur ordre.
  Un historique Gemma invalide produit une erreur, pas une conversation corrigée.
- **LLM-4 — référence** : les chemins optimisés conservent leurs oracles CPU,
  goldens et critères de dérive existants. Un succès de compilation ne démontre
  pas la parité numérique.

## Reconnaissance vocale Whisper

- **STT-1 — couverture** : `WhisperModel::transcribe` traite tout le PCM mono
  16 kHz, en fenêtres contiguës d’au plus 30 s, sans perte ni recouvrement.
  Deux occurrences parlées séparées par du silence doivent rester présentes
  dans le texte final, dans l’ordre.
- **STT-2 — silence** : cinq secondes de zéros doivent produire un texte vide.
  C’est un critère de qualité exigé par la recette réelle, encore à mesurer sur
  les poids locaux ; le test ne masque pas une hallucination éventuelle.
- **STT-3 — arrêt** : une limite du décodeur sans EOT est une erreur explicite.
- **STT-4 — qualité** : la matrice HTTP exige une similarité de mots ≥ 0,8 et
  un WER ≤ 0,20. Le WER est la distance de Levenshtein en mots divisée par le
  nombre de mots de référence, sans plafond artificiel à 1.

Le corpus FLEURS du point 7 reste différé : voir
[son emplacement réservé](tests/fixtures/wer/README.md). Aucun score corpus
n’est déduit de la seule voix embarquée.

## Synthèse vocale

- **TTS-1 — sortie utilisable** : WAV mono PCM16, fréquence entre 8 et 192 kHz,
  corps complet et non silencieux. Durée ≥ `max(1 s, nombre de mots / 5)` pour
  les phrases de la matrice E2E. Ce plancher détecte une troncature grossière,
  sans prétendre mesurer l’intelligibilité.
- **TTS-2 — voix** : un cas CustomVoice utilise explicitement `voice` et un
  snapshot dont `tts_model_type` vaut `custom_voice`. L’acceptation du champ
  ne prouve pas que la voix demandée est utilisée.
- **TTS-3 — raisons d’arrêt et intelligibilité** : Eos / FrameCap / RepeatFrame,
  RMS final, absence de plateau et aller-retour WER ≤ 15 % sont les critères
  des points 1–6, réservés à la branche de correction TTS. Ils ne sont pas
  revendiqués comme livrés par cette branche.

## Embeddings texte CPU

- **EMB-1 — calcul** : chargement local du tokenizer et des poids, préfixe query
  ou passage, forward BERT, mean-pooling et normalisation L2. Pas de GPU.
- **EMB-2 — oracle** : le mini-BERT à une couche utilise deux directions
  orthogonales et une attention uniforme. Les 384 coordonnées de son résultat
  sont comparées à la solution analytique, pas à une deuxième exécution.
- **EMB-3 — transport** : `/v1/embeddings` conserve l’ordre des entrées et leurs
  indices, le nom de modèle, la forme JSON et les mêmes vecteurs qu’EMB-2.
  Les entrées scalaire et tableau sont exercées par le vrai serveur local.

## Entrées et flux

- **WIRE-1 — WAV et multipart** : une entrée hostile produit une erreur sans
  panique ; les limites de taille et fréquence précèdent les allocations PCM.
  Le serveur transforme les erreurs de décodage en HTTP 400 avant inférence.
- **WIRE-2 — SSE** : fractionner une même suite d’octets ne change ni les
  événements ni la réussite ou l’échec du parseur. `[DONE]` reste identifiable.
- **WIRE-3 — code partagé** : les cibles fuzz appellent les parseurs publics
  employés par le serveur et le client de mesure ; elles n’en recopient pas
  une approximation.

## Exécution des recettes

- **RUN-1 — séparation** : les gates CPU ordinaires ne téléchargent aucun poids.
  Le mini-modèle est généré dans un répertoire temporaire à partir de constantes.
- **RUN-2 — mode strict** : `nightly-real.sh` exige les assets réels et Metal,
  sérialise les tests sous gpulock, propage leur code de sortie et refuse un
  `skip:` signalé dans le journal. Un modèle absent n’est pas un succès.
- **RUN-3 — régressions** : les graines proptest sont versionnées ; les mutants
  survivants sont documentés et donnent lieu à des assertions ciblées. Ni
  suppression d’oracle, ni élargissement de tolérance pour verdir un résultat.

La [stratégie de test](docs/TEST_STRATEGY.md) relie chaque contrat à sa preuve
et distingue les gates CPU des recettes nécessitant des poids.
