# Génération Qwen3-TTS

Sur `/v1/audio/speech`, `voice` sélectionne une clé de `talker_config.spk_id`
pour CustomVoice (insensible à la casse). Sans ce champ, le locuteur est
`serena`. Une voix inconnue produit HTTP 400, paramètre `voice`. Chaque requête
sans voix rétablit le défaut ; une voix précédente ne persiste pas.
VoiceDesign et Base ignorent ce champ, car ils n'ont pas de locuteur nommé.

Le préfixe CustomVoice français est : `think, think_bos, langue, think_eos,
locuteur, pad, bos`. Le locuteur est un embedding codec du `spk_id`, pas un
embedding texte. Le préfixe VoiceDesign reste identique.
Source vérifiée localement : `mlx_audio 0.4.3`,
`tts/models/qwen3_tts/qwen3_tts.py`, `_prepare_inputs` (lignes 348–436) et
`_sample_token` (lignes 750–780). Le locuteur utilise les embeddings codec et
précède le suffixe pad/bos. La pénalité porte une fois sur chaque identifiant
présent dans l’historique : multiplication si négatif, division sinon.
SHA-256 de cette source :
`fcb37415039f5f0b67377c6fe5cb19010f3d91b7c8a3f1bdc623fccf688e13d2`.
Les identifiants viennent exclusivement du `config.json` du snapshot chargé.

VoiceDesign et CustomVoice chargent `generation_config.json` : `do_sample`,
`temperature`, `top_k`, `top_p`, `repetition_penalty`, les quatre réglages
`subtalker_*` et `max_new_tokens`. En l’absence du fichier, le moteur reprend
la génération gloutonne historique et avertit une seule fois sur stderr. Un
fichier présent mais illisible ou invalide reste une erreur explicite. Les budgets
serveur restent prioritaires s'ils sont plus courts.
La pénalité du talker porte sur les cb0 déjà générés, une fois par identifiant :
logit négatif multiplié, logit positif divisé, avant température et top-k/top-p.
EOS reste exempt ; le code predictor n'a pas de pénalité inter-codebooks.

La graine hors clonage vaut 0 par défaut ; l'extension JSON `seed` accepte un
entier non signé et est réinitialisée pour chaque requête. Batch et streaming
partagent la même politique. `SARAGOSSA_TTS_GREEDY=1` force l'oracle historique
sur les deux niveaux, sans sampling ni pénalité. Les méthodes explicitement
nommées `synthesize_greedy*` restent également des oracles.

Les goldens audio VoiceDesign et clone appellent explicitement le chemin par
défaut sous le flag glouton (override local au thread de test). Aucune tolérance
ni fixture golden n’est modifiée.

La réponse WAV porte `X-Saragossa-TTS-Stop-Reason: eos|repeat|cap`. En streaming,
ce champ est annoncé dans `Trailer` puis envoyé après le chunk terminal, lorsque
la cause est connue. L'arrêt `repeat` est toujours journalisé, même sans trace.
`X-Saragossa-TTS-Frame-Limit-Reached` reste compatible et vaut vrai seulement
pour `cap` ; un arrêt `repeat` sur la dernière trame reste identifié comme tel.
