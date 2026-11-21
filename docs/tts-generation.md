# Génération Qwen3-TTS

Sur `/v1/audio/speech`, `voice` sélectionne une clé de `talker_config.spk_id`
pour CustomVoice (insensible à la casse). Sans ce champ, le locuteur est
`serena`. Une voix inconnue (y compris `alloy`, `nova` et les autres noms
OpenAI) se replie sur Serena, avec HTTP 200 et l’en-tête
`X-Saragossa-TTS-Voice-Fallback: <voix demandée>` en WAV comme en streaming.
La cause d’origine est conservée dans le diagnostic stderr. Les octets de
contrôle, non ASCII et `%` du nom sont encodés `%HH` dans cet en-tête. Une
requête valide n’a pas cet en-tête. Chaque requête sans voix rétablit le défaut ;
une voix précédente ne persiste pas.
VoiceDesign et Base ignorent ce champ, car ils n'ont pas de locuteur nommé.

Le préfixe CustomVoice français est : `think, think_bos, langue, think_eos,
locuteur, pad, bos`. Le locuteur est un embedding codec du `spk_id`, pas un
embedding texte. `DEFAULT_INSTRUCT` conditionne uniquement VoiceDesign.
CustomVoice ne reçoit aucun préfixe d’instruction par défaut ; une instruction
explicitement fournie à `with_voice_design_instruction` reste utilisable. Une
instruction vide n’ajoute aucun message utilisateur, conformément à `mlx_audio`.
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

En mode échantillonné, les 15 pas du code predictor et leur tirage tiennent
dans un seul command buffer (`SARAGOSSA_TTS_CP_FUSED=0` rétablit le pas à pas
avec tirage CPU), et le codec streaming tourne sur un thread dédié en
recouvrement des trames (`SARAGOSSA_TTS_CODEC_OVERLAP=0` le rend synchrone).
Le chemin glouton n'emprunte ni l'un ni l'autre.

Les goldens audio VoiceDesign et clone appellent explicitement le chemin par
défaut sous le flag glouton (override local au thread de test). Aucune tolérance
ni fixture golden n’est modifiée.

La réponse WAV porte `X-Saragossa-TTS-Stop-Reason: eos|repeat|cap`. En streaming,
ce champ est annoncé dans `Trailer` puis envoyé après le chunk terminal, lorsque
la cause est connue. L'arrêt `repeat` est toujours journalisé, même sans trace.
`X-Saragossa-TTS-Frame-Limit-Reached` reste compatible et vaut vrai seulement
pour `cap` ; un arrêt `repeat` sur la dernière trame reste identifié comme tel.
