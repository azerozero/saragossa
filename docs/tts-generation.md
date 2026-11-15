# Génération Qwen3-TTS

Sur `/v1/audio/speech`, `voice` sélectionne une clé de `talker_config.spk_id`
pour CustomVoice (insensible à la casse). Sans ce champ, le locuteur est
`serena`. Une voix inconnue produit HTTP 400, paramètre `voice`. Chaque requête
sans voix rétablit le défaut ; une voix précédente ne persiste pas.
VoiceDesign et Base ignorent ce champ, car ils n'ont pas de locuteur nommé.

Le préfixe CustomVoice français est : `think, think_bos, langue, think_eos,
locuteur, pad, bos`. Le locuteur est un embedding codec du `spk_id`, pas un
embedding texte. Le préfixe VoiceDesign reste identique.
Source de cet ordre : contrat du correctif TTS du 15 novembre 2026 ; aucun
paquet Python `qwen_tts`, `qwen3_tts` ou `mlx_audio` trouvé dans les installations
locales inspectées. Les identifiants viennent exclusivement du `config.json`
du snapshot chargé. Il ne s'agit pas d'une vérification de la référence amont.
