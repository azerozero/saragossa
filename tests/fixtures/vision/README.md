# Fixtures vision — entrées et référence LLM

Ces fichiers contiennent des **sorties et entrées synthétiques**, aucun poids.
Le [contrat](../../../docs/vision-spec.md) précise les portes et limites.

- `manifest.json` : 1 296 tenseurs vision identifiés sur quatre checkpoints,
  SHA256 des payloads, formes, layouts, processeurs, versions et sources.
- `images/` : trois images RGB uint8 déterministes et leurs grilles. Les patchs
  f32 sont recalculés depuis les RGB ; leurs formes et SHA256 restent vérifiés.
- Chaque répertoire de checkpoint contient trois embeddings f32 issus de vrais
  poids BF16 promus.
- `tables/` : tables d'interpolation int64/f32 capturées depuis
  `fast_pos_embed_interpolate`, identiques entre les quatre checkpoints et
  stockées une seule fois. Leur rejeu est exact.
- `mrope-inputs.json` : six préfixes par modèle (trois multi-images), spans
  d'insertion, positions Python exactes, delta signé et 16 tokens imposés.
- `Qwen3.8-27B-BF16/llm-calibration.json` : capture Transformers f32/BF16
  du vocabulaire entier, 17 états par prompt, bruit observé et seuils figés
  à exactement 2× ce bruit. Les douze payloads `llm-*.bin` contiennent les
  logits, sans poids. Le manifeste authentifie les embeddings, la
  configuration, les 851 tenseurs LLM et les sources exécutées.

Les `.bin` sont little-endian, contigus ; chaque JSON associé porte forme,
dtype, taille et SHA256. Les indices d'interpolation sont avant permutation
2×2 ; les embeddings sont après merger, dans l'ordre des placeholders.
Le delta RoPE ne change pas les positions KV.

L'oracle LLM utilise les embeddings fournis, sans encodeur Rust. Sa
[qualification](../../../docs/vision-llm.md) distingue la référence capturée
et la comparaison du moteur. Les images redimensionnées restent hors de
cette qualification. Le SHA Git de Transformers n'est pas disponible dans
le wheel ; ses fichiers sources sont empreintés, sans inventer de SHA amont.

Vérification sans poids : `"$REF_PYTHON" -I -B tools/goldens/test_vision.py`.
