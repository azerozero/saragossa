# Fixtures vision — étape 0

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

Les `.bin` sont little-endian, contigus ; chaque JSON associé porte forme,
dtype, taille et SHA256. Les indices d'interpolation sont avant permutation
2×2 ; les embeddings sont après merger, dans l'ordre des placeholders.
Le delta RoPE ne change pas les positions KV.

Ces fixtures permettent de commencer l'oracle LLM de l'étape 3 sans encodeur
Rust. Les logits, la calibration BF16 et les images redimensionnées ne sont
pas encore qualifiés. Le SHA Git de Transformers n'est pas disponible dans le
wheel ; ses fichiers sources sont empreintés, sans inventer de SHA amont.

Vérification sans poids : `"$REF_PYTHON" -I -B tools/goldens/test_vision.py`.
