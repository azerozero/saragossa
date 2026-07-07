# Fixtures golden de parité metal-rs ↔ référence figée

Sorties de référence figées pour les oracles de parité STT/TTS/clone de
`crates/saragossa`. Elles permettent aux tests `golden_*` de comparer metal-rs à une
référence stable sans mlx-rs.

- **Provenance** : reti HEAD `be456e2`, fork mlx-rs vendoré (`vendor/mlx-rs`, MLX 0.31.2).
- **Capture** : 2026-06-13, via `RETI_GOLDEN_CAPTURE=1 cargo test -p saragossa --features metal -- --ignored live_`.
- **Format** : `<name>.bin` (octets little-endian bruts, dtype dans le `.json`) + `<name>.json`
  (en-tête `dtype`/`shape`/`tolerance`/`source`/`mlx_sha`/`captured`). Transcription en `.txt`.
- **Recapture** : modifier `RETI_GOLDEN_CAPTURE=1` et relancer la commande de capture
  pour régénérer les `.bin/.json`.

Chaque fixture est consommée par le test `golden_*` homonyme.
Voir `/tmp/golden_oracles_plan.md` pour le mapping complet entrée → référence → tolérance.
