# Bench concurrents — Qwen3.8-27B, M5 Max 128 Gio, novembre 2026

Données brutes produites par le harnais [`bench/competitors`](../../README.md),
batch 1, sur secteur. L'analyse et ses réserves sont dans [`rapport.md`](rapport.md).

| dossier | date | contenu |
|---|---|---|
| `night1/` | 4 et 6 nov. | nuit 1 (eval-S, eval-M, sampled-S) puis nuit 2 (256k, run 2), qualité (PPL, KL, `llama-bench`) |
| `night4/` | 7 nov. | saragossa seul, binaire `main`, 256k compris |
| `night5/` | 8 nov. | tous les moteurs dans la même session, eval-S et sampled-S ; `pairs_sar.txt` donne les rapports appariés |
| `night6/` | 9 nov. | saragossa et mlx-lm sur le checkpoint u4ls (`tools/quant/convert_ls.py`), concurrents GGUF inchangés ; 36 prompts disjoints en 3 blocs, 256k de saragossa et mlx-lm ; `sensors.jsonl.gz` donne les capteurs à 1 Hz |

Par dossier :

- `results.jsonl.gz` : journal complet (chargements, requêtes, sondes) ;
- `per_request.jsonl` : une ligne par requête avec ses preuves d'admissibilité ;
- `analysis.json`, `tables.md` : sortie de `analyze.py` ;
- `matrix.frozen.json`, `plan*.json`, `env-*.json` : réglages gelés, plan, versions et sha256 des binaires ;
- `prompts/expected.json` : empreintes des prompts ; `prompts.json` n'est pas publié,
  `prompts.py build` le reconstruit depuis rust-src et wikitext-2 (sha256 vérifiés) ;
- `logs/` : sortie des serveurs.

Les chemins personnels sont remplacés par `~`. Le fichier `bf16.kld` (3,8 Go,
logits de référence du KL) n'est pas publié, `quality_llama.sh` le régénère.
