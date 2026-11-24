# Cache de préfixe : TTFT d'un tour d'agent (24 nov. 2026)

Modèle : Qwen3.8-27B 4 bits, M5 Max. Mesure : `prefill_ms` du tour N+1
(+58 à +103 tokens) d'une conversation d'agent, via
`bench/tour_agent_ttft.py`. Données brutes : journaux de bench locaux.

| Contexte | Cas | prefill_ms | Tokens réutilisés |
|---|---|---|---|
| 16k | froid | 19,0 s | 0 |
| 16k | cache RAM | 0,30 s | prompt N |
| 16k | cache SSD, avant correctif | 19,7 s | 0 |
| 16k | cache SSD, après correctif | 1,87 s | 15232 |
| 16k | édition en tête du prompt | 20,6 à 21,8 s | 0 (attendu) |
| 32k | froid | 45,1 s | 0 |
| 32k | cache RAM | 0,44 s | prompt N |
| 32k | cache SSD, après correctif | non re-mesuré | |

Le correctif (`2d53374`) fait relire par le prefill MTP la frontière alignée
la plus longue de l'index SSD : la clé exacte du prompt N+1 n'y est jamais.

## Limites connues

- `prefix_ssd skipped reason=writer_busy` : l'écrivain SSD n'a qu'une place,
  l'écriture du prompt complet est perdue s'il est occupé.
- Un hit RAM plus court reste préféré à une frontière SSD plus longue.
