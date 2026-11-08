# Bench concurrents — nuit 1 (4 nov. 2026), Qwen3.8-27B, M5 Max 128 Go, batch 1

Campagnes : réglage `50b27c1e1cb1`, évaluation `2d4bf214e0ae` (eval-S puis eval-M). Fin 11:35, code 0. Protocole v5 (`protocol_v5.md`), décisions post-validation et post-nuit 1 aux §15 bis et §15 ter. Référence des rapports T1/T3 : `llkm-ar` (llama.cpp Q4_K_M AR), fixée avant la mesure. Les rapports sont des moyennes géométriques de rapports appariés par prompt, IC 95 % bootstrap apparié (blocs → prompts → clés). Verdicts : « indistinct » si l'IC contient 1 ; « notable » si |gain| ≥ 5 %. eval-M (32k) : 1 prompt par type, résultats **indicatifs**, sans IC. « ≈ » : au moins une requête du groupe a un débit approximatif (§9.3).

Divulgation : l'auteur développe saragossa. Le drafter DFlash de saragossa est chargé en 4 bits, celui de llama.cpp en BF16.

## 1. Admissibilité (eval-S + eval-M)

| config | mesurées | admissibles | exclues (motif) | débit exact / ≈ | identité avec l'AR |
|---|---|---|---|---|---|
| llkm-ar | 66 | 66 | — | 64 / 2 | — |
| llkm-dflash | 66 | 21 | validation_failed (greedy_identity=False) ×45 | 66 / 0 | 21/66, 1er écart {'prompt': 'code4k-3', 'char': 375} |
| llkm-mtp | 66 | 65 | validation_failed (greedy_identity=False) ×1 | 64 / 2 | 65/66, 1er écart {'prompt': 'code4k-2', 'char': 531} |
| llmd-ar | 66 | 66 | — | 49 / 17 | — |
| llmd-native | 66 | 0 | failed:unsupported ×66 | 0 / 0 | — |
| llq40-ar | 66 | 66 | — | 66 / 0 | — |
| llq40-dflash | 66 | 65 | validation_failed (greedy_identity=False) ×1 | 66 / 0 | 65/66, 1er écart {'prompt': 'chat-5', 'char': 1133} |
| llq40-mtp | 66 | 65 | validation_failed (greedy_identity=False) ×1 | 66 / 0 | 65/66, 1er écart {'prompt': 'chat-5', 'char': 1133} |
| mlx-ar | 66 | 66 | — | 66 / 0 | — |
| mlx-dflash | 60 | 0 | failed:unsupported ×60 | 0 / 0 | — |
| mlx-mtp | 60 | 0 | failed:unsupported ×60 | 0 / 0 | — |
| sar-ar | 66 | 66 | — | 61 / 5 | — |
| sar-dflash | 66 | 65 | validation_failed (greedy_identity=False) ×1 | 61 / 5 | 65/66, 1er écart {'prompt': 'prose32k-1', 'char': 779} |
| sar-mtp | 66 | 65 | validation_failed (greedy_identity=False) ×1 | 61 / 5 | 65/66, 1er écart {'prompt': 'prose32k-1', 'char': 779} |

## 2. T1 — AR contre AR (rapport à llkm-ar)

### eval-S / chat

| config | débit de livraison (tok/s) | TTFT (s) | prefill interne (s) | latence à L tokens communs (s) |
|---|---|---|---|---|
| sar-ar | ≈32.21 — ≈1.18× [1.15–1.20] (30 paires, 0 manquantes), meilleur (notable) | 0.16 — 0.62× [0.60–0.65] (30 paires, 0 manquantes), meilleur (notable) | 0.15 — 0.58× [0.57–0.61] (30 paires, 0 manquantes), meilleur (notable) | 12.55 — 0.85× [0.83–0.87] (23 paires, 7 manquantes), meilleur (notable) |
| llq40-ar | ≈30.67 — ≈1.12× [1.09–1.15] (30 paires, 0 manquantes), meilleur (notable) | 0.22 — 0.85× [0.84–0.93] (30 paires, 0 manquantes), meilleur (notable) | 0.22 — 0.85× [0.83–0.92] (30 paires, 0 manquantes), meilleur (notable) | 13.23 — 0.89× [0.87–0.92] (28 paires, 2 manquantes), meilleur (notable) |
| llkm-ar | ≈27.24 | 0.26 | 0.26 | 14.90 |
| mlx-ar | ≈30.62 — ≈1.12× [1.10–1.14] (30 paires, 0 manquantes), meilleur (notable) | 0.24 — 0.93× [0.92–0.98] (30 paires, 0 manquantes), meilleur (notable) | n/d | 13.27 — 0.89× [0.87–0.91] (28 paires, 2 manquantes), meilleur (notable) |
| llmd-ar | ≈25.64 — ≈0.94× [0.91–0.96] (30 paires, 0 manquantes), moins bon (notable) | 0.95 — 3.72× [3.65–4.05] (30 paires, 0 manquantes), moins bon (notable) | n/d | 16.51 — 1.11× [1.09–1.15] (21 paires, 9 manquantes), moins bon (notable) |

### eval-S / code

| config | débit de livraison (tok/s) | TTFT (s) | prefill interne (s) | latence à L tokens communs (s) |
|---|---|---|---|---|
| sar-ar | 30.96 — 1.15× [1.10–1.20] (15 paires, 0 manquantes), meilleur (notable) | 5.01 — 0.81× [0.75–0.90] (15 paires, 0 manquantes), meilleur (notable) | 4.98 — 0.81× [0.75–0.90] (15 paires, 0 manquantes), meilleur (notable) | 17.87 — 0.87× [0.83–0.89] (15 paires, 0 manquantes), meilleur (notable) |
| llq40-ar | 30.07 — 1.12× [1.06–1.16] (15 paires, 0 manquantes), meilleur (notable) | 5.41 — 0.88× [0.80–1.00] (15 paires, 0 manquantes), indistinct | 5.41 — 0.88× [0.80–1.00] (15 paires, 0 manquantes), indistinct | 18.75 — 0.90× [0.87–0.93] (15 paires, 0 manquantes), meilleur (notable) |
| llkm-ar | 26.79 | 5.99 | 5.98 | 20.79 |
| mlx-ar | 29.95 — 1.11× [1.07–1.15] (15 paires, 0 manquantes), meilleur (notable) | 4.81 — 0.78× [0.69–0.91] (15 paires, 0 manquantes), meilleur (notable) | n/d | 18.08 — 0.88× [0.84–0.90] (15 paires, 0 manquantes), meilleur (notable) |
| llmd-ar | ≈24.17 — ≈0.90× [0.86–0.92] (15 paires, 0 manquantes), moins bon (notable) | 15.69 — 2.53× [2.42–2.70] (15 paires, 0 manquantes), moins bon (notable) | n/d | 32.19 — 1.56× (8 paires, 7 manquantes), indicatif (moins de 3 prompts) |

### eval-S / prose

| config | débit de livraison (tok/s) | TTFT (s) | prefill interne (s) | latence à L tokens communs (s) |
|---|---|---|---|---|
| sar-ar | 30.98 — 1.14× [1.11–1.16] (15 paires, 0 manquantes), meilleur (notable) | 5.20 — 0.84× [0.76–0.92] (15 paires, 0 manquantes), meilleur (notable) | 5.17 — 0.84× [0.76–0.91] (15 paires, 0 manquantes), meilleur (notable) | 18.06 — 0.87× [0.84–0.89] (15 paires, 0 manquantes), meilleur (notable) |
| llq40-ar | 30.04 — 1.11× [1.07–1.13] (15 paires, 0 manquantes), meilleur (notable) | 5.44 — 0.90× [0.80–0.99] (15 paires, 0 manquantes), meilleur (notable) | 5.43 — 0.90× [0.80–0.99] (15 paires, 0 manquantes), meilleur (notable) | 18.76 — 0.90× [0.88–0.93] (15 paires, 0 manquantes), meilleur (notable) |
| llkm-ar | 26.99 | 6.02 | 6.01 | 20.88 |
| mlx-ar | 29.91 — 1.10× [1.07–1.13] (15 paires, 0 manquantes), meilleur (notable) | 4.81 — 0.81× [0.70–0.90] (15 paires, 0 manquantes), meilleur (notable) | n/d | 18.16 — 0.88× [0.85–0.91] (15 paires, 0 manquantes), meilleur (notable) |
| llmd-ar | 24.19 — 0.89× [0.86–0.91] (15 paires, 0 manquantes), moins bon (notable) | 15.69 — 2.59× [2.36–2.67] (15 paires, 0 manquantes), moins bon (notable) | n/d | 32.19 — 1.55× [1.53–1.57] (15 paires, 0 manquantes), moins bon (notable) |

### eval-M / code (indicatif)

| config | débit de livraison (tok/s) | TTFT (s) | prefill interne (s) | latence à L tokens communs (s) |
|---|---|---|---|---|
| sar-ar | 27.89 — 1.12× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 44.85 — 0.78× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 44.81 — 0.78× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 59.15 — 0.81× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llq40-ar | 26.45 — 1.07× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 52.33 — 0.92× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 52.31 — 0.92× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 67.42 — 0.92× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llkm-ar | 24.81 | 57.14 | 57.12 | 73.22 |
| mlx-ar | 27.02 — 1.09× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 42.60 — 0.75× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | n/d | 57.37 — 0.78× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llmd-ar | ≈17.49 — ≈0.70× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 139.76 — 2.45× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | n/d | n/d |

### eval-M / prose (indicatif)

| config | débit de livraison (tok/s) | TTFT (s) | prefill interne (s) | latence à L tokens communs (s) |
|---|---|---|---|---|
| sar-ar | 27.94 — 1.19× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 47.00 — 0.85× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 46.96 — 0.85× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 61.17 — 0.84× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llq40-ar | 26.28 — 1.12× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 54.66 — 0.98× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 54.63 — 0.98× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 69.87 — 0.96× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llkm-ar | 23.55 | 55.51 | 55.48 | 72.47 |
| mlx-ar | 26.99 — 1.15× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 44.90 — 0.81× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | n/d | 59.74 — 0.82× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llmd-ar | 17.51 — 0.74× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 139.82 — 2.52× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | n/d | 162.63 — 2.24× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |

## 3. T2 — gain spéculatif contre l'AR du même moteur et du même format

### eval-S / chat

| config | débit de livraison (tok/s) | TTFT (s) | prefill interne (s) | latence à L tokens communs (s) |
|---|---|---|---|---|
| sar-mtp (contre sar-ar) | ≈43.35 — ≈1.34× [1.26–1.43] (30 paires, 0 manquantes), meilleur (notable) | 0.15 — 0.96× [0.95–1.00] (30 paires, 0 manquantes), meilleur (< 5 %) | 0.17 — 1.11× [1.09–1.15] (30 paires, 0 manquantes), moins bon (notable) | 9.47 — 0.77× [0.73–0.81] (25 paires, 0 manquantes), meilleur (notable) |
| sar-dflash (contre sar-ar) | ≈58.67 — ≈1.69× [1.48–1.90] (30 paires, 0 manquantes), meilleur (notable) | 0.17 — 1.06× [1.04–1.08] (30 paires, 0 manquantes), moins bon (notable) | 0.17 — 1.16× [1.13–1.18] (30 paires, 0 manquantes), moins bon (notable) | 7.24 — 0.62× [0.55–0.71] (25 paires, 0 manquantes), meilleur (notable) |
| llq40-mtp (contre llq40-ar) | 40.07 — 1.30× [1.26–1.35] (29 paires, 1 manquantes), meilleur (notable) | 0.23 — 1.07× [1.01–1.09] (29 paires, 1 manquantes), moins bon (notable) | 0.23 — 1.07× [1.01–1.09] (29 paires, 1 manquantes), moins bon (notable) | 10.20 — 0.78× [0.74–0.80] (29 paires, 1 manquantes), meilleur (notable) |
| llq40-dflash (contre llq40-ar) | 49.35 — 1.62× [1.51–1.74] (29 paires, 1 manquantes), meilleur (notable) | 0.23 — 1.08× [1.02–1.10] (29 paires, 1 manquantes), moins bon (notable) | 0.23 — 1.08× [1.02–1.10] (29 paires, 1 manquantes), moins bon (notable) | 8.31 — 0.62× [0.58–0.67] (29 paires, 1 manquantes), meilleur (notable) |
| llkm-mtp (contre llkm-ar) | ≈37.81 — ≈1.37× [1.31–1.42] (30 paires, 0 manquantes), meilleur (notable) | 0.26 — 1.04× [1.00–1.10] (30 paires, 0 manquantes), moins bon (< 5 %) | 0.26 — 1.04× [1.00–1.10] (30 paires, 0 manquantes), moins bon (< 5 %) | 10.82 — 0.74× [0.71–0.77] (28 paires, 0 manquantes), meilleur (notable) |
| llkm-dflash (contre llkm-ar) | ≈43.24 — ≈1.58× [1.53–1.69] (7 paires, 23 manquantes), meilleur (notable) | 0.26 — 1.02× [0.99–1.05] (7 paires, 23 manquantes), indistinct | 0.26 — 1.02× [0.99–1.05] (7 paires, 23 manquantes), indistinct | 9.48 — 0.64× [0.60–0.66] (7 paires, 21 manquantes), meilleur (notable) |

### eval-S / code

| config | débit de livraison (tok/s) | TTFT (s) | prefill interne (s) | latence à L tokens communs (s) |
|---|---|---|---|---|
| sar-mtp (contre sar-ar) | 46.31 — 1.48× [1.42–1.54] (15 paires, 0 manquantes), meilleur (notable) | 5.39 — 1.06× [0.97–1.14] (15 paires, 0 manquantes), indistinct | 5.43 — 1.07× [0.98–1.16] (15 paires, 0 manquantes), indistinct | 14.02 — 0.78× [0.75–0.81] (15 paires, 0 manquantes), meilleur (notable) |
| sar-dflash (contre sar-ar) | 55.24 — 1.79× [1.74–1.84] (15 paires, 0 manquantes), meilleur (notable) | 5.64 — 1.11× [1.02–1.17] (15 paires, 0 manquantes), moins bon (notable) | 5.65 — 1.11× [1.02–1.18] (15 paires, 0 manquantes), moins bon (notable) | 12.80 — 0.71× [0.69–0.73] (15 paires, 0 manquantes), meilleur (notable) |
| llq40-mtp (contre llq40-ar) | 40.63 — 1.33× [1.30–1.39] (15 paires, 0 manquantes), meilleur (notable) | 6.07 — 1.10× [0.97–1.21] (15 paires, 0 manquantes), indistinct | 6.06 — 1.10× [0.96–1.21] (15 paires, 0 manquantes), indistinct | 15.90 — 0.85× [0.82–0.87] (15 paires, 0 manquantes), meilleur (notable) |
| llq40-dflash (contre llq40-ar) | 51.85 — 1.72× [1.66–1.78] (15 paires, 0 manquantes), meilleur (notable) | 5.89 — 1.08× [0.94–1.18] (15 paires, 0 manquantes), indistinct | 5.88 — 1.08× [0.94–1.18] (15 paires, 0 manquantes), indistinct | 13.69 — 0.72× [0.70–0.75] (15 paires, 0 manquantes), meilleur (notable) |
| llkm-mtp (contre llkm-ar) | 37.82 — 1.40× [1.29–1.51] (14 paires, 1 manquantes), meilleur (notable) | 6.35 — 1.04× [0.95–1.15] (14 paires, 1 manquantes), indistinct | 6.34 — 1.04× [0.95–1.16] (14 paires, 1 manquantes), indistinct | 17.09 — 0.82× [0.79–0.84] (14 paires, 1 manquantes), meilleur (notable) |
| llkm-dflash (contre llkm-ar) | 42.75 — 1.61× (5 paires, 10 manquantes), indicatif (moins de 3 prompts) | 6.04 — 1.02× (5 paires, 10 manquantes), indicatif (moins de 3 prompts) | 6.04 — 1.02× (5 paires, 10 manquantes), indicatif (moins de 3 prompts) | 15.21 — 0.74× (5 paires, 10 manquantes), indicatif (moins de 3 prompts) |

### eval-S / prose

| config | débit de livraison (tok/s) | TTFT (s) | prefill interne (s) | latence à L tokens communs (s) |
|---|---|---|---|---|
| sar-mtp (contre sar-ar) | 44.30 — 1.42× [1.37–1.47] (15 paires, 0 manquantes), meilleur (notable) | 5.27 — 1.03× [0.92–1.14] (15 paires, 0 manquantes), indistinct | 5.32 — 1.04× [0.93–1.15] (15 paires, 0 manquantes), indistinct | 14.45 — 0.80× [0.76–0.83] (15 paires, 0 manquantes), meilleur (notable) |
| sar-dflash (contre sar-ar) | 45.87 — 1.50× [1.39–1.65] (15 paires, 0 manquantes), meilleur (notable) | 5.63 — 1.10× [0.95–1.16] (15 paires, 0 manquantes), indistinct | 5.64 — 1.11× [0.95–1.17] (15 paires, 0 manquantes), indistinct | 14.14 — 0.78× [0.73–0.83] (15 paires, 0 manquantes), meilleur (notable) |
| llq40-mtp (contre llq40-ar) | 39.09 — 1.30× [1.25–1.36] (15 paires, 0 manquantes), meilleur (notable) | 5.98 — 1.11× [0.95–1.23] (15 paires, 0 manquantes), indistinct | 5.98 — 1.11× [0.95–1.23] (15 paires, 0 manquantes), indistinct | 16.39 — 0.87× [0.84–0.90] (15 paires, 0 manquantes), meilleur (notable) |
| llq40-dflash (contre llq40-ar) | 45.21 — 1.51× [1.43–1.60] (15 paires, 0 manquantes), meilleur (notable) | 5.85 — 1.08× [0.99–1.13] (15 paires, 0 manquantes), indistinct | 5.85 — 1.08× [0.99–1.13] (15 paires, 0 manquantes), indistinct | 14.87 — 0.79× [0.76–0.81] (15 paires, 0 manquantes), meilleur (notable) |
| llkm-mtp (contre llkm-ar) | 37.61 — 1.38× [1.32–1.48] (15 paires, 0 manquantes), meilleur (notable) | 6.65 — 1.09× [1.00–1.15] (15 paires, 0 manquantes), moins bon (notable) | 6.64 — 1.09× [1.00–1.15] (15 paires, 0 manquantes), moins bon (notable) | 17.13 — 0.82× [0.79–0.85] (15 paires, 0 manquantes), meilleur (notable) |
| llkm-dflash (contre llkm-ar) | 39.83 — 1.48× [1.30–1.59] (7 paires, 8 manquantes), meilleur (notable) | 6.23 — 1.04× [0.89–1.12] (7 paires, 8 manquantes), indistinct | 6.23 — 1.04× [0.93–1.12] (7 paires, 8 manquantes), indistinct | 16.57 — 0.78× [0.74–0.81] (7 paires, 8 manquantes), meilleur (notable) |

### eval-M / code (indicatif)

| config | débit de livraison (tok/s) | TTFT (s) | prefill interne (s) | latence à L tokens communs (s) |
|---|---|---|---|---|
| sar-mtp (contre sar-ar) | 38.70 — 1.39× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 47.69 — 1.06× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 47.86 — 1.07× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 58.09 — 0.98× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| sar-dflash (contre sar-ar) | 49.00 — 1.76× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 46.60 — 1.04× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 46.60 — 1.04× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 54.78 — 0.93× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llq40-mtp (contre llq40-ar) | 35.39 — 1.34× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 54.72 — 1.05× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 54.70 — 1.05× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 66.00 — 0.98× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llq40-dflash (contre llq40-ar) | 44.63 — 1.69× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 55.72 — 1.06× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 55.70 — 1.06× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 64.68 — 0.96× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llkm-mtp (contre llkm-ar) | 32.87 — 1.32× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 57.62 — 1.01× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 57.60 — 1.01× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 69.76 — 0.95× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llkm-dflash (contre llkm-ar) | 38.45 — 1.63× (1 paires, 2 manquantes), indicatif (moins de 3 prompts) | 58.49 — 1.07× (1 paires, 2 manquantes), indicatif (moins de 3 prompts) | 58.46 — 1.07× (1 paires, 2 manquantes), indicatif (moins de 3 prompts) | 68.86 — 0.96× (1 paires, 2 manquantes), indicatif (moins de 3 prompts) |

### eval-M / prose (indicatif)

| config | débit de livraison (tok/s) | TTFT (s) | prefill interne (s) | latence à L tokens communs (s) |
|---|---|---|---|---|
| sar-mtp (contre sar-ar) | 39.06 — 1.40× (2 paires, 1 manquantes), indicatif (moins de 3 prompts) | 46.01 — 1.00× (2 paires, 1 manquantes), indicatif (moins de 3 prompts) | 46.17 — 1.01× (2 paires, 1 manquantes), indicatif (moins de 3 prompts) | 56.23 — 0.93× (2 paires, 1 manquantes), indicatif (moins de 3 prompts) |
| sar-dflash (contre sar-ar) | 45.10 — 1.62× (2 paires, 1 manquantes), indicatif (moins de 3 prompts) | 47.81 — 1.04× (2 paires, 1 manquantes), indicatif (moins de 3 prompts) | 47.80 — 1.04× (2 paires, 1 manquantes), indicatif (moins de 3 prompts) | 56.67 — 0.94× (2 paires, 1 manquantes), indicatif (moins de 3 prompts) |
| llq40-mtp (contre llq40-ar) | 34.54 — 1.31× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 55.89 — 1.02× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 55.86 — 1.02× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 66.84 — 0.96× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llq40-dflash (contre llq40-ar) | 36.98 — 1.41× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 53.33 — 0.98× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 53.30 — 0.98× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 64.17 — 0.92× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llkm-mtp (contre llkm-ar) | 33.65 — 1.43× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 58.57 — 1.06× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 58.54 — 1.06× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 70.29 — 0.97× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llkm-dflash (contre llkm-ar) | 36.50 — 1.55× (1 paires, 2 manquantes), indicatif (moins de 3 prompts) | 58.85 — 1.06× (1 paires, 2 manquantes), indicatif (moins de 3 prompts) | 58.82 — 1.06× (1 paires, 2 manquantes), indicatif (moins de 3 prompts) | 69.78 — 0.96× (1 paires, 2 manquantes), indicatif (moins de 3 prompts) |

## 4. T3 — toutes les configurations disponibles (rapport à llkm-ar)

### eval-S / chat

| config | débit de livraison (tok/s) | TTFT (s) | prefill interne (s) | latence à L tokens communs (s) |
|---|---|---|---|---|
| llkm-ar | ≈27.24 | 0.26 | 0.26 | 14.90 |
| llkm-dflash | ≈43.24 — ≈1.58× [1.53–1.69] (7 paires, 23 manquantes), meilleur (notable) | 0.26 — 1.02× [0.99–1.05] (7 paires, 23 manquantes), indistinct | 0.26 — 1.02× [0.99–1.05] (7 paires, 23 manquantes), indistinct | 9.48 — 0.64× [0.60–0.66] (7 paires, 21 manquantes), meilleur (notable) |
| llkm-mtp | ≈37.81 — ≈1.37× [1.31–1.42] (30 paires, 0 manquantes), meilleur (notable) | 0.26 — 1.04× [1.00–1.10] (30 paires, 0 manquantes), moins bon (< 5 %) | 0.26 — 1.04× [1.01–1.11] (30 paires, 0 manquantes), moins bon (< 5 %) | 10.82 — 0.74× [0.71–0.77] (28 paires, 0 manquantes), meilleur (notable) |
| llmd-ar | ≈25.64 — ≈0.94× [0.91–0.96] (30 paires, 0 manquantes), moins bon (notable) | 0.95 — 3.72× [3.65–4.05] (30 paires, 0 manquantes), moins bon (notable) | n/d | 16.51 — 1.11× [1.09–1.15] (21 paires, 9 manquantes), moins bon (notable) |
| llq40-ar | ≈30.67 — ≈1.12× [1.09–1.15] (30 paires, 0 manquantes), meilleur (notable) | 0.22 — 0.85× [0.84–0.93] (30 paires, 0 manquantes), meilleur (notable) | 0.22 — 0.85× [0.83–0.93] (30 paires, 0 manquantes), meilleur (notable) | 13.23 — 0.89× [0.87–0.92] (28 paires, 2 manquantes), meilleur (notable) |
| llq40-dflash | ≈49.35 — ≈1.82× [1.70–1.96] (29 paires, 1 manquantes), meilleur (notable) | 0.23 — 0.92× [0.88–0.98] (29 paires, 1 manquantes), meilleur (notable) | 0.23 — 0.92× [0.88–0.99] (29 paires, 1 manquantes), meilleur (notable) | 8.31 — 0.55× [0.51–0.60] (28 paires, 1 manquantes), meilleur (notable) |
| llq40-mtp | ≈40.07 — ≈1.45× [1.40–1.53] (29 paires, 1 manquantes), meilleur (notable) | 0.23 — 0.91× [0.88–0.99] (29 paires, 1 manquantes), meilleur (notable) | 0.23 — 0.91× [0.88–0.99] (29 paires, 1 manquantes), meilleur (notable) | 10.20 — 0.69× [0.66–0.72] (28 paires, 1 manquantes), meilleur (notable) |
| mlx-ar | ≈30.62 — ≈1.12× [1.10–1.15] (30 paires, 0 manquantes), meilleur (notable) | 0.24 — 0.93× [0.92–0.98] (30 paires, 0 manquantes), meilleur (notable) | n/d | 13.27 — 0.89× [0.87–0.91] (28 paires, 2 manquantes), meilleur (notable) |
| sar-ar | ≈32.21 — ≈1.18× [1.15–1.20] (30 paires, 0 manquantes), meilleur (notable) | 0.16 — 0.62× [0.60–0.65] (30 paires, 0 manquantes), meilleur (notable) | 0.15 — 0.58× [0.57–0.61] (30 paires, 0 manquantes), meilleur (notable) | 12.55 — 0.85× [0.83–0.87] (23 paires, 7 manquantes), meilleur (notable) |
| sar-dflash | ≈58.67 — ≈1.99× [1.76–2.25] (30 paires, 0 manquantes), meilleur (notable) | 0.17 — 0.66× [0.64–0.69] (30 paires, 0 manquantes), meilleur (notable) | 0.17 — 0.67× [0.65–0.70] (30 paires, 0 manquantes), meilleur (notable) | 7.24 — 0.52× [0.47–0.60] (23 paires, 7 manquantes), meilleur (notable) |
| sar-mtp | ≈43.35 — ≈1.58× [1.48–1.70] (30 paires, 0 manquantes), meilleur (notable) | 0.15 — 0.59× [0.58–0.64] (30 paires, 0 manquantes), meilleur (notable) | 0.17 — 0.64× [0.63–0.69] (30 paires, 0 manquantes), meilleur (notable) | 9.47 — 0.65× [0.61–0.68] (23 paires, 7 manquantes), meilleur (notable) |

### eval-S / code

| config | débit de livraison (tok/s) | TTFT (s) | prefill interne (s) | latence à L tokens communs (s) |
|---|---|---|---|---|
| llkm-ar | 26.79 | 5.99 | 5.98 | 20.79 |
| llkm-dflash | 42.75 — 1.61× (5 paires, 10 manquantes), indicatif (moins de 3 prompts) | 6.04 — 1.02× (5 paires, 10 manquantes), indicatif (moins de 3 prompts) | 6.04 — 1.02× (5 paires, 10 manquantes), indicatif (moins de 3 prompts) | 15.21 — 0.74× (5 paires, 10 manquantes), indicatif (moins de 3 prompts) |
| llkm-mtp | 37.82 — 1.40× [1.28–1.50] (14 paires, 1 manquantes), meilleur (notable) | 6.35 — 1.04× [0.95–1.16] (14 paires, 1 manquantes), indistinct | 6.34 — 1.04× [0.95–1.15] (14 paires, 1 manquantes), indistinct | 17.09 — 0.82× [0.79–0.84] (14 paires, 1 manquantes), meilleur (notable) |
| llmd-ar | ≈24.17 — ≈0.90× [0.86–0.92] (15 paires, 0 manquantes), moins bon (notable) | 15.69 — 2.53× [2.34–2.70] (15 paires, 0 manquantes), moins bon (notable) | n/d | 32.19 — 1.56× (8 paires, 7 manquantes), indicatif (moins de 3 prompts) |
| llq40-ar | 30.07 — 1.12× [1.07–1.16] (15 paires, 0 manquantes), meilleur (notable) | 5.41 — 0.88× [0.80–1.00] (15 paires, 0 manquantes), indistinct | 5.41 — 0.88× [0.80–1.00] (15 paires, 0 manquantes), indistinct | 18.75 — 0.90× [0.87–0.93] (15 paires, 0 manquantes), meilleur (notable) |
| llq40-dflash | 51.85 — 1.93× [1.85–1.99] (15 paires, 0 manquantes), meilleur (notable) | 5.89 — 0.95× [0.87–1.07] (15 paires, 0 manquantes), indistinct | 5.88 — 0.95× [0.86–1.07] (15 paires, 0 manquantes), indistinct | 13.69 — 0.65× [0.62–0.68] (15 paires, 0 manquantes), meilleur (notable) |
| llq40-mtp | 40.63 — 1.49× [1.43–1.57] (15 paires, 0 manquantes), meilleur (notable) | 6.07 — 0.96× [0.90–1.07] (15 paires, 0 manquantes), indistinct | 6.06 — 0.96× [0.90–1.07] (15 paires, 0 manquantes), indistinct | 15.90 — 0.77× [0.73–0.79] (15 paires, 0 manquantes), meilleur (notable) |
| mlx-ar | 29.95 — 1.11× [1.06–1.15] (15 paires, 0 manquantes), meilleur (notable) | 4.81 — 0.78× [0.69–0.91] (15 paires, 0 manquantes), meilleur (notable) | n/d | 18.08 — 0.88× [0.84–0.90] (15 paires, 0 manquantes), meilleur (notable) |
| sar-ar | 30.96 — 1.15× [1.10–1.20] (15 paires, 0 manquantes), meilleur (notable) | 5.01 — 0.81× [0.75–0.90] (15 paires, 0 manquantes), meilleur (notable) | 4.98 — 0.81× [0.75–0.90] (15 paires, 0 manquantes), meilleur (notable) | 17.87 — 0.87× [0.83–0.89] (15 paires, 0 manquantes), meilleur (notable) |
| sar-dflash | 55.24 — 2.06× [1.96–2.13] (15 paires, 0 manquantes), meilleur (notable) | 5.64 — 0.90× [0.86–0.96] (15 paires, 0 manquantes), meilleur (notable) | 5.65 — 0.90× [0.86–0.96] (15 paires, 0 manquantes), meilleur (notable) | 12.80 — 0.62× [0.60–0.63] (15 paires, 0 manquantes), meilleur (notable) |
| sar-mtp | 46.31 — 1.71× [1.63–1.77] (15 paires, 0 manquantes), meilleur (notable) | 5.39 — 0.86× [0.82–0.94] (15 paires, 0 manquantes), meilleur (notable) | 5.43 — 0.87× [0.83–0.95] (15 paires, 0 manquantes), meilleur (notable) | 14.02 — 0.68× [0.65–0.70] (15 paires, 0 manquantes), meilleur (notable) |

### eval-S / prose

| config | débit de livraison (tok/s) | TTFT (s) | prefill interne (s) | latence à L tokens communs (s) |
|---|---|---|---|---|
| llkm-ar | 26.99 | 6.02 | 6.01 | 20.88 |
| llkm-dflash | 39.83 — 1.48× [1.30–1.58] (7 paires, 8 manquantes), meilleur (notable) | 6.23 — 1.04× [0.91–1.12] (7 paires, 8 manquantes), indistinct | 6.23 — 1.04× [0.89–1.12] (7 paires, 8 manquantes), indistinct | 16.57 — 0.78× [0.74–0.82] (7 paires, 8 manquantes), meilleur (notable) |
| llkm-mtp | 37.61 — 1.38× [1.32–1.48] (15 paires, 0 manquantes), meilleur (notable) | 6.65 — 1.09× [1.00–1.14] (15 paires, 0 manquantes), moins bon (notable) | 6.64 — 1.09× [1.00–1.14] (15 paires, 0 manquantes), moins bon (notable) | 17.13 — 0.82× [0.79–0.85] (15 paires, 0 manquantes), meilleur (notable) |
| llmd-ar | 24.19 — 0.89× [0.86–0.91] (15 paires, 0 manquantes), moins bon (notable) | 15.69 — 2.59× [2.36–2.67] (15 paires, 0 manquantes), moins bon (notable) | n/d | 32.19 — 1.55× [1.53–1.57] (15 paires, 0 manquantes), moins bon (notable) |
| llq40-ar | 30.04 — 1.11× [1.08–1.13] (15 paires, 0 manquantes), meilleur (notable) | 5.44 — 0.90× [0.80–0.99] (15 paires, 0 manquantes), meilleur (notable) | 5.43 — 0.90× [0.80–0.99] (15 paires, 0 manquantes), meilleur (notable) | 18.76 — 0.90× [0.88–0.93] (15 paires, 0 manquantes), meilleur (notable) |
| llq40-dflash | 45.21 — 1.67× [1.55–1.77] (15 paires, 0 manquantes), meilleur (notable) | 5.85 — 0.97× [0.86–1.04] (15 paires, 0 manquantes), indistinct | 5.85 — 0.97× [0.86–1.04] (15 paires, 0 manquantes), indistinct | 14.87 — 0.71× [0.67–0.74] (15 paires, 0 manquantes), meilleur (notable) |
| llq40-mtp | 39.09 — 1.44× [1.36–1.51] (15 paires, 0 manquantes), meilleur (notable) | 5.98 — 1.00× [0.87–1.07] (15 paires, 0 manquantes), indistinct | 5.98 — 1.00× [0.88–1.07] (15 paires, 0 manquantes), indistinct | 16.39 — 0.79× [0.75–0.81] (15 paires, 0 manquantes), meilleur (notable) |
| mlx-ar | 29.91 — 1.10× [1.07–1.13] (15 paires, 0 manquantes), meilleur (notable) | 4.81 — 0.81× [0.72–0.89] (15 paires, 0 manquantes), meilleur (notable) | n/d | 18.16 — 0.88× [0.85–0.91] (15 paires, 0 manquantes), meilleur (notable) |
| sar-ar | 30.98 — 1.14× [1.11–1.16] (15 paires, 0 manquantes), meilleur (notable) | 5.20 — 0.84× [0.77–0.92] (15 paires, 0 manquantes), meilleur (notable) | 5.17 — 0.84× [0.76–0.91] (15 paires, 0 manquantes), meilleur (notable) | 18.06 — 0.87× [0.84–0.89] (15 paires, 0 manquantes), meilleur (notable) |
| sar-dflash | 45.87 — 1.70× [1.59–1.86] (15 paires, 0 manquantes), meilleur (notable) | 5.63 — 0.93× [0.84–0.97] (15 paires, 0 manquantes), meilleur (notable) | 5.64 — 0.93× [0.84–0.97] (15 paires, 0 manquantes), meilleur (notable) | 14.14 — 0.68× [0.64–0.71] (15 paires, 0 manquantes), meilleur (notable) |
| sar-mtp | 44.30 — 1.61× [1.55–1.69] (15 paires, 0 manquantes), meilleur (notable) | 5.27 — 0.87× [0.79–0.92] (15 paires, 0 manquantes), meilleur (notable) | 5.32 — 0.88× [0.80–0.93] (15 paires, 0 manquantes), meilleur (notable) | 14.45 — 0.69× [0.66–0.72] (15 paires, 0 manquantes), meilleur (notable) |

### eval-M / code (indicatif)

| config | débit de livraison (tok/s) | TTFT (s) | prefill interne (s) | latence à L tokens communs (s) |
|---|---|---|---|---|
| llkm-ar | 24.81 | 57.14 | 57.12 | 73.22 |
| llkm-dflash | 38.45 — 1.63× (1 paires, 2 manquantes), indicatif (moins de 3 prompts) | 58.49 — 1.07× (1 paires, 2 manquantes), indicatif (moins de 3 prompts) | 58.46 — 1.07× (1 paires, 2 manquantes), indicatif (moins de 3 prompts) | 68.86 — 0.96× (1 paires, 2 manquantes), indicatif (moins de 3 prompts) |
| llkm-mtp | 32.87 — 1.32× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 57.62 — 1.01× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 57.60 — 1.01× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 69.76 — 0.95× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llmd-ar | ≈17.49 — ≈0.70× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 139.76 — 2.45× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | n/d | n/d |
| llq40-ar | 26.45 — 1.07× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 52.33 — 0.92× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 52.31 — 0.92× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 67.42 — 0.92× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llq40-dflash | 44.63 — 1.80× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 55.72 — 0.98× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 55.70 — 0.98× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 64.68 — 0.88× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llq40-mtp | 35.39 — 1.43× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 54.72 — 0.96× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 54.70 — 0.96× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 66.00 — 0.90× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| mlx-ar | 27.02 — 1.09× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 42.60 — 0.75× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | n/d | 57.37 — 0.78× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| sar-ar | 27.89 — 1.12× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 44.85 — 0.78× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 44.81 — 0.78× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 59.15 — 0.81× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| sar-dflash | 49.00 — 1.98× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 46.60 — 0.82× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 46.60 — 0.82× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 54.78 — 0.75× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| sar-mtp | 38.70 — 1.56× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 47.69 — 0.83× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 47.86 — 0.84× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 58.09 — 0.79× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |

### eval-M / prose (indicatif)

| config | débit de livraison (tok/s) | TTFT (s) | prefill interne (s) | latence à L tokens communs (s) |
|---|---|---|---|---|
| llkm-ar | 23.55 | 55.51 | 55.48 | 72.47 |
| llkm-dflash | 36.50 — 1.55× (1 paires, 2 manquantes), indicatif (moins de 3 prompts) | 58.85 — 1.06× (1 paires, 2 manquantes), indicatif (moins de 3 prompts) | 58.82 — 1.06× (1 paires, 2 manquantes), indicatif (moins de 3 prompts) | 69.78 — 0.96× (1 paires, 2 manquantes), indicatif (moins de 3 prompts) |
| llkm-mtp | 33.65 — 1.43× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 58.57 — 1.06× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 58.54 — 1.06× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 70.29 — 0.97× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llmd-ar | 17.51 — 0.74× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 139.82 — 2.52× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | n/d | 162.63 — 2.24× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llq40-ar | 26.28 — 1.12× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 54.66 — 0.98× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 54.63 — 0.98× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 69.87 — 0.96× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llq40-dflash | 36.98 — 1.57× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 53.33 — 0.96× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 53.30 — 0.96× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 64.17 — 0.89× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| llq40-mtp | 34.54 — 1.47× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 55.89 — 1.01× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 55.86 — 1.01× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 66.84 — 0.92× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| mlx-ar | 26.99 — 1.15× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 44.90 — 0.81× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | n/d | 59.74 — 0.82× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| sar-ar | 27.94 — 1.19× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 47.00 — 0.85× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 46.96 — 0.85× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) | 61.17 — 0.84× (3 paires, 0 manquantes), indicatif (moins de 3 prompts) |
| sar-dflash | 45.10 — 1.92× (2 paires, 1 manquantes), indicatif (moins de 3 prompts) | 47.81 — 0.86× (2 paires, 1 manquantes), indicatif (moins de 3 prompts) | 47.80 — 0.86× (2 paires, 1 manquantes), indicatif (moins de 3 prompts) | 56.67 — 0.78× (2 paires, 1 manquantes), indicatif (moins de 3 prompts) |
| sar-mtp | 39.06 — 1.66× (2 paires, 1 manquantes), indicatif (moins de 3 prompts) | 46.01 — 0.83× (2 paires, 1 manquantes), indicatif (moins de 3 prompts) | 46.17 — 0.83× (2 paires, 1 manquantes), indicatif (moins de 3 prompts) | 56.23 — 0.78× (2 paires, 1 manquantes), indicatif (moins de 3 prompts) |

### Classement T3 par critère (médianes, eval-S)

- eval-S/chat, débit de livraison (tok/s) : sar-dflash 58.67 > llq40-dflash 49.35 > sar-mtp 43.35 > llkm-dflash 43.24 > llq40-mtp 40.07 > llkm-mtp 37.81 > sar-ar 32.21 > llq40-ar 30.67 > mlx-ar 30.62 > llkm-ar 27.24 > llmd-ar 25.64
- eval-S/chat, TTFT (s) : sar-mtp 0.15 > sar-ar 0.16 > sar-dflash 0.17 > llq40-ar 0.22 > llq40-mtp 0.23 > llq40-dflash 0.23 > mlx-ar 0.24 > llkm-ar 0.26 > llkm-dflash 0.26 > llkm-mtp 0.26 > llmd-ar 0.95
- eval-S/chat, latence à L tokens communs (s) : sar-dflash 7.24 > llq40-dflash 8.31 > sar-mtp 9.47 > llkm-dflash 9.48 > llq40-mtp 10.20 > llkm-mtp 10.82 > sar-ar 12.55 > llq40-ar 13.23 > mlx-ar 13.27 > llkm-ar 14.90 > llmd-ar 16.51
- eval-S/code, débit de livraison (tok/s) : sar-dflash 55.24 > llq40-dflash 51.85 > sar-mtp 46.31 > llkm-dflash 42.75 > llq40-mtp 40.63 > llkm-mtp 37.82 > sar-ar 30.96 > llq40-ar 30.07 > mlx-ar 29.95 > llkm-ar 26.79 > llmd-ar 24.17
- eval-S/code, TTFT (s) : mlx-ar 4.81 > sar-ar 5.01 > sar-mtp 5.39 > llq40-ar 5.41 > sar-dflash 5.64 > llq40-dflash 5.89 > llkm-ar 5.99 > llkm-dflash 6.04 > llq40-mtp 6.07 > llkm-mtp 6.35 > llmd-ar 15.69
- eval-S/code, latence à L tokens communs (s) : sar-dflash 12.80 > llq40-dflash 13.69 > sar-mtp 14.02 > llkm-dflash 15.21 > llq40-mtp 15.90 > llkm-mtp 17.09 > sar-ar 17.87 > mlx-ar 18.08 > llq40-ar 18.75 > llkm-ar 20.79 > llmd-ar 32.19
- eval-S/prose, débit de livraison (tok/s) : sar-dflash 45.87 > llq40-dflash 45.21 > sar-mtp 44.30 > llkm-dflash 39.83 > llq40-mtp 39.09 > llkm-mtp 37.61 > sar-ar 30.98 > llq40-ar 30.04 > mlx-ar 29.91 > llkm-ar 26.99 > llmd-ar 24.19
- eval-S/prose, TTFT (s) : mlx-ar 4.81 > sar-ar 5.20 > sar-mtp 5.27 > llq40-ar 5.44 > sar-dflash 5.63 > llq40-dflash 5.85 > llq40-mtp 5.98 > llkm-ar 6.02 > llkm-dflash 6.23 > llkm-mtp 6.65 > llmd-ar 15.69
- eval-S/prose, latence à L tokens communs (s) : sar-dflash 14.14 > sar-mtp 14.45 > llq40-dflash 14.87 > llq40-mtp 16.39 > llkm-dflash 16.57 > llkm-mtp 17.13 > sar-ar 18.06 > mlx-ar 18.16 > llq40-ar 18.76 > llkm-ar 20.88 > llmd-ar 32.19

## 5. Mémoire

| config | prefill max (Go) | decode max (Go) | max à vie (Go) | mapped file résident (vmmap, eval-S / eval-M) |
|---|---|---|---|---|
| llkm-ar | 3.3 | 3.3 | 3.3 | 15.4G / 15.4G |
| llkm-dflash | 4.6 | 4.6 | 4.6 | 19.0G / 19.0G |
| llkm-mtp | 4.0 | 4.0 | 4.0 | 17.0G / 17.0G |
| llmd-ar | 39.8 | 39.8 | 43.9 | 7584K / 7584K |
| llq40-ar | 3.3 | 3.3 | 3.3 | 14.1G / 14.1G |
| llq40-dflash | 4.6 | 4.6 | 4.6 | 17.5G / 17.7G |
| llq40-mtp | 4.0 | 4.0 | 4.0 | 15.6G / 15.6G |
| mlx-ar | 27.7 | 22.4 | 27.7 | 1888K / 1888K |
| sar-ar | 25.5 | 25.5 | 25.6 | 5408K / 5408K |
| sar-dflash | 41.1 | 38.7 | 41.3 | 6000K / 6000K |
| sar-mtp | 25.1 | 26.4 | 27.7 | 5632K / 5728K |

phys_footprint exclut les poids mappés depuis le fichier : llama.cpp en mappe 14 à 19 Go, à lire avec la dernière colonne. Le maximum à vie couvre chargement, chauffe et requêtes précédentes du chargement. llmd : fractions 0,5 × 0,5.

## 6. Contrôles

- Dérive (même charge, rang et heure) : {} ; variation > 5 % signalée, sinon rien.
- Thermique : 0 requête(s) sur 912 dans un chargement bridé (pmset).
- Sonde de cache llmd : 8/8 chargements passés, rapports [0.999, 0.999, 1.0, 1.001, 1.001, 1.004, 1.007, 1.008].
- Écarts inter-événements (part < 0,5 ms / 20–50 ms / 50–100 ms) :
  - llkm-ar : 0% / 100% / 0%
  - llkm-dflash : 63% / 0% / 37%
  - llkm-mtp : 45% / 38% / 17%
  - llmd-ar : 0% / 90% / 9%
  - llq40-ar : 0% / 100% / 0%
  - llq40-dflash : 63% / 0% / 37%
  - llq40-mtp : 45% / 49% / 5%
  - mlx-ar : 0% / 99% / 1%
  - sar-ar : 0% / 100% / 0%
  - sar-dflash : 48% / 0% / 28%
  - sar-mtp : 50% / 39% / 12%

## 7. Anomalies

- Statuts d'exécution non `ok` (hors non supportés) : aucun.
- Moteurs non arrêtés proprement : 0.
- Divergence greedy llkm-mtp contre llkm-ar : 1/66 paires, premiers écarts [{'prompt': 'code4k-2', 'char': 531}].
- Divergence greedy sar-mtp contre sar-ar : 1/66 paires, premiers écarts [{'prompt': 'prose32k-1', 'char': 779}].
- Divergence greedy llq40-dflash contre llq40-ar : 1/66 paires, premiers écarts [{'prompt': 'chat-5', 'char': 1133}].
- Divergence greedy sar-dflash contre sar-ar : 1/66 paires, premiers écarts [{'prompt': 'prose32k-1', 'char': 779}].
- Divergence greedy llkm-dflash contre llkm-ar : 45/66 paires, premiers écarts [{'prompt': 'code4k-3', 'char': 375}, {'prompt': 'chat-5', 'char': 362}, {'prompt': 'chat-6', 'char': 731}, {'prompt': 'chat-2', 'char': 408}].
- Divergence greedy llq40-mtp contre llq40-ar : 1/66 paires, premiers écarts [{'prompt': 'chat-5', 'char': 1133}].

## 8. Cellules où saragossa perd, sans filtre

Pour chaque groupe et chaque critère : toute configuration concurrente dont la médiane est meilleure que celle d'une configuration saragossa, avec le rapport apparié et le verdict (indistinct compris).

- eval-S/chat, débit de livraison (tok/s) : **llq40-mtp** 40.07 contre sar-ar 32.21 — ≈1.23× [1.20–1.29] (29 paires, 1 manquantes) (llq40-mtp/sar-ar), meilleur (notable)
- eval-S/chat, débit de livraison (tok/s) : **llq40-dflash** 49.35 contre sar-ar 32.21 — ≈1.55× [1.44–1.65] (29 paires, 1 manquantes) (llq40-dflash/sar-ar), meilleur (notable)
- eval-S/chat, débit de livraison (tok/s) : **llkm-mtp** 37.81 contre sar-ar 32.21 — ≈1.16× [1.12–1.19] (30 paires, 0 manquantes) (llkm-mtp/sar-ar), meilleur (notable)
- eval-S/chat, débit de livraison (tok/s) : **llkm-dflash** 43.24 contre sar-ar 32.21 — ≈1.34× [1.26–1.41] (7 paires, 23 manquantes) (llkm-dflash/sar-ar), meilleur (notable)
- eval-S/chat, débit de livraison (tok/s) : **llq40-dflash** 49.35 contre sar-mtp 43.35 — ≈1.15× [1.10–1.20] (29 paires, 1 manquantes) (llq40-dflash/sar-mtp), meilleur (notable)
- eval-S/chat, latence à L tokens communs (s) : **llq40-mtp** 10.20 contre sar-ar 12.55 — 0.83× [0.79–0.85] (24 paires, 6 manquantes) (llq40-mtp/sar-ar), meilleur (notable)
- eval-S/chat, latence à L tokens communs (s) : **llq40-dflash** 8.31 contre sar-ar 12.55 — 0.67× [0.63–0.72] (24 paires, 6 manquantes) (llq40-dflash/sar-ar), meilleur (notable)
- eval-S/chat, latence à L tokens communs (s) : **llkm-mtp** 10.82 contre sar-ar 12.55 — 0.88× [0.85–0.92] (23 paires, 7 manquantes) (llkm-mtp/sar-ar), meilleur (notable)
- eval-S/chat, latence à L tokens communs (s) : **llkm-dflash** 9.48 contre sar-ar 12.55 — 0.76× (6 paires, 20 manquantes) (llkm-dflash/sar-ar), indicatif (moins de 3 prompts)
- eval-S/chat, latence à L tokens communs (s) : **llq40-dflash** 8.31 contre sar-mtp 9.47 — 0.88× [0.84–0.93] (24 paires, 6 manquantes) (llq40-dflash/sar-mtp), meilleur (notable)
- eval-S/code, débit de livraison (tok/s) : **llq40-mtp** 40.63 contre sar-ar 30.96 — 1.30× [1.26–1.36] (15 paires, 0 manquantes) (llq40-mtp/sar-ar), meilleur (notable)
- eval-S/code, débit de livraison (tok/s) : **llq40-dflash** 51.85 contre sar-ar 30.96 — 1.67× [1.62–1.73] (15 paires, 0 manquantes) (llq40-dflash/sar-ar), meilleur (notable)
- eval-S/code, débit de livraison (tok/s) : **llkm-mtp** 37.82 contre sar-ar 30.96 — 1.21× [1.17–1.31] (14 paires, 1 manquantes) (llkm-mtp/sar-ar), meilleur (notable)
- eval-S/code, débit de livraison (tok/s) : **llkm-dflash** 42.75 contre sar-ar 30.96 — 1.42× (5 paires, 10 manquantes) (llkm-dflash/sar-ar), indicatif (moins de 3 prompts)
- eval-S/code, débit de livraison (tok/s) : **llq40-dflash** 51.85 contre sar-mtp 46.31 — 1.13× [1.06–1.18] (15 paires, 0 manquantes) (llq40-dflash/sar-mtp), meilleur (notable)
- eval-S/code, TTFT (s) : **mlx-ar** 4.81 contre sar-ar 5.01 — 0.96× [0.88–1.05] (15 paires, 0 manquantes) (mlx-ar/sar-ar), indistinct
- eval-S/code, TTFT (s) : **mlx-ar** 4.81 contre sar-mtp 5.39 — 0.91× [0.80–1.02] (15 paires, 0 manquantes) (mlx-ar/sar-mtp), indistinct
- eval-S/code, TTFT (s) : **llq40-ar** 5.41 contre sar-dflash 5.64 — 0.97× [0.90–1.10] (15 paires, 0 manquantes) (llq40-ar/sar-dflash), indistinct
- eval-S/code, TTFT (s) : **mlx-ar** 4.81 contre sar-dflash 5.64 — 0.87× [0.77–1.01] (15 paires, 0 manquantes) (mlx-ar/sar-dflash), indistinct
- eval-S/code, prefill interne (s) : **llq40-ar** 5.41 contre sar-mtp 5.43 — 1.01× [0.93–1.11] (15 paires, 0 manquantes) (llq40-ar/sar-mtp), indistinct
- eval-S/code, prefill interne (s) : **llq40-ar** 5.41 contre sar-dflash 5.65 — 0.97× [0.90–1.10] (15 paires, 0 manquantes) (llq40-ar/sar-dflash), indistinct
- eval-S/code, latence à L tokens communs (s) : **llq40-mtp** 15.90 contre sar-ar 17.87 — 0.88× [0.86–0.91] (15 paires, 0 manquantes) (llq40-mtp/sar-ar), meilleur (notable)
- eval-S/code, latence à L tokens communs (s) : **llq40-dflash** 13.69 contre sar-ar 17.87 — 0.76× [0.72–0.79] (15 paires, 0 manquantes) (llq40-dflash/sar-ar), meilleur (notable)
- eval-S/code, latence à L tokens communs (s) : **llkm-mtp** 17.09 contre sar-ar 17.87 — 0.95× [0.91–0.97] (14 paires, 1 manquantes) (llkm-mtp/sar-ar), meilleur (< 5 %)
- eval-S/code, latence à L tokens communs (s) : **llkm-dflash** 15.21 contre sar-ar 17.87 — 0.84× (5 paires, 10 manquantes) (llkm-dflash/sar-ar), indicatif (moins de 3 prompts)
- eval-S/code, latence à L tokens communs (s) : **llq40-dflash** 13.69 contre sar-mtp 14.02 — 0.97× [0.92–1.01] (15 paires, 0 manquantes) (llq40-dflash/sar-mtp), indistinct
- eval-S/prose, débit de livraison (tok/s) : **llq40-mtp** 39.09 contre sar-ar 30.98 — 1.26× [1.22–1.32] (15 paires, 0 manquantes) (llq40-mtp/sar-ar), meilleur (notable)
- eval-S/prose, débit de livraison (tok/s) : **llq40-dflash** 45.21 contre sar-ar 30.98 — 1.47× [1.39–1.56] (15 paires, 0 manquantes) (llq40-dflash/sar-ar), meilleur (notable)
- eval-S/prose, débit de livraison (tok/s) : **llkm-mtp** 37.61 contre sar-ar 30.98 — 1.21× [1.17–1.30] (15 paires, 0 manquantes) (llkm-mtp/sar-ar), meilleur (notable)
- eval-S/prose, débit de livraison (tok/s) : **llkm-dflash** 39.83 contre sar-ar 30.98 — 1.30× [1.18–1.38] (7 paires, 8 manquantes) (llkm-dflash/sar-ar), meilleur (notable)
- eval-S/prose, débit de livraison (tok/s) : **llq40-dflash** 45.21 contre sar-mtp 44.30 — 1.03× [0.98–1.09] (15 paires, 0 manquantes) (llq40-dflash/sar-mtp), indistinct
- eval-S/prose, TTFT (s) : **mlx-ar** 4.81 contre sar-ar 5.20 — 0.96× [0.83–1.09] (15 paires, 0 manquantes) (mlx-ar/sar-ar), indistinct
- eval-S/prose, TTFT (s) : **mlx-ar** 4.81 contre sar-mtp 5.27 — 0.93× [0.85–0.98] (15 paires, 0 manquantes) (mlx-ar/sar-mtp), meilleur (notable)
- eval-S/prose, TTFT (s) : **llq40-ar** 5.44 contre sar-dflash 5.63 — 0.97× [0.92–1.14] (15 paires, 0 manquantes) (llq40-ar/sar-dflash), indistinct
- eval-S/prose, TTFT (s) : **mlx-ar** 4.81 contre sar-dflash 5.63 — 0.87× [0.81–0.97] (15 paires, 0 manquantes) (mlx-ar/sar-dflash), meilleur (notable)
- eval-S/prose, prefill interne (s) : **llq40-ar** 5.43 contre sar-dflash 5.64 — 0.97× [0.92–1.14] (15 paires, 0 manquantes) (llq40-ar/sar-dflash), indistinct
- eval-S/prose, latence à L tokens communs (s) : **llq40-mtp** 16.39 contre sar-ar 18.06 — 0.91× [0.87–0.93] (15 paires, 0 manquantes) (llq40-mtp/sar-ar), meilleur (notable)
- eval-S/prose, latence à L tokens communs (s) : **llq40-dflash** 14.87 contre sar-ar 18.06 — 0.82× [0.76–0.85] (15 paires, 0 manquantes) (llq40-dflash/sar-ar), meilleur (notable)
- eval-S/prose, latence à L tokens communs (s) : **llkm-mtp** 17.13 contre sar-ar 18.06 — 0.95× [0.90–0.98] (15 paires, 0 manquantes) (llkm-mtp/sar-ar), meilleur (notable)
- eval-S/prose, latence à L tokens communs (s) : **llkm-dflash** 16.57 contre sar-ar 18.06 — 0.90× [0.84–0.95] (7 paires, 8 manquantes) (llkm-dflash/sar-ar), meilleur (notable)
- eval-M/code, débit de livraison (tok/s) : **llq40-mtp** 35.39 contre sar-ar 27.89 — 1.27× (3 paires, 0 manquantes) (llq40-mtp/sar-ar), indicatif (moins de 3 prompts)
- eval-M/code, débit de livraison (tok/s) : **llq40-dflash** 44.63 contre sar-ar 27.89 — 1.60× (3 paires, 0 manquantes) (llq40-dflash/sar-ar), indicatif (moins de 3 prompts)
- eval-M/code, débit de livraison (tok/s) : **llkm-mtp** 32.87 contre sar-ar 27.89 — 1.18× (3 paires, 0 manquantes) (llkm-mtp/sar-ar), indicatif (moins de 3 prompts)
- eval-M/code, débit de livraison (tok/s) : **llkm-dflash** 38.45 contre sar-ar 27.89 — 1.36× (1 paires, 2 manquantes) (llkm-dflash/sar-ar), indicatif (moins de 3 prompts)
- eval-M/code, débit de livraison (tok/s) : **llq40-dflash** 44.63 contre sar-mtp 38.70 — 1.15× (3 paires, 0 manquantes) (llq40-dflash/sar-mtp), indicatif (moins de 3 prompts)
- eval-M/code, TTFT (s) : **mlx-ar** 42.60 contre sar-ar 44.85 — 0.95× (3 paires, 0 manquantes) (mlx-ar/sar-ar), indicatif (moins de 3 prompts)
- eval-M/code, TTFT (s) : **mlx-ar** 42.60 contre sar-mtp 47.69 — 0.89× (3 paires, 0 manquantes) (mlx-ar/sar-mtp), indicatif (moins de 3 prompts)
- eval-M/code, TTFT (s) : **mlx-ar** 42.60 contre sar-dflash 46.60 — 0.91× (3 paires, 0 manquantes) (mlx-ar/sar-dflash), indicatif (moins de 3 prompts)
- eval-M/code, latence à L tokens communs (s) : **mlx-ar** 57.37 contre sar-ar 59.15 — 0.97× (3 paires, 0 manquantes) (mlx-ar/sar-ar), indicatif (moins de 3 prompts)
- eval-M/code, latence à L tokens communs (s) : **mlx-ar** 57.37 contre sar-mtp 58.09 — 0.99× (3 paires, 0 manquantes) (mlx-ar/sar-mtp), indicatif (moins de 3 prompts)
- eval-M/prose, débit de livraison (tok/s) : **llq40-mtp** 34.54 contre sar-ar 27.94 — 1.24× (3 paires, 0 manquantes) (llq40-mtp/sar-ar), indicatif (moins de 3 prompts)
- eval-M/prose, débit de livraison (tok/s) : **llq40-dflash** 36.98 contre sar-ar 27.94 — 1.32× (3 paires, 0 manquantes) (llq40-dflash/sar-ar), indicatif (moins de 3 prompts)
- eval-M/prose, débit de livraison (tok/s) : **llkm-mtp** 33.65 contre sar-ar 27.94 — 1.20× (3 paires, 0 manquantes) (llkm-mtp/sar-ar), indicatif (moins de 3 prompts)
- eval-M/prose, débit de livraison (tok/s) : **llkm-dflash** 36.50 contre sar-ar 27.94 — 1.31× (1 paires, 2 manquantes) (llkm-dflash/sar-ar), indicatif (moins de 3 prompts)
- eval-M/prose, TTFT (s) : **mlx-ar** 44.90 contre sar-ar 47.00 — 0.96× (3 paires, 0 manquantes) (mlx-ar/sar-ar), indicatif (moins de 3 prompts)
- eval-M/prose, TTFT (s) : **mlx-ar** 44.90 contre sar-mtp 46.01 — 0.98× (2 paires, 1 manquantes) (mlx-ar/sar-mtp), indicatif (moins de 3 prompts)
- eval-M/prose, TTFT (s) : **mlx-ar** 44.90 contre sar-dflash 47.81 — 0.95× (2 paires, 1 manquantes) (mlx-ar/sar-dflash), indicatif (moins de 3 prompts)
- eval-M/prose, latence à L tokens communs (s) : **mlx-ar** 59.74 contre sar-ar 61.17 — 0.98× (3 paires, 0 manquantes) (mlx-ar/sar-ar), indicatif (moins de 3 prompts)

### Mémoire : saragossa perd partout (sans IC, une mesure par chargement)

Comparaison de l'empreinte maximale à vie avec l'empreinte plus le résident mappé de llama.cpp (deux colonnes du §5) :

| saragossa | concurrent | écart |
|---|---|---|
| sar-ar 25,6 Go | llq40-ar 3,3 + 14,1 ≈ 17,4 Go ; llkm-ar 3,3 + 15,4 ≈ 18,7 Go | +37 à +47 % |
| sar-mtp 27,7 Go | llq40-mtp 4,0 + 15,6 ≈ 19,6 Go | +41 % |
| sar-dflash 41,3 Go | llq40-dflash 4,6 + 17,7 ≈ 22,3 Go | +85 % (drafter 4 bits chez saragossa, BF16 chez llama.cpp) |
| sar-ar 25,6 Go | mlx-ar 27,7 Go | −8 % |

Le maximum à vie de saragossa couvre l'arène de decode, le KV f32 sous 16k et les buffers de prefill. llama.cpp mappe ses poids et n'alloue que le KV f16.

## 9. Nuit 2 (non lancée)

Contenu : contrôle llmd du prompt 256k (chargement à 262 144), qualité (ids du corpus, PPL BF16, Q4_0 pur, Q4_K_M avec KL, PPL MLX, llama-bench pp4096/tg400), échantillonné exploratoire (8 chargements, 192 requêtes), 256k (9 chargements, 9 requêtes, plafond 1 h chacune). Durée estimée ≈ 7 h, ≈ 12 h au pire. Commande : `night.sh 2`.

