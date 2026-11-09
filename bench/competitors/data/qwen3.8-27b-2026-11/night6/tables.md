
### eval-L-mlx / code — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| mlx-ar | 14.39 | — | — |
| sar-ar | 14.59 | — | — |
| sar-dflash | 34.04 | — | — |
| sar-mtp | 21.40 | — | — |

### eval-L-mlx / code — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| mlx-ar | 719.62 | — | — |
| sar-ar | 734.12 | — | — |
| sar-dflash | 728.30 | — | — |
| sar-mtp | 761.89 | — | — |

### eval-L-mlx / code — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| mlx-ar | 747.36 | — | — |
| sar-ar | 761.47 | — | — |
| sar-dflash | 740.02 | — | — |
| sar-mtp | 780.54 | — | — |

### eval-L-mlx / code — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| mlx-ar | 747.36 | 400 | 1 |
| sar-ar | 761.47 | 400 | 1 |
| sar-dflash | 740.02 | 400 | 1 |
| sar-mtp | 780.54 | 400 | 1 |

### eval-L-mlx / code — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| sar-dflash | sar-ar | 2.33× (1 paires, 0 manquantes) | 0.99× (1 paires, 0 manquantes) | 0.97× (1 paires, 0 manquantes) |
| sar-mtp | sar-ar | 1.47× (1 paires, 0 manquantes) | 1.04× (1 paires, 0 manquantes) | 1.03× (1 paires, 0 manquantes) |

### eval-S / chat — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 27.18 | — | — |
| llkm-dflash | 48.86 | 1.80× (2 paires, 16 manquantes) | indicatif (moins de 3 prompts) |
| llkm-mtp | 37.50 | 1.36× [1.33–1.41] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar | ≈25.51 | ≈0.93× [0.91–0.94] IC approximatif (3 blocs) (18 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 30.62 | 1.11× [1.09–1.13] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 51.07 | 1.79× [1.68–1.90] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| llq40-mtp | 39.56 | 1.44× [1.40–1.49] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| mlx-ar | 30.53 | 1.11× [1.09–1.12] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 32.19 | 1.17× [1.15–1.18] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 53.45 | 1.91× [1.77–2.08] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 45.16 | 1.64× [1.58–1.70] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |

### eval-S / chat — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 0.25 | — | — |
| llkm-dflash | 0.27 | 1.05× (2 paires, 16 manquantes) | indicatif (moins de 3 prompts) |
| llkm-mtp | 0.27 | 1.08× [1.02–1.15] IC approximatif (3 blocs) (18 paires, 0 manquantes) | moins bon (notable) |
| llmd-ar | 0.93 | 3.76× [3.67–3.83] IC approximatif (3 blocs) (18 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 0.22 | 0.87× [0.85–0.89] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 0.23 | 0.92× [0.90–0.94] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| llq40-mtp | 0.24 | 0.95× [0.94–0.97] IC approximatif (3 blocs) (18 paires, 0 manquantes) | indistinct |
| mlx-ar | 0.24 | 0.97× [0.93–1.00] IC approximatif (3 blocs) (18 paires, 0 manquantes) | indistinct |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 0.17 | 0.67× [0.65–0.70] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 0.18 | 0.74× [0.72–0.76] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 0.16 | 0.63× [0.62–0.65] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |

### eval-S / chat — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 14.92 | — | — |
| llkm-dflash | 8.49 | 0.56× (2 paires, 16 manquantes) | indicatif (moins de 3 prompts) |
| llkm-mtp | 10.91 | 0.74× [0.72–0.76] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar | 16.57 | 1.13× [1.11–1.14] IC approximatif (3 blocs) (17 paires, 1 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 13.25 | 0.90× [0.89–0.91] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 8.04 | 0.57× [0.53–0.60] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| llq40-mtp | 10.33 | 0.70× [0.68–0.72] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| mlx-ar | 13.31 | 0.90× [0.89–0.92] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 12.56 | 0.85× [0.84–0.87] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 7.66 | 0.53× [0.49–0.57] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 8.99 | 0.61× [0.59–0.64] IC approximatif (3 blocs) (18 paires, 0 manquantes) | meilleur (notable) |

### eval-S / chat — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 14.92 | 400 | 18 |
| llkm-dflash | 8.49 | 400 | 2 |
| llkm-mtp | 10.91 | 400 | 18 |
| llmd-ar | 16.57 | 400 | 18 |
| llq40-ar | 13.25 | 400 | 18 |
| llq40-dflash | 8.04 | 400 | 18 |
| llq40-mtp | 10.33 | 400 | 18 |
| mlx-ar | 13.31 | 400 | 18 |
| sar-ar | 12.56 | 400 | 18 |
| sar-dflash | 7.66 | 400 | 18 |
| sar-mtp | 8.99 | 400 | 18 |

### eval-S / chat — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | 1.80× (2 paires, 16 manquantes) | 1.05× (2 paires, 16 manquantes) | 0.56× (2 paires, 16 manquantes) |
| llkm-mtp | llkm-ar | 1.36× [1.33–1.41] IC approximatif (3 blocs) (18 paires, 0 manquantes) | 1.08× [1.03–1.15] IC approximatif (3 blocs) (18 paires, 0 manquantes) | 0.74× [0.72–0.76] IC approximatif (3 blocs) (18 paires, 0 manquantes) |
| llmd-native | llmd-ar | n/d | n/d | n/d |
| llq40-dflash | llq40-ar | 1.61× [1.51–1.70] IC approximatif (3 blocs) (18 paires, 0 manquantes) | 1.06× [1.04–1.08] IC approximatif (3 blocs) (18 paires, 0 manquantes) | 0.63× [0.60–0.67] IC approximatif (3 blocs) (18 paires, 0 manquantes) |
| llq40-mtp | llq40-ar | 1.30× [1.26–1.33] IC approximatif (3 blocs) (18 paires, 0 manquantes) | 1.10× [1.06–1.13] IC approximatif (3 blocs) (18 paires, 0 manquantes) | 0.78× [0.76–0.80] IC approximatif (3 blocs) (18 paires, 0 manquantes) |
| sar-dflash | sar-ar | 1.64× [1.52–1.77] IC approximatif (3 blocs) (18 paires, 0 manquantes) | 1.10× [1.07–1.13] IC approximatif (3 blocs) (18 paires, 0 manquantes) | 0.62× [0.57–0.67] IC approximatif (3 blocs) (18 paires, 0 manquantes) |
| sar-mtp | sar-ar | 1.40× [1.36–1.45] IC approximatif (3 blocs) (18 paires, 0 manquantes) | 0.95× [0.89–1.00] IC approximatif (3 blocs) (18 paires, 0 manquantes) | 0.72× [0.69–0.74] IC approximatif (3 blocs) (18 paires, 0 manquantes) |

### eval-S / code — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 26.98 | — | — |
| llkm-dflash | 44.94 | 1.67× [1.66–1.69] (3 paires, 6 manquantes) | meilleur (notable) |
| llkm-mtp | 37.79 | 1.40× [1.36–1.44] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar | ≈24.07 | ≈0.88× [0.87–0.89] IC approximatif (3 blocs) (9 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 29.97 | 1.10× [1.08–1.12] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 50.04 | 1.82× [1.75–1.89] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| llq40-mtp | 38.99 | 1.44× [1.40–1.47] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| mlx-ar | 29.92 | 1.10× [1.08–1.11] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 30.94 | 1.13× [1.11–1.15] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 55.46 | 2.03× [1.91–2.14] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 46.89 | 1.71× [1.65–1.77] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |

### eval-S / code — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 5.83 | — | — |
| llkm-dflash | 6.02 | 1.03× [1.00–1.07] (3 paires, 6 manquantes) | indistinct |
| llkm-mtp | 6.56 | 1.07× [1.03–1.12] IC approximatif (3 blocs) (9 paires, 0 manquantes) | moins bon (notable) |
| llmd-ar | 15.68 | 2.69× [2.62–2.70] IC approximatif (3 blocs) (27 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 5.46 | 0.93× [0.90–0.94] IC approximatif (3 blocs) (27 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 6.13 | 0.99× [0.96–1.02] IC approximatif (3 blocs) (9 paires, 0 manquantes) | indistinct |
| llq40-mtp | 5.73 | 0.95× [0.92–0.97] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| mlx-ar | 4.95 | 0.84× [0.80–0.88] IC approximatif (3 blocs) (27 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 5.04 | 0.86× [0.83–0.88] IC approximatif (3 blocs) (27 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 5.49 | 0.90× [0.85–0.93] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 5.29 | 0.88× [0.85–0.90] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |

### eval-S / code — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 20.76 | — | — |
| llkm-dflash | 14.90 | 0.72× [0.71–0.73] (3 paires, 6 manquantes) | meilleur (notable) |
| llkm-mtp | 16.91 | 0.82× [0.81–0.83] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar | 32.26 | 1.56× [1.55–1.58] (3 paires, 6 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 18.64 | 0.90× [0.89–0.91] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 14.08 | 0.68× [0.66–0.70] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| llq40-mtp | 15.97 | 0.77× [0.76–0.78] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| mlx-ar | 17.98 | 0.88× [0.86–0.90] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 17.83 | 0.87× [0.85–0.88] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 12.55 | 0.61× [0.59–0.64] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 13.83 | 0.67× [0.66–0.68] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |

### eval-S / code — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 20.76 | 400 | 9 |
| llkm-dflash | 14.90 | 400 | 3 |
| llkm-mtp | 16.91 | 400 | 9 |
| llmd-ar | 32.26 | 400 | 9 |
| llq40-ar | 18.64 | 400 | 9 |
| llq40-dflash | 14.08 | 400 | 9 |
| llq40-mtp | 15.97 | 400 | 9 |
| mlx-ar | 17.98 | 400 | 9 |
| sar-ar | 17.83 | 400 | 9 |
| sar-dflash | 12.55 | 400 | 9 |
| sar-mtp | 13.83 | 400 | 9 |

### eval-S / code — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | 1.67× [1.66–1.69] (3 paires, 6 manquantes) | 1.03× [1.00–1.07] (3 paires, 6 manquantes) | 0.72× [0.71–0.73] (3 paires, 6 manquantes) |
| llkm-mtp | llkm-ar | 1.40× [1.37–1.44] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 1.07× [1.03–1.12] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 0.82× [0.81–0.83] IC approximatif (3 blocs) (9 paires, 0 manquantes) |
| llmd-native | llmd-ar | n/d | n/d | n/d |
| llq40-dflash | llq40-ar | 1.66× [1.60–1.70] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 1.11× [1.07–1.15] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 0.75× [0.74–0.76] IC approximatif (3 blocs) (9 paires, 0 manquantes) |
| llq40-mtp | llq40-ar | 1.31× [1.29–1.32] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 1.06× [1.04–1.09] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 0.85× [0.84–0.86] IC approximatif (3 blocs) (9 paires, 0 manquantes) |
| sar-dflash | sar-ar | 1.79× [1.69–1.89] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 1.09× [1.03–1.13] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 0.71× [0.68–0.74] IC approximatif (3 blocs) (9 paires, 0 manquantes) |
| sar-mtp | sar-ar | 1.51× [1.47–1.55] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 1.06× [1.02–1.10] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 0.77× [0.76–0.79] IC approximatif (3 blocs) (9 paires, 0 manquantes) |

### eval-S / prose — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 27.12 | — | — |
| llkm-dflash | 38.20 | 1.41× [1.34–1.45] (3 paires, 6 manquantes) | meilleur (notable) |
| llkm-mtp | 36.63 | 1.35× [1.31–1.39] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar | 24.09 | 0.88× [0.87–0.89] IC approximatif (3 blocs) (9 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 30.04 | 1.10× [1.08–1.11] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 45.48 | 1.67× [1.62–1.72] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| llq40-mtp | 39.28 | 1.43× [1.39–1.46] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| mlx-ar | 29.79 | 1.09× [1.07–1.10] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 30.97 | 1.13× [1.11–1.14] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 45.36 | 1.67× [1.57–1.80] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 45.43 | 1.66× [1.59–1.73] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |

### eval-S / prose — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 5.83 | — | — |
| llkm-dflash | 5.99 | 1.00× [0.97–1.02] (3 paires, 6 manquantes) | indistinct |
| llkm-mtp | 6.33 | 1.05× [1.00–1.12] IC approximatif (3 blocs) (9 paires, 0 manquantes) | indistinct |
| llmd-ar | 15.68 | 2.69× [2.60–2.70] IC approximatif (3 blocs) (27 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 5.46 | 0.93× [0.90–0.94] IC approximatif (3 blocs) (27 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 5.65 | 0.94× [0.89–0.99] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| llq40-mtp | 5.74 | 0.96× [0.92–1.00] IC approximatif (3 blocs) (9 paires, 0 manquantes) | indistinct |
| mlx-ar | 4.97 | 0.84× [0.80–0.88] IC approximatif (3 blocs) (27 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 5.05 | 0.87× [0.83–0.87] IC approximatif (3 blocs) (27 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 5.20 | 0.87× [0.84–0.90] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 5.16 | 0.86× [0.82–0.88] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |

### eval-S / prose — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 20.72 | — | — |
| llkm-dflash | 16.47 | 0.79× [0.78–0.81] (3 paires, 6 manquantes) | meilleur (notable) |
| llkm-mtp | 17.17 | 0.83× [0.82–0.85] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar | 32.26 | 1.56× [1.55–1.58] IC approximatif (3 blocs) (9 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 18.63 | 0.91× [0.89–0.92] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 14.41 | 0.70× [0.69–0.71] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| llq40-mtp | 16.07 | 0.78× [0.77–0.78] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| mlx-ar | 18.25 | 0.89× [0.87–0.92] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 17.88 | 0.87× [0.86–0.88] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 14.14 | 0.68× [0.65–0.71] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 14.00 | 0.68× [0.66–0.69] IC approximatif (3 blocs) (9 paires, 0 manquantes) | meilleur (notable) |

### eval-S / prose — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 20.72 | 400 | 9 |
| llkm-dflash | 16.47 | 400 | 3 |
| llkm-mtp | 17.17 | 400 | 9 |
| llmd-ar | 32.26 | 400 | 9 |
| llq40-ar | 18.63 | 400 | 9 |
| llq40-dflash | 14.41 | 400 | 9 |
| llq40-mtp | 16.07 | 400 | 9 |
| mlx-ar | 18.25 | 400 | 9 |
| sar-ar | 17.88 | 400 | 9 |
| sar-dflash | 14.14 | 400 | 9 |
| sar-mtp | 14.00 | 400 | 9 |

### eval-S / prose — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | 1.41× [1.34–1.45] (3 paires, 6 manquantes) | 1.00× [0.97–1.02] (3 paires, 6 manquantes) | 0.79× [0.78–0.81] (3 paires, 6 manquantes) |
| llkm-mtp | llkm-ar | 1.35× [1.31–1.39] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 1.05× [1.00–1.12] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 0.83× [0.82–0.85] IC approximatif (3 blocs) (9 paires, 0 manquantes) |
| llmd-native | llmd-ar | n/d | n/d | n/d |
| llq40-dflash | llq40-ar | 1.52× [1.49–1.56] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 1.05× [1.01–1.11] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 0.77× [0.76–0.79] IC approximatif (3 blocs) (9 paires, 0 manquantes) |
| llq40-mtp | llq40-ar | 1.30× [1.28–1.33] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 1.07× [1.02–1.13] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 0.86× [0.85–0.87] IC approximatif (3 blocs) (9 paires, 0 manquantes) |
| sar-dflash | sar-ar | 1.48× [1.40–1.58] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 1.05× [1.00–1.11] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 0.78× [0.74–0.82] IC approximatif (3 blocs) (9 paires, 0 manquantes) |
| sar-mtp | sar-ar | 1.47× [1.43–1.52] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 1.03× [1.01–1.06] IC approximatif (3 blocs) (9 paires, 0 manquantes) | 0.78× [0.77–0.79] IC approximatif (3 blocs) (9 paires, 0 manquantes) |

### sampled-S / chat — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | ≈27.44 | — | — |
| llkm-dflash | ≈42.31 | ≈1.52× [1.39–1.65] (12 paires, 0 manquantes) | meilleur (notable) |
| llkm-mtp | ≈37.59 | ≈1.36× [1.30–1.40] (12 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | ≈25.53 | ≈0.93× [0.92–0.93] (12 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | ≈30.36 | ≈1.10× [1.10–1.11] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | ≈31.58 | ≈1.15× [1.14–1.16] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | ≈49.84 | ≈1.75× [1.57–1.93] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | ≈43.67 | ≈1.59× [1.51–1.66] (12 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / chat — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 0.25 | — | — |
| llkm-dflash | 0.26 | 1.02× [0.99–1.05] (12 paires, 0 manquantes) | indistinct |
| llkm-mtp | 0.27 | 1.04× [1.01–1.08] (12 paires, 0 manquantes) | moins bon (< 5 %) |
| llmd-ar-s | 0.93 | 3.65× [3.55–3.73] (12 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 0.24 | 0.93× [0.90–0.95] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 0.17 | 0.66× [0.63–0.68] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 0.17 | 0.64× [0.62–0.66] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 0.15 | 0.60× [0.58–0.62] (12 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / chat — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 14.75 | — | — |
| llkm-dflash | 9.88 | 0.69× [0.65–0.75] (8 paires, 2 manquantes) | meilleur (notable) |
| llkm-mtp | 10.93 | 0.75× [0.73–0.78] (10 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | 16.56 | 1.13× [1.12–1.14] (9 paires, 1 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 13.38 | 0.91× [0.90–0.91] (10 paires, 2 manquantes) | meilleur (notable) |
| sar-ar | 12.81 | 0.87× [0.86–0.88] (10 paires, 2 manquantes) | meilleur (notable) |
| sar-dflash | 8.17 | 0.59× [0.54–0.66] (10 paires, 2 manquantes) | meilleur (notable) |
| sar-mtp | 9.29 | 0.64× [0.62–0.67] (10 paires, 2 manquantes) | meilleur (notable) |

### sampled-S / chat — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 14.81 | 400 | 12 |
| llkm-dflash | 9.67 | 400 | 12 |
| llkm-mtp | 10.88 | 400 | 12 |
| llmd-ar-s | 16.56 | 400 | 12 |
| mlx-ar | 13.38 | 400 | 12 |
| sar-ar | 12.81 | 400 | 12 |
| sar-dflash | 8.17 | 400 | 12 |
| sar-mtp | 9.29 | 400 | 12 |

### sampled-S / chat — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | ≈1.52× [1.38–1.65] (12 paires, 0 manquantes) | 1.02× [0.99–1.05] (12 paires, 0 manquantes) | 0.69× [0.65–0.75] (8 paires, 2 manquantes) |
| llkm-mtp | llkm-ar | ≈1.36× [1.30–1.40] (12 paires, 0 manquantes) | 1.04× [1.01–1.08] (12 paires, 0 manquantes) | 0.75× [0.73–0.78] (10 paires, 0 manquantes) |
| llmd-native-s | llmd-ar-s | n/d | n/d | n/d |
| sar-dflash | sar-ar | 1.52× [1.35–1.68] (12 paires, 0 manquantes) | 0.98× [0.96–1.00] (12 paires, 0 manquantes) | 0.66× [0.60–0.74] (12 paires, 0 manquantes) |
| sar-mtp | sar-ar | 1.38× [1.32–1.44] (12 paires, 0 manquantes) | 0.91× [0.90–0.93] (12 paires, 0 manquantes) | 0.73× [0.70–0.76] (12 paires, 0 manquantes) |

### sampled-S / code — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 27.76 | — | — |
| llkm-dflash | 40.42 | 1.48× [1.42–1.55] (6 paires, 0 manquantes) | meilleur (notable) |
| llkm-mtp | 35.53 | 1.28× [1.26–1.31] (6 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | ≈24.09 | ≈0.87× [0.86–0.89] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 29.65 | 1.07× [1.06–1.09] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 30.87 | 1.12× [1.10–1.14] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 49.11 | 1.82× [1.72–1.96] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 43.61 | 1.57× [1.50–1.64] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / code — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 5.99 | — | — |
| llkm-dflash | 6.42 | 1.06× [1.01–1.11] (6 paires, 0 manquantes) | moins bon (notable) |
| llkm-mtp | 6.10 | 1.01× [1.00–1.02] (6 paires, 0 manquantes) | indistinct |
| llmd-ar-s | 15.68 | 2.60× [2.57–2.63] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 4.57 | 0.76× [0.75–0.77] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 4.79 | 0.80× [0.78–0.82] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 5.21 | 0.87× [0.85–0.89] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 5.23 | 0.88× [0.84–0.93] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / code — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 20.45 | — | — |
| llkm-dflash | 16.11 | 0.79× [0.76–0.81] (6 paires, 0 manquantes) | meilleur (notable) |
| llkm-mtp | 17.31 | 0.85× [0.83–0.86] (6 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | 32.25 | 1.57× (3 paires, 3 manquantes) | indicatif (moins de 3 prompts) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 18.03 | 0.88× [0.87–0.89] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 17.72 | 0.87× [0.86–0.88] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 13.42 | 0.64× [0.61–0.67] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 14.50 | 0.71× [0.69–0.72] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / code — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 20.45 | 400 | 6 |
| llkm-dflash | 16.11 | 400 | 6 |
| llkm-mtp | 17.31 | 400 | 6 |
| llmd-ar-s | 32.24 | 400 | 6 |
| mlx-ar | 18.03 | 400 | 6 |
| sar-ar | 17.72 | 400 | 6 |
| sar-dflash | 13.42 | 400 | 6 |
| sar-mtp | 14.50 | 400 | 6 |

### sampled-S / code — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | 1.48× [1.42–1.55] (6 paires, 0 manquantes) | 1.06× [1.01–1.11] (6 paires, 0 manquantes) | 0.79× [0.76–0.81] (6 paires, 0 manquantes) |
| llkm-mtp | llkm-ar | 1.28× [1.26–1.31] (6 paires, 0 manquantes) | 1.01× [1.00–1.02] (6 paires, 0 manquantes) | 0.85× [0.83–0.86] (6 paires, 0 manquantes) |
| llmd-native-s | llmd-ar-s | n/d | n/d | n/d |
| sar-dflash | sar-ar | 1.63× [1.54–1.78] (6 paires, 0 manquantes) | 1.08× [1.04–1.12] (6 paires, 0 manquantes) | 0.74× [0.70–0.77] (6 paires, 0 manquantes) |
| sar-mtp | sar-ar | 1.41× [1.36–1.45] (6 paires, 0 manquantes) | 1.09× [1.03–1.16] (6 paires, 0 manquantes) | 0.81× [0.80–0.83] (6 paires, 0 manquantes) |

### sampled-S / prose — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 27.28 | — | — |
| llkm-dflash | 37.29 | 1.42× [1.33–1.58] (6 paires, 0 manquantes) | meilleur (notable) |
| llkm-mtp | 36.02 | 1.32× [1.28–1.35] (6 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | 24.08 | 0.88× [0.86–0.89] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 29.69 | 1.08× [1.06–1.10] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 30.77 | 1.12× [1.10–1.13] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 42.81 | 1.55× [1.47–1.63] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 43.04 | 1.59× [1.52–1.67] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / prose — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 5.96 | — | — |
| llkm-dflash | 5.96 | 1.01× [0.98–1.05] (6 paires, 0 manquantes) | indistinct |
| llkm-mtp | 6.14 | 1.04× [1.02–1.06] (6 paires, 0 manquantes) | moins bon (< 5 %) |
| llmd-ar-s | 15.67 | 2.62× [2.58–2.65] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 4.45 | 0.75× [0.73–0.76] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 4.85 | 0.83× [0.79–0.89] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 5.14 | 0.86× [0.85–0.89] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 5.34 | 0.89× [0.88–0.91] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / prose — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 20.59 | — | — |
| llkm-dflash | 16.62 | 0.79× [0.75–0.82] (6 paires, 0 manquantes) | meilleur (notable) |
| llkm-mtp | 17.21 | 0.84× [0.83–0.86] (6 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | 32.24 | 1.57× [1.56–1.59] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 17.90 | 0.87× [0.87–0.88] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 17.81 | 0.87× [0.86–0.89] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 14.42 | 0.71× [0.68–0.74] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 14.66 | 0.71× [0.69–0.73] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / prose — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 20.59 | 400 | 6 |
| llkm-dflash | 16.62 | 400 | 6 |
| llkm-mtp | 17.21 | 400 | 6 |
| llmd-ar-s | 32.24 | 400 | 6 |
| mlx-ar | 17.90 | 400 | 6 |
| sar-ar | 17.81 | 400 | 6 |
| sar-dflash | 14.42 | 400 | 6 |
| sar-mtp | 14.66 | 400 | 6 |

### sampled-S / prose — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | 1.42× [1.33–1.57] (6 paires, 0 manquantes) | 1.01× [0.98–1.04] (6 paires, 0 manquantes) | 0.79× [0.76–0.82] (6 paires, 0 manquantes) |
| llkm-mtp | llkm-ar | 1.32× [1.28–1.35] (6 paires, 0 manquantes) | 1.04× [1.02–1.06] (6 paires, 0 manquantes) | 0.84× [0.83–0.86] (6 paires, 0 manquantes) |
| llmd-native-s | llmd-ar-s | n/d | n/d | n/d |
| sar-dflash | sar-ar | 1.38× [1.31–1.45] (6 paires, 0 manquantes) | 1.04× [1.00–1.08] (6 paires, 0 manquantes) | 0.81× [0.79–0.83] (6 paires, 0 manquantes) |
| sar-mtp | sar-ar | 1.41× [1.37–1.49] (6 paires, 0 manquantes) | 1.08× [1.01–1.15] (6 paires, 0 manquantes) | 0.81× [0.80–0.83] (6 paires, 0 manquantes) |