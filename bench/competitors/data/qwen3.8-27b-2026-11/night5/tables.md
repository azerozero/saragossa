
### eval-S / chat — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | ≈27.33 | — | — |
| llkm-dflash | ≈43.18 | ≈1.58× [1.50–1.67] (7 paires, 23 manquantes) | meilleur (notable) |
| llkm-mtp | ≈37.91 | ≈1.37× [1.32–1.40] (30 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar | ≈25.54 | ≈0.93× [0.91–0.94] (30 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | ≈30.60 | ≈1.11× [1.09–1.12] (30 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | ≈49.33 | ≈1.81× [1.68–1.93] (29 paires, 1 manquantes) | meilleur (notable) |
| llq40-mtp | ≈40.33 | ≈1.46× [1.40–1.51] (29 paires, 1 manquantes) | meilleur (notable) |
| mlx-ar | ≈30.53 | ≈1.11× [1.09–1.12] (30 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | ≈32.16 | ≈1.17× [1.15–1.18] (30 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | ≈57.79 | ≈1.99× [1.75–2.22] (30 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | ≈45.70 | ≈1.65× [1.56–1.74] (30 paires, 0 manquantes) | meilleur (notable) |

### eval-S / chat — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 0.25 | — | — |
| llkm-dflash | 0.27 | 1.06× [1.00–1.08] (7 paires, 23 manquantes) | moins bon (notable) |
| llkm-mtp | 0.27 | 1.06× [1.02–1.13] (30 paires, 0 manquantes) | moins bon (notable) |
| llmd-ar | 0.95 | 3.75× [3.62–3.98] (30 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 0.22 | 0.86× [0.82–0.90] (30 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 0.23 | 0.92× [0.86–0.97] (29 paires, 1 manquantes) | meilleur (notable) |
| llq40-mtp | 0.23 | 0.92× [0.86–0.98] (29 paires, 1 manquantes) | meilleur (notable) |
| mlx-ar | 0.24 | 0.94× [0.90–0.98] (30 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 0.16 | 0.62× [0.59–0.65] (30 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 0.17 | 0.66× [0.64–0.69] (30 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 0.15 | 0.60× [0.57–0.64] (30 paires, 0 manquantes) | meilleur (notable) |

### eval-S / chat — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 14.86 | — | — |
| llkm-dflash | 9.49 | 0.64× [0.61–0.67] (7 paires, 21 manquantes) | meilleur (notable) |
| llkm-mtp | 10.85 | 0.74× [0.72–0.77] (28 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar | 16.58 | 1.12× [1.11–1.16] (21 paires, 9 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 13.26 | 0.90× [0.89–0.92] (28 paires, 2 manquantes) | meilleur (notable) |
| llq40-dflash | 8.31 | 0.56× [0.53–0.60] (28 paires, 1 manquantes) | meilleur (notable) |
| llq40-mtp | 10.14 | 0.69× [0.67–0.72] (28 paires, 1 manquantes) | meilleur (notable) |
| mlx-ar | 13.31 | 0.90× [0.89–0.92] (28 paires, 2 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 12.57 | 0.85× [0.84–0.87] (23 paires, 7 manquantes) | meilleur (notable) |
| sar-dflash | 7.24 | 0.52× [0.47–0.60] (23 paires, 7 manquantes) | meilleur (notable) |
| sar-mtp | 8.95 | 0.61× [0.59–0.65] (23 paires, 7 manquantes) | meilleur (notable) |

### eval-S / chat — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 14.86 | 400 | 30 |
| llkm-dflash | 9.49 | 400 | 7 |
| llkm-mtp | 10.80 | 400 | 30 |
| llmd-ar | 16.58 | 400 | 30 |
| llq40-ar | 13.26 | 400 | 30 |
| llq40-dflash | 8.31 | 400 | 29 |
| llq40-mtp | 10.14 | 400 | 29 |
| mlx-ar | 13.31 | 400 | 30 |
| sar-ar | 12.56 | 400 | 30 |
| sar-dflash | 7.07 | 400 | 30 |
| sar-mtp | 8.89 | 400 | 30 |

### eval-S / chat — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | ≈1.58× [1.49–1.67] (7 paires, 23 manquantes) | 1.06× [1.00–1.08] (7 paires, 23 manquantes) | 0.64× [0.61–0.67] (7 paires, 21 manquantes) |
| llkm-mtp | llkm-ar | ≈1.37× [1.32–1.41] (30 paires, 0 manquantes) | 1.06× [1.01–1.12] (30 paires, 0 manquantes) | 0.74× [0.72–0.77] (28 paires, 0 manquantes) |
| llmd-native | llmd-ar | n/d | n/d | n/d |
| llq40-dflash | llq40-ar | 1.62× [1.51–1.74] (29 paires, 1 manquantes) | 1.07× [1.04–1.09] (29 paires, 1 manquantes) | 0.62× [0.58–0.67] (29 paires, 1 manquantes) |
| llq40-mtp | llq40-ar | 1.31× [1.27–1.36] (29 paires, 1 manquantes) | 1.08× [1.04–1.11] (29 paires, 1 manquantes) | 0.77× [0.74–0.79] (29 paires, 1 manquantes) |
| sar-dflash | sar-ar | ≈1.70× [1.49–1.90] (30 paires, 0 manquantes) | 1.07× [1.04–1.09] (30 paires, 0 manquantes) | 0.62× [0.55–0.70] (25 paires, 0 manquantes) |
| sar-mtp | sar-ar | ≈1.41× [1.34–1.49] (30 paires, 0 manquantes) | 0.97× [0.95–1.00] (30 paires, 0 manquantes) | 0.73× [0.69–0.76] (25 paires, 0 manquantes) |

### eval-S / code — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 27.07 | — | — |
| llkm-dflash | 42.88 | 1.61× (5 paires, 10 manquantes) | indicatif (moins de 3 prompts) |
| llkm-mtp | 37.81 | 1.39× [1.33–1.45] (14 paires, 1 manquantes) | meilleur (notable) |
| llmd-ar | ≈24.08 | ≈0.88× [0.85–0.90] (15 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 29.98 | 1.09× [1.06–1.12] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 51.37 | 1.88× [1.81–1.95] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-mtp | 40.39 | 1.46× [1.40–1.54] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-ar | 29.87 | 1.09× [1.05–1.12] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 30.93 | 1.13× [1.09–1.16] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 55.16 | 1.99× [1.88–2.09] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 47.71 | 1.73× [1.66–1.80] (15 paires, 0 manquantes) | meilleur (notable) |

### eval-S / code — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 6.16 | — | — |
| llkm-dflash | 6.14 | 1.01× (5 paires, 10 manquantes) | indicatif (moins de 3 prompts) |
| llkm-mtp | 6.37 | 1.03× [0.94–1.13] (14 paires, 1 manquantes) | indistinct |
| llmd-ar | 15.70 | 2.51× [2.35–2.64] (15 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 5.40 | 0.87× [0.79–0.99] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 5.78 | 0.95× [0.87–1.06] (15 paires, 0 manquantes) | indistinct |
| llq40-mtp | 6.00 | 0.97× [0.87–1.06] (15 paires, 0 manquantes) | indistinct |
| mlx-ar | 4.93 | 0.78× [0.68–0.89] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 4.96 | 0.80× [0.75–0.89] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 5.62 | 0.89× [0.85–0.95] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 5.41 | 0.86× [0.81–0.91] (15 paires, 0 manquantes) | meilleur (notable) |

### eval-S / code — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 20.83 | — | — |
| llkm-dflash | 15.24 | 0.73× (5 paires, 10 manquantes) | indicatif (moins de 3 prompts) |
| llkm-mtp | 16.96 | 0.82× [0.80–0.84] (14 paires, 1 manquantes) | meilleur (notable) |
| llmd-ar | 32.27 | 1.55× (8 paires, 7 manquantes) | indicatif (moins de 3 prompts) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 18.69 | 0.90× [0.89–0.93] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 13.66 | 0.66× [0.64–0.68] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-mtp | 15.90 | 0.76× [0.74–0.79] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-ar | 18.24 | 0.88× [0.85–0.90] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 17.85 | 0.86× [0.85–0.89] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 12.80 | 0.62× [0.60–0.64] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 13.72 | 0.66× [0.65–0.68] (15 paires, 0 manquantes) | meilleur (notable) |

### eval-S / code — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 20.83 | 400 | 15 |
| llkm-dflash | 15.24 | 400 | 5 |
| llkm-mtp | 16.96 | 400 | 14 |
| llmd-ar | 32.27 | 400 | 15 |
| llq40-ar | 18.69 | 400 | 15 |
| llq40-dflash | 13.66 | 400 | 15 |
| llq40-mtp | 15.90 | 400 | 15 |
| mlx-ar | 18.24 | 400 | 15 |
| sar-ar | 17.85 | 400 | 15 |
| sar-dflash | 12.80 | 400 | 15 |
| sar-mtp | 13.72 | 400 | 15 |

### eval-S / code — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | 1.61× (5 paires, 10 manquantes) | 1.01× (5 paires, 10 manquantes) | 0.73× (5 paires, 10 manquantes) |
| llkm-mtp | llkm-ar | 1.39× [1.33–1.46] (14 paires, 1 manquantes) | 1.03× [0.94–1.13] (14 paires, 1 manquantes) | 0.82× [0.80–0.84] (14 paires, 1 manquantes) |
| llmd-native | llmd-ar | n/d | n/d | n/d |
| llq40-dflash | llq40-ar | 1.72× [1.66–1.79] (15 paires, 0 manquantes) | 1.09× [0.94–1.18] (15 paires, 0 manquantes) | 0.73× [0.70–0.76] (15 paires, 0 manquantes) |
| llq40-mtp | llq40-ar | 1.34× [1.31–1.39] (15 paires, 0 manquantes) | 1.11× [0.97–1.22] (15 paires, 0 manquantes) | 0.85× [0.82–0.86] (15 paires, 0 manquantes) |
| sar-dflash | sar-ar | 1.77× [1.71–1.83] (15 paires, 0 manquantes) | 1.11× [1.03–1.17] (15 paires, 0 manquantes) | 0.72× [0.69–0.74] (15 paires, 0 manquantes) |
| sar-mtp | sar-ar | 1.54× [1.50–1.59] (15 paires, 0 manquantes) | 1.08× [0.98–1.14] (15 paires, 0 manquantes) | 0.77× [0.73–0.79] (15 paires, 0 manquantes) |

### eval-S / prose — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 27.15 | — | — |
| llkm-dflash | 39.61 | 1.48× [1.35–1.59] (7 paires, 8 manquantes) | meilleur (notable) |
| llkm-mtp | 37.33 | 1.37× [1.31–1.44] (15 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar | 24.09 | 0.88× [0.86–0.90] (15 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 29.98 | 1.10× [1.07–1.12] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 43.99 | 1.66× [1.55–1.75] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-mtp | 38.99 | 1.42× [1.36–1.50] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-ar | 29.87 | 1.09× [1.07–1.12] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 30.86 | 1.13× [1.11–1.16] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 46.24 | 1.69× [1.53–1.87] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 46.22 | 1.68× [1.62–1.74] (15 paires, 0 manquantes) | meilleur (notable) |

### eval-S / prose — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 6.09 | — | — |
| llkm-dflash | 6.29 | 1.05× [0.95–1.15] (7 paires, 8 manquantes) | indistinct |
| llkm-mtp | 6.66 | 1.10× [1.01–1.16] (15 paires, 0 manquantes) | moins bon (notable) |
| llmd-ar | 15.69 | 2.60× [2.39–2.68] (15 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 5.49 | 0.91× [0.85–0.99] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 5.89 | 0.98× [0.89–1.03] (15 paires, 0 manquantes) | indistinct |
| llq40-mtp | 5.99 | 1.00× [0.89–1.07] (15 paires, 0 manquantes) | indistinct |
| mlx-ar | 5.02 | 0.83× [0.73–0.92] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 5.28 | 0.85× [0.78–0.92] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 5.64 | 0.93× [0.83–0.96] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 5.31 | 0.88× [0.80–0.92] (15 paires, 0 manquantes) | meilleur (notable) |

### eval-S / prose — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 20.74 | — | — |
| llkm-dflash | 16.55 | 0.79× [0.76–0.82] (7 paires, 8 manquantes) | meilleur (notable) |
| llkm-mtp | 17.20 | 0.83× [0.80–0.86] (15 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar | 32.26 | 1.55× [1.54–1.57] (15 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 18.82 | 0.91× [0.89–0.94] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 14.95 | 0.71× [0.68–0.74] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-mtp | 16.40 | 0.79× [0.75–0.81] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-ar | 18.38 | 0.89× [0.86–0.91] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 18.20 | 0.87× [0.85–0.89] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 13.98 | 0.68× [0.65–0.74] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 14.13 | 0.68× [0.66–0.70] (15 paires, 0 manquantes) | meilleur (notable) |

### eval-S / prose — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 20.74 | 400 | 15 |
| llkm-dflash | 16.55 | 400 | 7 |
| llkm-mtp | 17.20 | 400 | 15 |
| llmd-ar | 32.26 | 400 | 15 |
| llq40-ar | 18.82 | 400 | 15 |
| llq40-dflash | 14.95 | 400 | 15 |
| llq40-mtp | 16.40 | 400 | 15 |
| mlx-ar | 18.38 | 400 | 15 |
| sar-ar | 18.20 | 400 | 15 |
| sar-dflash | 13.98 | 400 | 15 |
| sar-mtp | 14.13 | 400 | 15 |

### eval-S / prose — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | 1.48× [1.35–1.59] (7 paires, 8 manquantes) | 1.05× [0.95–1.15] (7 paires, 8 manquantes) | 0.79× [0.76–0.82] (7 paires, 8 manquantes) |
| llkm-mtp | llkm-ar | 1.37× [1.31–1.44] (15 paires, 0 manquantes) | 1.10× [1.01–1.16] (15 paires, 0 manquantes) | 0.83× [0.79–0.86] (15 paires, 0 manquantes) |
| llmd-native | llmd-ar | n/d | n/d | n/d |
| llq40-dflash | llq40-ar | 1.51× [1.43–1.61] (15 paires, 0 manquantes) | 1.08× [0.99–1.14] (15 paires, 0 manquantes) | 0.79× [0.75–0.81] (15 paires, 0 manquantes) |
| llq40-mtp | llq40-ar | 1.30× [1.25–1.35] (15 paires, 0 manquantes) | 1.10× [0.96–1.22] (15 paires, 0 manquantes) | 0.87× [0.83–0.90] (15 paires, 0 manquantes) |
| sar-dflash | sar-ar | 1.50× [1.36–1.64] (15 paires, 0 manquantes) | 1.09× [0.96–1.17] (15 paires, 0 manquantes) | 0.79× [0.73–0.84] (15 paires, 0 manquantes) |
| sar-mtp | sar-ar | 1.49× [1.45–1.53] (15 paires, 0 manquantes) | 1.03× [0.94–1.13] (15 paires, 0 manquantes) | 0.78× [0.74–0.81] (15 paires, 0 manquantes) |

### sampled-S / chat — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | ≈27.33 | — | — |
| llkm-dflash | ≈42.14 | ≈1.52× [1.39–1.65] (12 paires, 0 manquantes) | meilleur (notable) |
| llkm-mtp | ≈38.13 | ≈1.37× [1.32–1.42] (12 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | ≈25.52 | ≈0.93× [0.92–0.94] (12 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | ≈30.37 | ≈1.11× [1.10–1.12] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | ≈31.61 | ≈1.15× [1.13–1.16] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | ≈49.48 | ≈1.75× [1.53–2.00] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | ≈42.85 | ≈1.56× [1.49–1.63] (12 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / chat — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 0.26 | — | — |
| llkm-dflash | 0.27 | 1.04× [1.02–1.06] (12 paires, 0 manquantes) | moins bon (< 5 %) |
| llkm-mtp | 0.26 | 1.04× [0.99–1.09] (12 paires, 0 manquantes) | indistinct |
| llmd-ar-s | 0.95 | 3.73× [3.67–3.82] (12 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 0.24 | 0.93× [0.92–0.94] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 0.16 | 0.62× [0.61–0.63] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 0.17 | 0.66× [0.65–0.67] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 0.16 | 0.62× [0.61–0.63] (12 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / chat — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 14.84 | — | — |
| llkm-dflash | 9.86 | 0.69× [0.65–0.74] (8 paires, 2 manquantes) | meilleur (notable) |
| llkm-mtp | 10.90 | 0.74× [0.71–0.77] (10 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | 16.58 | 1.12× [1.12–1.13] (9 paires, 1 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 13.38 | 0.91× [0.90–0.91] (10 paires, 2 manquantes) | meilleur (notable) |
| sar-ar | 12.78 | 0.87× [0.86–0.88] (10 paires, 1 manquantes) | meilleur (notable) |
| sar-dflash | 8.39 | 0.60× [0.55–0.68] (10 paires, 1 manquantes) | meilleur (notable) |
| sar-mtp | 9.48 | 0.65× [0.63–0.68] (10 paires, 1 manquantes) | meilleur (notable) |

### sampled-S / chat — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 14.86 | 400 | 12 |
| llkm-dflash | 9.71 | 400 | 12 |
| llkm-mtp | 10.72 | 400 | 12 |
| llmd-ar-s | 16.58 | 400 | 12 |
| mlx-ar | 13.38 | 400 | 12 |
| sar-ar | 12.78 | 400 | 12 |
| sar-dflash | 8.24 | 400 | 12 |
| sar-mtp | 9.47 | 400 | 12 |

### sampled-S / chat — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | ≈1.52× [1.39–1.65] (12 paires, 0 manquantes) | 1.04× [1.02–1.06] (12 paires, 0 manquantes) | 0.69× [0.65–0.74] (8 paires, 2 manquantes) |
| llkm-mtp | llkm-ar | ≈1.37× [1.32–1.42] (12 paires, 0 manquantes) | 1.04× [0.99–1.09] (12 paires, 0 manquantes) | 0.74× [0.71–0.77] (10 paires, 0 manquantes) |
| llmd-native-s | llmd-ar-s | n/d | n/d | n/d |
| sar-dflash | sar-ar | ≈1.52× [1.33–1.73] (12 paires, 0 manquantes) | 1.06× [1.06–1.07] (12 paires, 0 manquantes) | 0.65× [0.56–0.75] (11 paires, 0 manquantes) |
| sar-mtp | sar-ar | ≈1.36× [1.30–1.41] (12 paires, 0 manquantes) | 1.01× [0.99–1.02] (12 paires, 0 manquantes) | 0.75× [0.72–0.79] (10 paires, 2 manquantes) |

### sampled-S / code — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 26.99 | — | — |
| llkm-dflash | 40.02 | 1.50× [1.43–1.58] (6 paires, 0 manquantes) | meilleur (notable) |
| llkm-mtp | 35.92 | 1.36× [1.31–1.44] (6 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | ≈24.08 | ≈0.89× [0.89–0.89] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 29.64 | 1.10× [1.09–1.10] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | ≈30.89 | ≈1.14× [1.14–1.15] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | ≈47.59 | ≈1.79× [1.73–1.93] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 42.58 | 1.57× [1.55–1.58] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / code — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 5.92 | — | — |
| llkm-dflash | 6.81 | 1.11× [1.01–1.17] (6 paires, 0 manquantes) | moins bon (notable) |
| llkm-mtp | 6.12 | 1.05× [1.02–1.10] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-ar-s | 15.69 | 2.65× [2.63–2.66] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 5.18 | 0.85× [0.76–0.92] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 4.76 | 0.81× [0.80–0.83] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 5.25 | 0.91× [0.88–0.97] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 5.32 | 0.90× [0.86–0.95] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / code — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 20.70 | — | — |
| llkm-dflash | 16.36 | 0.80× [0.76–0.83] (6 paires, 0 manquantes) | meilleur (notable) |
| llkm-mtp | 17.22 | 0.83× [0.81–0.84] (6 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | 32.26 | 1.56× (3 paires, 3 manquantes) | indicatif (moins de 3 prompts) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 18.64 | 0.89× [0.87–0.92] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 17.69 | 0.86× [0.85–0.86] (5 paires, 1 manquantes) | meilleur (notable) |
| sar-dflash | 13.60 | 0.66× [0.65–0.67] (5 paires, 1 manquantes) | meilleur (notable) |
| sar-mtp | 14.81 | 0.71× [0.70–0.73] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / code — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 20.70 | 400 | 6 |
| llkm-dflash | 16.36 | 400 | 6 |
| llkm-mtp | 17.22 | 400 | 6 |
| llmd-ar-s | 32.26 | 400 | 6 |
| mlx-ar | 18.64 | 400 | 6 |
| sar-ar | 17.66 | 400 | 6 |
| sar-dflash | 13.70 | 400 | 6 |
| sar-mtp | 14.81 | 400 | 6 |

### sampled-S / code — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | 1.50× [1.43–1.58] (6 paires, 0 manquantes) | 1.11× [1.01–1.17] (6 paires, 0 manquantes) | 0.80× [0.76–0.83] (6 paires, 0 manquantes) |
| llkm-mtp | llkm-ar | 1.36× [1.31–1.44] (6 paires, 0 manquantes) | 1.05× [1.02–1.11] (6 paires, 0 manquantes) | 0.83× [0.81–0.84] (6 paires, 0 manquantes) |
| llmd-native-s | llmd-ar-s | n/d | n/d | n/d |
| sar-dflash | sar-ar | ≈1.57× [1.52–1.66] (6 paires, 0 manquantes) | 1.13× [1.08–1.18] (6 paires, 0 manquantes) | 0.77× [0.75–0.78] (5 paires, 0 manquantes) |
| sar-mtp | sar-ar | ≈1.37× [1.36–1.38] (6 paires, 0 manquantes) | 1.12× [1.07–1.17] (6 paires, 0 manquantes) | 0.84× [0.82–0.85] (5 paires, 1 manquantes) |

### sampled-S / prose — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 27.01 | — | — |
| llkm-dflash | 37.34 | 1.42× [1.36–1.52] (6 paires, 0 manquantes) | meilleur (notable) |
| llkm-mtp | 36.22 | 1.33× [1.29–1.37] (6 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | 24.07 | 0.89× [0.89–0.89] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 29.63 | 1.10× [1.09–1.10] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 30.79 | 1.13× [1.12–1.14] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 40.07 | 1.52× [1.41–1.66] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 42.44 | 1.55× [1.49–1.59] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / prose — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 5.92 | — | — |
| llkm-dflash | 6.00 | 1.01× [1.01–1.02] (6 paires, 0 manquantes) | moins bon (< 5 %) |
| llkm-mtp | 6.11 | 1.06× [1.02–1.12] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-ar-s | 15.70 | 2.65× [2.65–2.66] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 4.50 | 0.76× [0.75–0.76] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 5.19 | 0.88× [0.82–0.94] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 5.22 | 0.90× [0.85–0.97] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 5.52 | 0.93× [0.90–0.96] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / prose — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 20.70 | — | — |
| llkm-dflash | 16.68 | 0.79× [0.76–0.81] (6 paires, 0 manquantes) | meilleur (notable) |
| llkm-mtp | 17.21 | 0.84× [0.83–0.85] (6 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | 32.27 | 1.56× [1.56–1.56] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 17.96 | 0.87× [0.87–0.87] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 18.15 | 0.88× [0.86–0.90] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 15.09 | 0.73× [0.68–0.78] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 15.06 | 0.73× [0.72–0.74] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / prose — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 20.70 | 400 | 6 |
| llkm-dflash | 16.68 | 400 | 6 |
| llkm-mtp | 17.21 | 400 | 6 |
| llmd-ar-s | 32.27 | 400 | 6 |
| mlx-ar | 17.96 | 400 | 6 |
| sar-ar | 18.15 | 400 | 6 |
| sar-dflash | 15.09 | 400 | 6 |
| sar-mtp | 15.06 | 400 | 6 |

### sampled-S / prose — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | 1.42× [1.36–1.52] (6 paires, 0 manquantes) | 1.01× [1.01–1.02] (6 paires, 0 manquantes) | 0.79× [0.76–0.81] (6 paires, 0 manquantes) |
| llkm-mtp | llkm-ar | 1.33× [1.29–1.37] (6 paires, 0 manquantes) | 1.06× [1.02–1.13] (6 paires, 0 manquantes) | 0.84× [0.83–0.85] (6 paires, 0 manquantes) |
| llmd-native-s | llmd-ar-s | n/d | n/d | n/d |
| sar-dflash | sar-ar | 1.34× [1.26–1.45] (6 paires, 0 manquantes) | 1.03× [0.97–1.08] (6 paires, 0 manquantes) | 0.83× [0.79–0.86] (6 paires, 0 manquantes) |
| sar-mtp | sar-ar | 1.37× [1.33–1.40] (6 paires, 0 manquantes) | 1.06× [0.96–1.14] (6 paires, 0 manquantes) | 0.83× [0.81–0.84] (6 paires, 0 manquantes) |