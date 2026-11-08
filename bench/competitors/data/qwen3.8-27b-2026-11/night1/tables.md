
### eval-L / code — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 12.06 | — | — |
| llkm-dflash | n/d | n/d | n/d |
| llkm-mtp | 15.65 | 1.30× (1 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llmd-ar | n/d | n/d | n/d |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 12.47 | 1.03× (1 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| mlx-ar | 14.25 | 1.18× (1 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-ar | 14.73 | 1.22× (1 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-dflash | n/d | n/d | n/d |
| sar-mtp | n/d | n/d | n/d |

### eval-L / code — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 1090.62 | — | — |
| llkm-dflash | n/d | n/d | n/d |
| llkm-mtp | 1145.11 | 1.05× (1 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llmd-ar | n/d | n/d | n/d |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 1065.29 | 0.98× (1 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| mlx-ar | 722.01 | 0.66× (1 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-ar | 732.11 | 0.67× (1 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-dflash | n/d | n/d | n/d |
| sar-mtp | n/d | n/d | n/d |

### eval-L / code — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 1123.71 | — | — |
| llkm-dflash | n/d | n/d | n/d |
| llkm-mtp | 1170.61 | 1.04× (1 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llmd-ar | n/d | n/d | n/d |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 1097.30 | 0.98× (1 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| mlx-ar | 750.01 | 0.67× (1 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-ar | 759.19 | 0.68× (1 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-dflash | n/d | n/d | n/d |
| sar-mtp | n/d | n/d | n/d |

### eval-L / code — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 1123.71 | 400 | 1 |
| llkm-mtp | 1170.61 | 400 | 1 |
| llq40-ar | 1097.30 | 400 | 1 |
| mlx-ar | 750.01 | 400 | 1 |
| sar-ar | 759.19 | 400 | 1 |

### eval-L / code — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | n/d | n/d | n/d |
| llkm-mtp | llkm-ar | 1.30× (1 paires, 0 manquantes) | 1.05× (1 paires, 0 manquantes) | 1.04× (1 paires, 0 manquantes) |
| llmd-native | llmd-ar | n/d | n/d | n/d |
| sar-dflash | sar-ar | n/d | n/d | n/d |
| sar-mtp | sar-ar | n/d | n/d | n/d |

### eval-M / code — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 24.81 | — | — |
| llkm-dflash | 38.45 | 1.63× (1 paires, 2 manquantes) | indicatif (moins de 3 prompts) |
| llkm-mtp | 32.87 | 1.32× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llmd-ar | ≈17.49 | ≈0.70× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 26.45 | 1.07× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llq40-dflash | 44.63 | 1.80× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llq40-mtp | 35.39 | 1.43× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| mlx-ar | 27.02 | 1.09× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-ar | 27.89 | 1.12× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-dflash | 49.00 | 1.98× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-mtp | 38.70 | 1.56× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |

### eval-M / code — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 57.14 | — | — |
| llkm-dflash | 58.49 | 1.07× (1 paires, 2 manquantes) | indicatif (moins de 3 prompts) |
| llkm-mtp | 57.62 | 1.01× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llmd-ar | 139.76 | 2.45× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 52.33 | 0.92× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llq40-dflash | 55.72 | 0.98× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llq40-mtp | 54.72 | 0.96× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| mlx-ar | 42.60 | 0.75× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-ar | 44.85 | 0.78× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-dflash | 46.60 | 0.82× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-mtp | 47.69 | 0.83× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |

### eval-M / code — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 73.22 | — | — |
| llkm-dflash | 68.86 | 0.96× (1 paires, 2 manquantes) | indicatif (moins de 3 prompts) |
| llkm-mtp | 69.76 | 0.95× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llmd-ar | n/d | n/d | n/d |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 67.42 | 0.92× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llq40-dflash | 64.68 | 0.88× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llq40-mtp | 66.00 | 0.90× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| mlx-ar | 57.37 | 0.78× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-ar | 59.15 | 0.81× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-dflash | 54.78 | 0.75× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-mtp | 58.09 | 0.79× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |

### eval-M / code — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 73.22 | 400 | 3 |
| llkm-dflash | 68.86 | 400 | 1 |
| llkm-mtp | 69.76 | 400 | 3 |
| llmd-ar | 162.57 | 400 | 3 |
| llq40-ar | 67.42 | 400 | 3 |
| llq40-dflash | 64.68 | 400 | 3 |
| llq40-mtp | 66.00 | 400 | 3 |
| mlx-ar | 57.37 | 400 | 3 |
| sar-ar | 59.15 | 400 | 3 |
| sar-dflash | 54.78 | 400 | 3 |
| sar-mtp | 58.09 | 400 | 3 |

### eval-M / code — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | 1.63× (1 paires, 2 manquantes) | 1.07× (1 paires, 2 manquantes) | 0.96× (1 paires, 2 manquantes) |
| llkm-mtp | llkm-ar | 1.32× (3 paires, 0 manquantes) | 1.01× (3 paires, 0 manquantes) | 0.95× (3 paires, 0 manquantes) |
| llmd-native | llmd-ar | n/d | n/d | n/d |
| llq40-dflash | llq40-ar | 1.69× (3 paires, 0 manquantes) | 1.06× (3 paires, 0 manquantes) | 0.96× (3 paires, 0 manquantes) |
| llq40-mtp | llq40-ar | 1.34× (3 paires, 0 manquantes) | 1.05× (3 paires, 0 manquantes) | 0.98× (3 paires, 0 manquantes) |
| sar-dflash | sar-ar | 1.76× (3 paires, 0 manquantes) | 1.04× (3 paires, 0 manquantes) | 0.93× (3 paires, 0 manquantes) |
| sar-mtp | sar-ar | 1.39× (3 paires, 0 manquantes) | 1.06× (3 paires, 0 manquantes) | 0.98× (3 paires, 0 manquantes) |

### eval-M / prose — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 23.55 | — | — |
| llkm-dflash | 36.50 | 1.55× (1 paires, 2 manquantes) | indicatif (moins de 3 prompts) |
| llkm-mtp | 33.65 | 1.43× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llmd-ar | 17.51 | 0.74× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 26.28 | 1.12× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llq40-dflash | 36.98 | 1.57× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llq40-mtp | 34.54 | 1.47× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| mlx-ar | 26.99 | 1.15× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-ar | 27.94 | 1.19× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-dflash | 45.10 | 1.92× (2 paires, 1 manquantes) | indicatif (moins de 3 prompts) |
| sar-mtp | 39.06 | 1.66× (2 paires, 1 manquantes) | indicatif (moins de 3 prompts) |

### eval-M / prose — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 55.51 | — | — |
| llkm-dflash | 58.85 | 1.06× (1 paires, 2 manquantes) | indicatif (moins de 3 prompts) |
| llkm-mtp | 58.57 | 1.06× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llmd-ar | 139.82 | 2.52× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 54.66 | 0.98× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llq40-dflash | 53.33 | 0.96× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llq40-mtp | 55.89 | 1.01× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| mlx-ar | 44.90 | 0.81× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-ar | 47.00 | 0.85× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-dflash | 47.81 | 0.86× (2 paires, 1 manquantes) | indicatif (moins de 3 prompts) |
| sar-mtp | 46.01 | 0.83× (2 paires, 1 manquantes) | indicatif (moins de 3 prompts) |

### eval-M / prose — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 72.47 | — | — |
| llkm-dflash | 69.78 | 0.96× (1 paires, 2 manquantes) | indicatif (moins de 3 prompts) |
| llkm-mtp | 70.29 | 0.97× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llmd-ar | 162.63 | 2.24× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 69.87 | 0.96× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llq40-dflash | 64.17 | 0.89× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| llq40-mtp | 66.84 | 0.92× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| mlx-ar | 59.74 | 0.82× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-ar | 61.17 | 0.84× (3 paires, 0 manquantes) | indicatif (moins de 3 prompts) |
| sar-dflash | 56.67 | 0.78× (2 paires, 1 manquantes) | indicatif (moins de 3 prompts) |
| sar-mtp | 56.23 | 0.78× (2 paires, 1 manquantes) | indicatif (moins de 3 prompts) |

### eval-M / prose — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 72.47 | 400 | 3 |
| llkm-dflash | 69.78 | 400 | 1 |
| llkm-mtp | 70.29 | 400 | 3 |
| llmd-ar | 162.63 | 400 | 3 |
| llq40-ar | 69.87 | 400 | 3 |
| llq40-dflash | 64.17 | 400 | 3 |
| llq40-mtp | 66.84 | 400 | 3 |
| mlx-ar | 59.74 | 400 | 3 |
| sar-ar | 61.17 | 400 | 3 |
| sar-dflash | 56.67 | 400 | 2 |
| sar-mtp | 56.23 | 400 | 2 |

### eval-M / prose — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | 1.55× (1 paires, 2 manquantes) | 1.06× (1 paires, 2 manquantes) | 0.96× (1 paires, 2 manquantes) |
| llkm-mtp | llkm-ar | 1.43× (3 paires, 0 manquantes) | 1.06× (3 paires, 0 manquantes) | 0.97× (3 paires, 0 manquantes) |
| llmd-native | llmd-ar | n/d | n/d | n/d |
| llq40-dflash | llq40-ar | 1.41× (3 paires, 0 manquantes) | 0.98× (3 paires, 0 manquantes) | 0.92× (3 paires, 0 manquantes) |
| llq40-mtp | llq40-ar | 1.31× (3 paires, 0 manquantes) | 1.02× (3 paires, 0 manquantes) | 0.96× (3 paires, 0 manquantes) |
| sar-dflash | sar-ar | 1.62× (2 paires, 1 manquantes) | 1.04× (2 paires, 1 manquantes) | 0.94× (2 paires, 1 manquantes) |
| sar-mtp | sar-ar | 1.40× (2 paires, 1 manquantes) | 1.00× (2 paires, 1 manquantes) | 0.93× (2 paires, 1 manquantes) |

### eval-S / chat — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | ≈27.24 | — | — |
| llkm-dflash | ≈43.24 | ≈1.58× [1.53–1.69] (7 paires, 23 manquantes) | meilleur (notable) |
| llkm-mtp | ≈37.81 | ≈1.37× [1.31–1.42] (30 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar | ≈25.64 | ≈0.94× [0.91–0.96] (30 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | ≈30.67 | ≈1.12× [1.09–1.15] (30 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | ≈49.35 | ≈1.82× [1.69–1.96] (29 paires, 1 manquantes) | meilleur (notable) |
| llq40-mtp | ≈40.07 | ≈1.45× [1.40–1.53] (29 paires, 1 manquantes) | meilleur (notable) |
| mlx-ar | ≈30.62 | ≈1.12× [1.10–1.14] (30 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | ≈32.21 | ≈1.18× [1.15–1.21] (30 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | ≈58.67 | ≈1.99× [1.76–2.24] (30 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | ≈43.35 | ≈1.58× [1.49–1.70] (30 paires, 0 manquantes) | meilleur (notable) |

### eval-S / chat — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 0.26 | — | — |
| llkm-dflash | 0.26 | 1.02× [0.99–1.05] (7 paires, 23 manquantes) | indistinct |
| llkm-mtp | 0.26 | 1.04× [1.00–1.10] (30 paires, 0 manquantes) | moins bon (< 5 %) |
| llmd-ar | 0.95 | 3.72× [3.65–4.06] (30 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 0.22 | 0.85× [0.84–0.92] (30 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 0.23 | 0.92× [0.88–0.99] (29 paires, 1 manquantes) | meilleur (notable) |
| llq40-mtp | 0.23 | 0.91× [0.88–0.99] (29 paires, 1 manquantes) | meilleur (notable) |
| mlx-ar | 0.24 | 0.93× [0.91–0.98] (30 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 0.16 | 0.62× [0.60–0.65] (30 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 0.17 | 0.66× [0.64–0.68] (30 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 0.15 | 0.59× [0.58–0.64] (30 paires, 0 manquantes) | meilleur (notable) |

### eval-S / chat — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 14.90 | — | — |
| llkm-dflash | 9.48 | 0.64× [0.60–0.66] (7 paires, 21 manquantes) | meilleur (notable) |
| llkm-mtp | 10.82 | 0.74× [0.71–0.77] (28 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar | 16.51 | 1.11× [1.09–1.15] (21 paires, 9 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 13.23 | 0.89× [0.87–0.92] (28 paires, 2 manquantes) | meilleur (notable) |
| llq40-dflash | 8.31 | 0.55× [0.52–0.60] (28 paires, 1 manquantes) | meilleur (notable) |
| llq40-mtp | 10.20 | 0.69× [0.66–0.72] (28 paires, 1 manquantes) | meilleur (notable) |
| mlx-ar | 13.27 | 0.89× [0.87–0.91] (28 paires, 2 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 12.55 | 0.85× [0.83–0.87] (23 paires, 7 manquantes) | meilleur (notable) |
| sar-dflash | 7.24 | 0.52× [0.47–0.60] (23 paires, 7 manquantes) | meilleur (notable) |
| sar-mtp | 9.47 | 0.65× [0.61–0.68] (23 paires, 7 manquantes) | meilleur (notable) |

### eval-S / chat — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 14.90 | 400 | 30 |
| llkm-dflash | 9.48 | 400 | 7 |
| llkm-mtp | 10.82 | 400 | 30 |
| llmd-ar | 16.51 | 400 | 30 |
| llq40-ar | 13.23 | 400 | 30 |
| llq40-dflash | 8.31 | 400 | 29 |
| llq40-mtp | 10.20 | 400 | 29 |
| mlx-ar | 13.27 | 400 | 30 |
| sar-ar | 12.55 | 400 | 30 |
| sar-dflash | 6.97 | 400 | 30 |
| sar-mtp | 9.35 | 400 | 30 |

### eval-S / chat — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | ≈1.58× [1.53–1.69] (7 paires, 23 manquantes) | 1.02× [0.99–1.05] (7 paires, 23 manquantes) | 0.64× [0.60–0.66] (7 paires, 21 manquantes) |
| llkm-mtp | llkm-ar | ≈1.37× [1.31–1.42] (30 paires, 0 manquantes) | 1.04× [1.01–1.11] (30 paires, 0 manquantes) | 0.74× [0.71–0.77] (28 paires, 0 manquantes) |
| llmd-native | llmd-ar | n/d | n/d | n/d |
| llq40-dflash | llq40-ar | 1.62× [1.51–1.74] (29 paires, 1 manquantes) | 1.08× [1.03–1.10] (29 paires, 1 manquantes) | 0.62× [0.58–0.67] (29 paires, 1 manquantes) |
| llq40-mtp | llq40-ar | 1.30× [1.26–1.35] (29 paires, 1 manquantes) | 1.07× [1.01–1.10] (29 paires, 1 manquantes) | 0.78× [0.74–0.80] (29 paires, 1 manquantes) |
| sar-dflash | sar-ar | ≈1.69× [1.48–1.90] (30 paires, 0 manquantes) | 1.06× [1.04–1.08] (30 paires, 0 manquantes) | 0.62× [0.55–0.71] (25 paires, 0 manquantes) |
| sar-mtp | sar-ar | ≈1.34× [1.26–1.43] (30 paires, 0 manquantes) | 0.96× [0.95–1.00] (30 paires, 0 manquantes) | 0.77× [0.73–0.81] (25 paires, 0 manquantes) |

### eval-S / code — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 26.79 | — | — |
| llkm-dflash | 42.75 | 1.61× (5 paires, 10 manquantes) | indicatif (moins de 3 prompts) |
| llkm-mtp | 37.82 | 1.40× [1.28–1.49] (14 paires, 1 manquantes) | meilleur (notable) |
| llmd-ar | ≈24.17 | ≈0.90× [0.86–0.92] (15 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 30.07 | 1.12× [1.07–1.16] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 51.85 | 1.93× [1.85–1.99] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-mtp | 40.63 | 1.49× [1.43–1.57] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-ar | 29.95 | 1.11× [1.06–1.15] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 30.96 | 1.15× [1.10–1.20] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 55.24 | 2.06× [1.96–2.13] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 46.31 | 1.71× [1.63–1.77] (15 paires, 0 manquantes) | meilleur (notable) |

### eval-S / code — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 5.99 | — | — |
| llkm-dflash | 6.04 | 1.02× (5 paires, 10 manquantes) | indicatif (moins de 3 prompts) |
| llkm-mtp | 6.35 | 1.04× [0.95–1.16] (14 paires, 1 manquantes) | indistinct |
| llmd-ar | 15.69 | 2.53× [2.34–2.70] (15 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 5.41 | 0.88× [0.80–1.00] (15 paires, 0 manquantes) | indistinct |
| llq40-dflash | 5.89 | 0.95× [0.86–1.07] (15 paires, 0 manquantes) | indistinct |
| llq40-mtp | 6.07 | 0.96× [0.90–1.07] (15 paires, 0 manquantes) | indistinct |
| mlx-ar | 4.81 | 0.78× [0.68–0.91] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 5.01 | 0.81× [0.75–0.90] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 5.64 | 0.90× [0.87–0.96] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 5.39 | 0.86× [0.82–0.94] (15 paires, 0 manquantes) | meilleur (notable) |

### eval-S / code — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 20.79 | — | — |
| llkm-dflash | 15.21 | 0.74× (5 paires, 10 manquantes) | indicatif (moins de 3 prompts) |
| llkm-mtp | 17.09 | 0.82× [0.79–0.84] (14 paires, 1 manquantes) | meilleur (notable) |
| llmd-ar | 32.19 | 1.56× (8 paires, 7 manquantes) | indicatif (moins de 3 prompts) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 18.75 | 0.90× [0.87–0.93] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 13.69 | 0.65× [0.62–0.68] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-mtp | 15.90 | 0.77× [0.73–0.79] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-ar | 18.08 | 0.88× [0.84–0.90] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 17.87 | 0.87× [0.83–0.89] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 12.80 | 0.62× [0.60–0.63] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 14.02 | 0.68× [0.65–0.70] (15 paires, 0 manquantes) | meilleur (notable) |

### eval-S / code — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 20.79 | 400 | 15 |
| llkm-dflash | 15.21 | 400 | 5 |
| llkm-mtp | 17.09 | 400 | 14 |
| llmd-ar | 32.20 | 400 | 15 |
| llq40-ar | 18.75 | 400 | 15 |
| llq40-dflash | 13.69 | 400 | 15 |
| llq40-mtp | 15.90 | 400 | 15 |
| mlx-ar | 18.08 | 400 | 15 |
| sar-ar | 17.87 | 400 | 15 |
| sar-dflash | 12.80 | 400 | 15 |
| sar-mtp | 14.02 | 400 | 15 |

### eval-S / code — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | 1.61× (5 paires, 10 manquantes) | 1.02× (5 paires, 10 manquantes) | 0.74× (5 paires, 10 manquantes) |
| llkm-mtp | llkm-ar | 1.40× [1.28–1.49] (14 paires, 1 manquantes) | 1.04× [0.95–1.16] (14 paires, 1 manquantes) | 0.82× [0.79–0.84] (14 paires, 1 manquantes) |
| llmd-native | llmd-ar | n/d | n/d | n/d |
| llq40-dflash | llq40-ar | 1.72× [1.66–1.78] (15 paires, 0 manquantes) | 1.08× [0.94–1.18] (15 paires, 0 manquantes) | 0.72× [0.70–0.75] (15 paires, 0 manquantes) |
| llq40-mtp | llq40-ar | 1.33× [1.30–1.39] (15 paires, 0 manquantes) | 1.10× [0.96–1.21] (15 paires, 0 manquantes) | 0.85× [0.82–0.87] (15 paires, 0 manquantes) |
| sar-dflash | sar-ar | 1.79× [1.74–1.83] (15 paires, 0 manquantes) | 1.11× [1.02–1.19] (15 paires, 0 manquantes) | 0.71× [0.69–0.73] (15 paires, 0 manquantes) |
| sar-mtp | sar-ar | 1.48× [1.42–1.54] (15 paires, 0 manquantes) | 1.06× [0.97–1.14] (15 paires, 0 manquantes) | 0.78× [0.74–0.81] (15 paires, 0 manquantes) |

### eval-S / prose — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 26.99 | — | — |
| llkm-dflash | 39.83 | 1.48× [1.30–1.59] (7 paires, 8 manquantes) | meilleur (notable) |
| llkm-mtp | 37.61 | 1.38× [1.32–1.47] (15 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar | 24.19 | 0.89× [0.86–0.91] (15 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 30.04 | 1.11× [1.07–1.13] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 45.21 | 1.67× [1.55–1.77] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-mtp | 39.09 | 1.44× [1.36–1.50] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-ar | 29.91 | 1.10× [1.07–1.13] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 30.98 | 1.14× [1.10–1.16] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 45.87 | 1.70× [1.60–1.86] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 44.30 | 1.61× [1.55–1.69] (15 paires, 0 manquantes) | meilleur (notable) |

### eval-S / prose — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 6.02 | — | — |
| llkm-dflash | 6.23 | 1.04× [0.93–1.12] (7 paires, 8 manquantes) | indistinct |
| llkm-mtp | 6.65 | 1.09× [1.00–1.14] (15 paires, 0 manquantes) | moins bon (notable) |
| llmd-ar | 15.69 | 2.59× [2.37–2.67] (15 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 5.44 | 0.90× [0.80–0.99] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 5.85 | 0.97× [0.86–1.04] (15 paires, 0 manquantes) | indistinct |
| llq40-mtp | 5.98 | 1.00× [0.89–1.07] (15 paires, 0 manquantes) | indistinct |
| mlx-ar | 4.81 | 0.81× [0.72–0.90] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 5.20 | 0.84× [0.76–0.92] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 5.63 | 0.93× [0.84–0.97] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 5.27 | 0.87× [0.79–0.92] (15 paires, 0 manquantes) | meilleur (notable) |

### eval-S / prose — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 20.88 | — | — |
| llkm-dflash | 16.57 | 0.78× [0.75–0.82] (7 paires, 8 manquantes) | meilleur (notable) |
| llkm-mtp | 17.13 | 0.82× [0.79–0.85] (15 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar | 32.19 | 1.55× [1.53–1.57] (15 paires, 0 manquantes) | moins bon (notable) |
| llmd-native | n/d | n/d | n/d |
| llq40-ar | 18.76 | 0.90× [0.88–0.93] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-dflash | 14.87 | 0.71× [0.67–0.74] (15 paires, 0 manquantes) | meilleur (notable) |
| llq40-mtp | 16.39 | 0.79× [0.75–0.81] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-ar | 18.16 | 0.88× [0.85–0.91] (15 paires, 0 manquantes) | meilleur (notable) |
| mlx-dflash | n/d | n/d | n/d |
| mlx-mtp | n/d | n/d | n/d |
| sar-ar | 18.06 | 0.87× [0.84–0.89] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 14.14 | 0.68× [0.65–0.71] (15 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 14.45 | 0.69× [0.66–0.71] (15 paires, 0 manquantes) | meilleur (notable) |

### eval-S / prose — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 20.88 | 400 | 15 |
| llkm-dflash | 16.57 | 400 | 7 |
| llkm-mtp | 17.13 | 400 | 15 |
| llmd-ar | 32.19 | 400 | 15 |
| llq40-ar | 18.76 | 400 | 15 |
| llq40-dflash | 14.87 | 400 | 15 |
| llq40-mtp | 16.39 | 400 | 15 |
| mlx-ar | 18.16 | 400 | 15 |
| sar-ar | 18.06 | 400 | 15 |
| sar-dflash | 14.14 | 400 | 15 |
| sar-mtp | 14.45 | 400 | 15 |

### eval-S / prose — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | 1.48× [1.30–1.59] (7 paires, 8 manquantes) | 1.04× [0.93–1.12] (7 paires, 8 manquantes) | 0.78× [0.75–0.82] (7 paires, 8 manquantes) |
| llkm-mtp | llkm-ar | 1.38× [1.32–1.48] (15 paires, 0 manquantes) | 1.09× [1.00–1.14] (15 paires, 0 manquantes) | 0.82× [0.79–0.85] (15 paires, 0 manquantes) |
| llmd-native | llmd-ar | n/d | n/d | n/d |
| llq40-dflash | llq40-ar | 1.51× [1.43–1.60] (15 paires, 0 manquantes) | 1.08× [0.99–1.13] (15 paires, 0 manquantes) | 0.79× [0.76–0.81] (15 paires, 0 manquantes) |
| llq40-mtp | llq40-ar | 1.30× [1.24–1.35] (15 paires, 0 manquantes) | 1.11× [0.95–1.23] (15 paires, 0 manquantes) | 0.87× [0.84–0.90] (15 paires, 0 manquantes) |
| sar-dflash | sar-ar | 1.50× [1.39–1.65] (15 paires, 0 manquantes) | 1.10× [0.95–1.16] (15 paires, 0 manquantes) | 0.78× [0.74–0.83] (15 paires, 0 manquantes) |
| sar-mtp | sar-ar | 1.42× [1.37–1.47] (15 paires, 0 manquantes) | 1.03× [0.92–1.14] (15 paires, 0 manquantes) | 0.80× [0.75–0.83] (15 paires, 0 manquantes) |

### sampled-S / chat — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | ≈28.30 | — | — |
| llkm-dflash | ≈42.99 | ≈1.50× [1.37–1.62] (12 paires, 0 manquantes) | meilleur (notable) |
| llkm-mtp | ≈39.04 | ≈1.34× [1.29–1.39] (12 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | ≈25.30 | ≈0.89× [0.89–0.90] (12 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | ≈29.99 | ≈1.05× [1.04–1.07] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | ≈31.32 | ≈1.11× [1.10–1.12] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | ≈50.17 | ≈1.72× [1.51–1.96] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | ≈33.57 | ≈1.18× [1.14–1.23] (12 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / chat — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 0.25 | — | — |
| llkm-dflash | 0.27 | 1.04× [1.02–1.07] (12 paires, 0 manquantes) | moins bon (< 5 %) |
| llkm-mtp | 0.27 | 1.06× [1.02–1.09] (12 paires, 0 manquantes) | moins bon (notable) |
| llmd-ar-s | 0.95 | 3.76× [3.69–3.84] (12 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 0.24 | 0.94× [0.92–0.95] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 0.16 | 0.62× [0.61–0.63] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 0.17 | 0.66× [0.65–0.67] (12 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 0.15 | 0.58× [0.57–0.59] (12 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / chat — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 14.35 | — | — |
| llkm-dflash | 9.74 | 0.70× [0.65–0.76] (8 paires, 2 manquantes) | meilleur (notable) |
| llkm-mtp | 10.79 | 0.76× [0.73–0.79] (10 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | 16.75 | 1.17× [1.17–1.18] (9 paires, 1 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 13.54 | 0.95× [0.94–0.97] (10 paires, 2 manquantes) | meilleur (< 5 %) |
| sar-ar | 12.96 | 0.90× [0.89–0.91] (10 paires, 1 manquantes) | meilleur (notable) |
| sar-dflash | 8.37 | 0.61× [0.55–0.69] (10 paires, 1 manquantes) | meilleur (notable) |
| sar-mtp | 12.08 | 0.85× [0.83–0.88] (10 paires, 1 manquantes) | meilleur (notable) |

### sampled-S / chat — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 14.35 | 400 | 12 |
| llkm-dflash | 9.53 | 400 | 12 |
| llkm-mtp | 10.49 | 400 | 12 |
| llmd-ar-s | 16.72 | 400 | 12 |
| mlx-ar | 13.54 | 400 | 12 |
| sar-ar | 12.90 | 400 | 12 |
| sar-dflash | 8.13 | 400 | 12 |
| sar-mtp | 12.04 | 400 | 12 |

### sampled-S / chat — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | ≈1.50× [1.37–1.62] (12 paires, 0 manquantes) | 1.04× [1.02–1.07] (12 paires, 0 manquantes) | 0.70× [0.65–0.76] (8 paires, 2 manquantes) |
| llkm-mtp | llkm-ar | ≈1.34× [1.29–1.40] (12 paires, 0 manquantes) | 1.06× [1.02–1.09] (12 paires, 0 manquantes) | 0.76× [0.73–0.79] (10 paires, 0 manquantes) |
| llmd-native-s | llmd-ar-s | n/d | n/d | n/d |
| sar-dflash | sar-ar | ≈1.55× [1.35–1.76] (12 paires, 0 manquantes) | 1.07× [1.06–1.07] (12 paires, 0 manquantes) | 0.64× [0.56–0.74] (11 paires, 0 manquantes) |
| sar-mtp | sar-ar | ≈1.07× [1.02–1.11] (12 paires, 0 manquantes) | 0.93× [0.92–0.94] (12 paires, 0 manquantes) | 0.95× [0.91–0.99] (10 paires, 2 manquantes) |

### sampled-S / code — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 27.87 | — | — |
| llkm-dflash | 40.99 | 1.48× [1.42–1.53] (6 paires, 0 manquantes) | meilleur (notable) |
| llkm-mtp | 35.86 | 1.29× [1.28–1.31] (6 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | ≈23.97 | ≈0.86× [0.85–0.86] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 29.36 | 1.05× [1.03–1.06] (6 paires, 0 manquantes) | meilleur (< 5 %) |
| sar-ar | 30.46 | 1.10× [1.09–1.10] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 47.29 | 1.73× [1.64–1.87] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 32.46 | 1.16× [1.14–1.17] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / code — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 5.91 | — | — |
| llkm-dflash | 6.56 | 1.10× [1.02–1.16] (6 paires, 0 manquantes) | moins bon (notable) |
| llkm-mtp | 6.16 | 1.05× [1.03–1.08] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-ar-s | 15.70 | 2.66× [2.65–2.67] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 4.53 | 0.77× [0.75–0.80] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 4.95 | 0.84× [0.82–0.85] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 5.27 | 0.92× [0.88–0.98] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 5.08 | 0.86× [0.82–0.89] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / code — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 20.24 | — | — |
| llkm-dflash | 16.10 | 0.80× [0.78–0.83] (6 paires, 0 manquantes) | meilleur (notable) |
| llkm-mtp | 17.30 | 0.86× [0.84–0.87] (6 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | 32.31 | 1.60× (3 paires, 3 manquantes) | indicatif (moins de 3 prompts) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 18.20 | 0.90× [0.89–0.91] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 18.01 | 0.89× [0.88–0.90] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 13.76 | 0.68× [0.66–0.70] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 17.43 | 0.86× [0.85–0.87] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / code — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 20.24 | 400 | 6 |
| llkm-dflash | 16.10 | 400 | 6 |
| llkm-mtp | 17.30 | 400 | 6 |
| llmd-ar-s | 32.44 | 400 | 6 |
| mlx-ar | 18.20 | 400 | 6 |
| sar-ar | 18.01 | 400 | 6 |
| sar-dflash | 13.76 | 400 | 6 |
| sar-mtp | 17.43 | 400 | 6 |

### sampled-S / code — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | 1.48× [1.42–1.54] (6 paires, 0 manquantes) | 1.10× [1.02–1.16] (6 paires, 0 manquantes) | 0.80× [0.78–0.83] (6 paires, 0 manquantes) |
| llkm-mtp | llkm-ar | 1.29× [1.28–1.31] (6 paires, 0 manquantes) | 1.05× [1.03–1.09] (6 paires, 0 manquantes) | 0.86× [0.84–0.87] (6 paires, 0 manquantes) |
| llmd-native-s | llmd-ar-s | n/d | n/d | n/d |
| sar-dflash | sar-ar | 1.58× [1.50–1.71] (6 paires, 0 manquantes) | 1.10× [1.04–1.20] (6 paires, 0 manquantes) | 0.76× [0.74–0.78] (6 paires, 0 manquantes) |
| sar-mtp | sar-ar | 1.06× [1.05–1.07] (6 paires, 0 manquantes) | 1.02× [0.97–1.09] (6 paires, 0 manquantes) | 0.97× [0.96–0.98] (6 paires, 0 manquantes) |

### sampled-S / prose — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 27.86 | — | — |
| llkm-dflash | 37.91 | 1.41× [1.35–1.53] (6 paires, 0 manquantes) | meilleur (notable) |
| llkm-mtp | 36.65 | 1.31× [1.28–1.33] (6 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | 24.03 | 0.86× [0.86–0.87] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 29.11 | 1.05× [1.04–1.06] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 30.61 | 1.10× [1.09–1.10] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 39.64 | 1.44× [1.37–1.54] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 31.05 | 1.12× [1.11–1.14] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / prose — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 5.93 | — | — |
| llkm-dflash | 6.09 | 1.04× [1.02–1.09] (6 paires, 0 manquantes) | moins bon (< 5 %) |
| llkm-mtp | 6.20 | 1.06× [1.03–1.12] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-ar-s | 15.74 | 2.65× [2.62–2.67] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 4.49 | 0.76× [0.74–0.77] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 5.27 | 0.87× [0.82–0.90] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 5.21 | 0.89× [0.86–0.94] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 5.32 | 0.90× [0.86–0.94] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / prose — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| llkm-ar | 20.25 | — | — |
| llkm-dflash | 16.59 | 0.81× [0.78–0.82] (6 paires, 0 manquantes) | meilleur (notable) |
| llkm-mtp | 17.14 | 0.85× [0.84–0.86] (6 paires, 0 manquantes) | meilleur (notable) |
| llmd-ar-s | 32.38 | 1.60× [1.59–1.61] (6 paires, 0 manquantes) | moins bon (notable) |
| llmd-native-s | n/d | n/d | n/d |
| mlx-ar | 18.20 | 0.89× [0.88–0.90] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-ar | 18.23 | 0.90× [0.89–0.90] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-dflash | 15.23 | 0.75× [0.72–0.79] (6 paires, 0 manquantes) | meilleur (notable) |
| sar-mtp | 18.17 | 0.89× [0.88–0.91] (6 paires, 0 manquantes) | meilleur (notable) |

### sampled-S / prose — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| llkm-ar | 20.25 | 400 | 6 |
| llkm-dflash | 16.59 | 400 | 6 |
| llkm-mtp | 17.14 | 400 | 6 |
| llmd-ar-s | 32.38 | 400 | 6 |
| mlx-ar | 18.20 | 400 | 6 |
| sar-ar | 18.23 | 400 | 6 |
| sar-dflash | 15.23 | 400 | 6 |
| sar-mtp | 18.17 | 400 | 6 |

### sampled-S / prose — T2, gain spéculatif contre l'AR du même moteur

| config | contre | débit de livraison (tok/s) | TTFT (s) | latence jusqu'à L tokens communs (s) |
|---|---|---|---|---|
| llkm-dflash | llkm-ar | 1.41× [1.35–1.54] (6 paires, 0 manquantes) | 1.04× [1.02–1.10] (6 paires, 0 manquantes) | 0.81× [0.78–0.82] (6 paires, 0 manquantes) |
| llkm-mtp | llkm-ar | 1.31× [1.28–1.33] (6 paires, 0 manquantes) | 1.06× [1.03–1.12] (6 paires, 0 manquantes) | 0.85× [0.84–0.86] (6 paires, 0 manquantes) |
| llmd-native-s | llmd-ar-s | n/d | n/d | n/d |
| sar-dflash | sar-ar | 1.31× [1.24–1.41] (6 paires, 0 manquantes) | 1.03× [0.97–1.08] (6 paires, 0 manquantes) | 0.84× [0.81–0.87] (6 paires, 0 manquantes) |
| sar-mtp | sar-ar | 1.02× [1.01–1.04] (6 paires, 0 manquantes) | 1.04× [0.96–1.10] (6 paires, 0 manquantes) | 1.00× [0.97–1.02] (6 paires, 0 manquantes) |

### tune / chat — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| tune-ref:llkm-ar | 26.71 | — | — |
| tune-ref:sar-ar | 32.20 | — | — |
| tune:llama-dflash:0 | n/d | n/d | n/d |
| tune:llama-dflash:1 | n/d | n/d | n/d |
| tune:llama-dflash:2 | n/d | n/d | n/d |
| tune:llama-mtp:0 | n/d | n/d | n/d |
| tune:llama-mtp:1 | 36.98 | — | — |
| tune:llama-mtp:2 | n/d | n/d | n/d |
| tune:llama-sampler:0 | 28.55 | — | — |
| tune:llama-sampler:1 | 27.08 | — | — |
| tune:sar-dflash:0 | 66.37 | — | — |
| tune:sar-dflash:1 | 54.28 | — | — |
| tune:sar-dflash:2 | n/d | n/d | n/d |
| tune:sar-mtp:0 | 45.81 | — | — |
| tune:sar-mtp:1 | 40.41 | — | — |
| tune:sar-mtp:2 | 42.92 | — | — |

### tune / chat — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| tune-ref:llkm-ar | 0.26 | — | — |
| tune-ref:sar-ar | 0.16 | — | — |
| tune:llama-dflash:0 | n/d | n/d | n/d |
| tune:llama-dflash:1 | n/d | n/d | n/d |
| tune:llama-dflash:2 | n/d | n/d | n/d |
| tune:llama-mtp:0 | n/d | n/d | n/d |
| tune:llama-mtp:1 | 0.28 | — | — |
| tune:llama-mtp:2 | n/d | n/d | n/d |
| tune:llama-sampler:0 | 0.25 | — | — |
| tune:llama-sampler:1 | 0.26 | — | — |
| tune:sar-dflash:0 | 0.17 | — | — |
| tune:sar-dflash:1 | 0.17 | — | — |
| tune:sar-dflash:2 | n/d | n/d | n/d |
| tune:sar-mtp:0 | 0.16 | — | — |
| tune:sar-mtp:1 | 0.15 | — | — |
| tune:sar-mtp:2 | 0.16 | — | — |

### tune / chat — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| tune-ref:llkm-ar | 15.20 | — | — |
| tune-ref:sar-ar | 12.55 | — | — |
| tune:llama-dflash:0 | n/d | n/d | n/d |
| tune:llama-dflash:1 | n/d | n/d | n/d |
| tune:llama-dflash:2 | n/d | n/d | n/d |
| tune:llama-mtp:0 | n/d | n/d | n/d |
| tune:llama-mtp:1 | 11.08 | — | — |
| tune:llama-mtp:2 | n/d | n/d | n/d |
| tune:llama-sampler:0 | 14.22 | — | — |
| tune:llama-sampler:1 | 15.00 | — | — |
| tune:sar-dflash:0 | 6.18 | — | — |
| tune:sar-dflash:1 | 7.52 | — | — |
| tune:sar-dflash:2 | n/d | n/d | n/d |
| tune:sar-mtp:0 | 8.87 | — | — |
| tune:sar-mtp:1 | 10.02 | — | — |
| tune:sar-mtp:2 | 9.45 | — | — |

### tune / chat — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| tune-ref:llkm-ar | 15.20 | 400 | 2 |
| tune-ref:sar-ar | 12.55 | 400 | 2 |
| tune:llama-mtp:1 | 11.08 | 400 | 2 |
| tune:llama-sampler:0 | 14.22 | 400 | 2 |
| tune:llama-sampler:1 | 15.00 | 400 | 2 |
| tune:sar-dflash:0 | 6.18 | 400 | 2 |
| tune:sar-dflash:1 | 7.52 | 400 | 2 |
| tune:sar-mtp:0 | 8.87 | 400 | 2 |
| tune:sar-mtp:1 | 10.02 | 400 | 2 |
| tune:sar-mtp:2 | 9.45 | 400 | 2 |

### tune / code — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| tune-ref:llkm-ar | 26.83 | — | — |
| tune-ref:sar-ar | 31.00 | — | — |
| tune:llama-ctxcp:0 | 25.77 | — | — |
| tune:llama-ctxcp:1 | 26.03 | — | — |
| tune:llama-dflash:0 | 44.48 | — | — |
| tune:llama-dflash:1 | 53.30 | — | — |
| tune:llama-dflash:2 | 49.66 | — | — |
| tune:llama-mtp:0 | 47.97 | — | — |
| tune:llama-mtp:1 | 38.01 | — | — |
| tune:llama-mtp:2 | 41.47 | — | — |
| tune:llama-sampler:0 | 27.50 | — | — |
| tune:llama-sampler:1 | 27.15 | — | — |
| tune:llama-ubatch:0 | 25.93 | — | — |
| tune:llama-ubatch:1 | 25.53 | — | — |
| tune:llama-ubatch:2 | 25.86 | — | — |
| tune:llmd-batching:0 | ≈20.86 | — | — |
| tune:llmd-batching:1 | ≈20.67 | — | — |
| tune:mlx-prefill:0 | 28.52 | — | — |
| tune:mlx-prefill:1 | 28.49 | — | — |
| tune:sar-dflash:0 | 54.64 | — | — |
| tune:sar-dflash:1 | 49.89 | — | — |
| tune:sar-dflash:2 | n/d | n/d | n/d |
| tune:sar-mtp:0 | 45.18 | — | — |
| tune:sar-mtp:1 | 45.89 | — | — |
| tune:sar-mtp:2 | 44.84 | — | — |

### tune / code — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| tune-ref:llkm-ar | 5.94 | — | — |
| tune-ref:sar-ar | 5.14 | — | — |
| tune:llama-ctxcp:0 | 30.63 | — | — |
| tune:llama-ctxcp:1 | 30.89 | — | — |
| tune:llama-dflash:0 | 6.20 | — | — |
| tune:llama-dflash:1 | 6.42 | — | — |
| tune:llama-dflash:2 | 6.53 | — | — |
| tune:llama-mtp:0 | 6.59 | — | — |
| tune:llama-mtp:1 | 6.32 | — | — |
| tune:llama-mtp:2 | 6.92 | — | — |
| tune:llama-sampler:0 | 5.95 | — | — |
| tune:llama-sampler:1 | 5.92 | — | — |
| tune:llama-ubatch:0 | 30.91 | — | — |
| tune:llama-ubatch:1 | 30.92 | — | — |
| tune:llama-ubatch:2 | 31.95 | — | — |
| tune:llmd-batching:0 | 84.23 | — | — |
| tune:llmd-batching:1 | 77.77 | — | — |
| tune:mlx-prefill:0 | 23.61 | — | — |
| tune:mlx-prefill:1 | 24.25 | — | — |
| tune:sar-dflash:0 | 5.60 | — | — |
| tune:sar-dflash:1 | 5.41 | — | — |
| tune:sar-dflash:2 | n/d | n/d | n/d |
| tune:sar-mtp:0 | 5.57 | — | — |
| tune:sar-mtp:1 | 5.35 | — | — |
| tune:sar-mtp:2 | 5.39 | — | — |

### tune / code — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| tune-ref:llkm-ar | 20.81 | — | — |
| tune-ref:sar-ar | 18.01 | — | — |
| tune:llama-ctxcp:0 | 46.59 | — | — |
| tune:llama-ctxcp:1 | 46.35 | — | — |
| tune:llama-dflash:0 | 15.17 | — | — |
| tune:llama-dflash:1 | 13.91 | — | — |
| tune:llama-dflash:2 | 14.56 | — | — |
| tune:llama-mtp:0 | 14.91 | — | — |
| tune:llama-mtp:1 | 16.82 | — | — |
| tune:llama-mtp:2 | 16.54 | — | — |
| tune:llama-sampler:0 | 20.45 | — | — |
| tune:llama-sampler:1 | 20.61 | — | — |
| tune:llama-ubatch:0 | 46.43 | — | — |
| tune:llama-ubatch:1 | 46.73 | — | — |
| tune:llama-ubatch:2 | 47.38 | — | — |
| tune:llmd-batching:0 | n/d | n/d | n/d |
| tune:llmd-batching:1 | n/d | n/d | n/d |
| tune:mlx-prefill:0 | 37.64 | — | — |
| tune:mlx-prefill:1 | 38.29 | — | — |
| tune:sar-dflash:0 | 12.90 | — | — |
| tune:sar-dflash:1 | 13.41 | — | — |
| tune:sar-dflash:2 | n/d | n/d | n/d |
| tune:sar-mtp:0 | 14.40 | — | — |
| tune:sar-mtp:1 | 14.05 | — | — |
| tune:sar-mtp:2 | 14.29 | — | — |

### tune / code — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| tune-ref:llkm-ar | 20.81 | 400 | 2 |
| tune-ref:sar-ar | 18.01 | 400 | 2 |
| tune:llama-ctxcp:0 | 46.59 | 400 | 4 |
| tune:llama-ctxcp:1 | 46.35 | 400 | 4 |
| tune:llama-dflash:0 | 15.17 | 400 | 2 |
| tune:llama-dflash:1 | 13.91 | 400 | 2 |
| tune:llama-dflash:2 | 14.56 | 400 | 2 |
| tune:llama-mtp:0 | 14.91 | 400 | 2 |
| tune:llama-mtp:1 | 16.82 | 400 | 2 |
| tune:llama-mtp:2 | 16.54 | 400 | 2 |
| tune:llama-sampler:0 | 20.45 | 400 | 2 |
| tune:llama-sampler:1 | 20.61 | 400 | 2 |
| tune:llama-ubatch:0 | 46.43 | 400 | 4 |
| tune:llama-ubatch:1 | 46.73 | 400 | 4 |
| tune:llama-ubatch:2 | 47.38 | 400 | 4 |
| tune:llmd-batching:0 | 103.85 | 400 | 4 |
| tune:llmd-batching:1 | 97.57 | 400 | 4 |
| tune:mlx-prefill:0 | 37.64 | 400 | 4 |
| tune:mlx-prefill:1 | 38.29 | 400 | 4 |
| tune:sar-dflash:0 | 12.90 | 400 | 2 |
| tune:sar-dflash:1 | 13.41 | 400 | 2 |
| tune:sar-mtp:0 | 14.40 | 400 | 2 |
| tune:sar-mtp:1 | 14.05 | 400 | 2 |
| tune:sar-mtp:2 | 14.29 | 400 | 2 |

### tune / prose — débit de livraison (tok/s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| tune-ref:llkm-ar | 28.34 | — | — |
| tune-ref:sar-ar | 30.78 | — | — |
| tune:llama-dflash:0 | 40.94 | — | — |
| tune:llama-dflash:1 | 43.19 | — | — |
| tune:llama-dflash:2 | 38.82 | — | — |
| tune:llama-mtp:0 | 45.37 | — | — |
| tune:llama-mtp:1 | 38.80 | — | — |
| tune:llama-mtp:2 | 44.28 | — | — |
| tune:llama-sampler:0 | 27.17 | — | — |
| tune:llama-sampler:1 | 28.70 | — | — |
| tune:sar-dflash:0 | 49.94 | — | — |
| tune:sar-dflash:1 | 42.72 | — | — |
| tune:sar-dflash:2 | n/d | n/d | n/d |
| tune:sar-mtp:0 | 44.15 | — | — |
| tune:sar-mtp:1 | 40.92 | — | — |
| tune:sar-mtp:2 | 41.82 | — | — |

### tune / prose — TTFT (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| tune-ref:llkm-ar | 6.48 | — | — |
| tune-ref:sar-ar | 5.28 | — | — |
| tune:llama-dflash:0 | 6.33 | — | — |
| tune:llama-dflash:1 | 6.71 | — | — |
| tune:llama-dflash:2 | 6.27 | — | — |
| tune:llama-mtp:0 | 6.96 | — | — |
| tune:llama-mtp:1 | 6.72 | — | — |
| tune:llama-mtp:2 | 6.52 | — | — |
| tune:llama-sampler:0 | 6.27 | — | — |
| tune:llama-sampler:1 | 6.47 | — | — |
| tune:sar-dflash:0 | 5.71 | — | — |
| tune:sar-dflash:1 | 5.19 | — | — |
| tune:sar-dflash:2 | n/d | n/d | n/d |
| tune:sar-mtp:0 | 5.40 | — | — |
| tune:sar-mtp:1 | 5.14 | — | — |
| tune:sar-mtp:2 | 5.59 | — | — |

### tune / prose — latence jusqu'à L tokens communs (s)

| config | médiane | rapport à llkm-ar [IC 95 %] | verdict |
|---|---|---|---|
| tune-ref:llkm-ar | 20.56 | — | — |
| tune-ref:sar-ar | 18.25 | — | — |
| tune:llama-dflash:0 | 16.08 | — | — |
| tune:llama-dflash:1 | 15.94 | — | — |
| tune:llama-dflash:2 | 16.55 | — | — |
| tune:llama-mtp:0 | 15.75 | — | — |
| tune:llama-mtp:1 | 17.01 | — | — |
| tune:llama-mtp:2 | 15.53 | — | — |
| tune:llama-sampler:0 | 20.96 | — | — |
| tune:llama-sampler:1 | 20.38 | — | — |
| tune:sar-dflash:0 | 13.70 | — | — |
| tune:sar-dflash:1 | 14.53 | — | — |
| tune:sar-dflash:2 | n/d | n/d | n/d |
| tune:sar-mtp:0 | 14.44 | — | — |
| tune:sar-mtp:1 | 14.89 | — | — |
| tune:sar-mtp:2 | 15.13 | — | — |

### tune / prose — latence totale, longueur naturelle (s) — descriptive

| config | médiane | N médian | requêtes |
|---|---|---|---|
| tune-ref:llkm-ar | 20.56 | 400 | 2 |
| tune-ref:sar-ar | 18.25 | 400 | 2 |
| tune:llama-dflash:0 | 16.08 | 400 | 1 |
| tune:llama-dflash:1 | 15.94 | 400 | 1 |
| tune:llama-dflash:2 | 16.55 | 400 | 1 |
| tune:llama-mtp:0 | 15.75 | 400 | 1 |
| tune:llama-mtp:1 | 17.01 | 400 | 2 |
| tune:llama-mtp:2 | 15.53 | 400 | 1 |
| tune:llama-sampler:0 | 20.96 | 400 | 2 |
| tune:llama-sampler:1 | 20.38 | 400 | 2 |
| tune:sar-dflash:0 | 13.70 | 400 | 2 |
| tune:sar-dflash:1 | 14.53 | 400 | 2 |
| tune:sar-mtp:0 | 14.44 | 400 | 2 |
| tune:sar-mtp:1 | 14.89 | 400 | 2 |
| tune:sar-mtp:2 | 15.13 | 400 | 2 |