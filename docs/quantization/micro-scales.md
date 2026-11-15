# Poids MLX à micro-échelles

Les projections et embeddings denses acceptent les modes MLX `mxfp4`,
`nvfp4` et `mxfp8`. Les codes restent empaquetés en U32 et les échelles en U8,
sans biais affine. Une erreur `UnsupportedQuantization` signale un mode
inconnu ; un groupe, un dtype ou une forme incompatible est refusé avant
le chargement des poids.

| Mode JSON | Code du poids | Échelle | Groupe | Bits |
|---|---|---|---:|---:|
| `mxfp4` | E2M1 | E8M0 | 32 | 4 |
| `nvfp4` | E2M1 | E4M3 | 16 | 4 |
| `mxfp8` | E4M3 | E8M0 | 32 | 8 |

La configuration MLX utilise `quantization.mode`, `group_size` et `bits`.
Les paramètres par chemin de couche prennent priorité sur les paramètres
communs. Le mode `affine`, et les configurations historiques sans `mode`,
gardent leur représentation habituelle.

Les colonnes sont contiguës : huit nibbles FP4 ou quatre octets FP8 par mot,
code de poids faible en premier. Les échelles suivent les groupes de colonnes
de chaque ligne. `MicroQuantizedTensor` fournit la déquantification f32,
la lecture d'une ligne et le produit CPU de référence. Ce stockage ne passe
jamais par les kernels affines.

Le contrat suit les headers de MLX 0.31.2 (`fp4.h`, `fp8.h`,
`fp_quantized.h`) et le convertisseur mlx-lm 0.31.3. En particulier, le
header local décode E4M3 `0x7f` en 480 et E8M0 `0xff` en +inf. Le format
NVFP4 de ce convertisseur n'a pas de facteur global supplémentaire.
Les représentations GGUF et les couches quantifiant aussi leurs activations
ne sont pas des alias de ce format.

Sur Metal, le QMV conserve les codes compacts. À partir de 16 lignes,
avec K divisible par 64, le GEMM NAX charge les poids dans une tuile bf16
64×64 et accumule en f32 ; les activations sont arrondies en bf16.
Les autres tailles utilisent le QMV par ligne. Les tests GPU sont ignorés
par défaut et doivent être exécutés sous le verrou GPU du poste.
Le backend micro actuel exige les primitives NAX du compilateur Metal.
Les piles d'experts de rang 3 sont refusées ; cette implémentation couvre
les projections et embeddings de rang 2.

## Vérification locale

Les tests CPU couvrent les tables de décodage, l'empaquetage, les groupes,
les lignes et le chargement depuis des shards distincts. Les tests Metal
`micro_dequant_and_matmul_bit_exact` et `micro_all_scale_bytes_bit_exact`
comparent exactement les bits sur des vecteurs construits à la main.
Le test ignoré `micro_bandwidth_and_gemm_probe` affiche les octets de poids
et d'échelles lus par seconde, ainsi que les TFLOPS et le comparatif u4.
Les temps comprennent commit/wait, amortis sur plusieurs dispatches.

L'exemple `quant_micro` mesure le corpus KL existant sans modifier ses goldens :

```sh
cargo build --release --example quant_micro
# Exécuter chaque commande suivante sous le verrou GPU du poste.
target/release/examples/quant_micro capture "$BF16_MODEL" "$LOCAL_REFERENCE"
target/release/examples/quant_micro score "$QUANTIZED_MODEL" "$LOCAL_REFERENCE"
target/release/examples/quant_micro bench "$QUANTIZED_MODEL" "$LOCAL_BENCH"
```

La capture est un artefact local distinct : top-64 et masse restante sur
huit morceaux de 512 tokens, avec les 256 dernières positions notées.
Le candidat doit conserver le vocabulaire et les tokens du corpus de la
référence. La sortie rapporte KL moyenne, p99/max Δp et accord top-1 ;
aucun seuil des oracles existants n'est modifié. Le benchmark effectue un
échauffement et trois mesures, pour 1 024 et 4 096 tokens de prompt et
64 tokens générés, en conservant les IDs pour comparer les backends.
