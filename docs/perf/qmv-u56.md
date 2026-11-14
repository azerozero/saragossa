# QMV u5/u6 et prefill NAX

Les kernels alignés u5/u6 calculent quatre sorties par SIMD, huit par
threadgroup. Chaque lane lit 32 codes quand K est multiple de 1024, sinon
16 codes quand K est multiple de 512. Les mots emballés sont chargés par
vecteurs `packed_uint4` ou `packed_ushort4` ; l'offset packed reste aligné
sur 4 octets. Les activations servent aux quatre sorties et scale/bias sont
lus une fois par groupe de 64, puis partagés par shuffle.

Le chemin aligné exige N multiple de 8 et K multiple de 512. Les autres
formes gs64 conservent le kernel scalaire borné. L'affine est appliquée après
les produits, comme dans l'u4 aligné : cette association change les arrondis
f32. Les comparaisons au générique utilisent l'epsilon absolu `1e-3` des
oracles u4 align64. Les bornes CPU existantes restent inchangées. Les offsets
et l'identité mono/lot restent testés bit à bit. En vérification greedy
stricte, l'opt-in u6 batché est écarté : son ordre scalaire ne reproduit pas
les bits du nouveau qmv aligné.

`SARAGOSSA_RUST_QMV_U5=0` désactive les kernels QMV u5 aligné et tail au
profit du générique, comme `SARAGOSSA_RUST_QMV_U6=0` pour u6. Les deux
flags sont ON par défaut et lus au premier usage. Le prefill NA conserve
son interrupteur indépendant `SARAGOSSA_RUST_QMM_NA`.

## Prefill

Le sélecteur owned utilise `QmmNaDequant` pour u5/u6 gs64 à partir de
`SARAGOSSA_RUST_QMM_NA_MIN_BATCH` (32 par défaut), avec K multiple de 512,
si la pipeline NAX est disponible et `SARAGOSSA_RUST_QMM_NA` actif.
La vérification stricte exclut ce chemin. L'opt-in
`SARAGOSSA_RUST_QMM_U6=1` conserve la priorité du qmm u6 batché.

Comme pour l'ancien chemin u8, le poids est déquantifié en bf16 transposé,
les activations sont arrondies en bf16 et le GEMM accumule en f32.
`private_bf16_buffer` utilise le cache scratch, indexé par
`(label, len, element, namespace)` : le poids transposé occupe `2*N*K`
octets par taille distincte (`N*K`) et namespace, et les activations `2*M*K`
octets par taille distincte (`M*K`). Les appels de même clé réutilisent
l'allocation et réécrivent son contenu ; les tailles différentes s'accumulent.
Les références `owned_buffers` ne sont donc pas seules à retenir ces buffers :
la fin du command buffer ne libère pas le cache dense scratch.

Les entrées restent dans le cache jusqu'à `release_scratch_namespace` : à
la sortie du scope, elles deviennent purgeables, puis le cache est réduit
par éviction LRU à `SARAGOSSA_RUST_SCRATCH_CAP_GB` (4 Gio par défaut).
Les entrées non évincées peuvent rester en cache après cette sortie. Ce
mécanisme commun s'applique déjà à u5/u6 comme à u8 ; le plafond n'est pas
une borne du pic pendant l'encodage. Sans sortie de scope, le cache reste
retenu jusqu'à la destruction de l'exécuteur.
La tête `lm_head` 248320×5120 demande ainsi 2 542 796 800 octets, soit
environ 2,54 Go pour le seul poids transposé, si elle emprunte ce chemin
owned avec M ≥ 32 au seuil par défaut. Les encodeurs résidents sans scratch
et l'API Tensor autonome gardent leurs routes historiques, comme pour u8.

## Oracles TTS u6 à poids réels

`src/tts/tests/golden.rs` contient cinq tests ignorés contre les fixtures
MLX figées : logits et audio VoiceDesign, entrées ICL, premier code cb0 et
audio du clone Base. Ils chargent les snapshots HF
`mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit` et
`mlx-community/Qwen3-TTS-12Hz-1.7B-Base-6bit`. Leur existence conserve
l'activation par défaut du prefill NA u6, y compris pour TTS. Elle ne
constitue pas une validation de ces goldens sur cette révision.

Depuis la racine du dépôt, lancer les cinq oracles (aucun benchmark) :

```sh
rtk proxy env SARAGOSSA_REQUIRE_REAL_MODELS=1 SARAGOSSA_REQUIRE_METAL=1 \
  SARAGOSSA_RUST_QMM_NA=1 SARAGOSSA_RUST_QMM_NA_MIN_BATCH=32 \
  SARAGOSSA_RUST_QMM_U6=0 \
  cargo test --lib --features metal tts::tests::golden::golden_ \
  -- --ignored --nocapture --test-threads=1
```

Pour comparer au repli sans NA, relancer la même commande avec
`SARAGOSSA_RUST_QMM_NA=0`. La résolution utilise le cache HF configuré ;
`SARAGOSSA_QWEN3_TTS_VOICEDESIGN_DIR` et `SARAGOSSA_QWEN3_TTS_BASE_DIR`
permettent de désigner explicitement un snapshot. Le mode
`SARAGOSSA_REQUIRE_REAL_MODELS=1` fait échouer les préconditions manquantes
au lieu de sauter silencieusement les tests. Aucun golden talker
CustomVoice dédié n'est présent. Ces tests ne vérifient pas le dispatch
NAX : une réussite sans pipeline NAX disponible ne qualifie que le repli.
Ajouter `SARAGOSSA_RUST_TRACE_DISPATCH_PATH=1` pour vérifier dans la sortie
les dispatches `dequant_u6_to_bf16_t_gs64` et `gemm_nax` du prefill.

## Sonde courte du 14 novembre 2026

Apple M5 Max, `qmv_aligned_bandwidth_probe`, M=1, poids synthétiques sans modèle.
Deux warmups puis sept mesures A/B scalaire/aligné, ordre alterné, 32 dispatches
par command buffer. Temps `commit`/`wait` inclus ; débit calculé sur les octets
packed uniquement, hors scales/biases. Durée totale : 7,92 s. Ce contrôle
local n'est pas une qualification de débit soutenu ni un benchmark d'inférence.

| Bits | N × K | Scalaire, Go/s | Aligné, Go/s | Médiane alignée, ms | p95 aligné, ms |
|---|---|---:|---:|---:|---:|
| 5 | 5120 × 17408 | 109 | 426 | 0,131 | 0,136 |
| 5 | 17408 × 5120 | 141 | 433 | 0,129 | 0,133 |
| 5 | 248320 × 5120 | 153 | 458 | 1,735 | 1,802 |
| 6 | 5120 × 17408 | 123 | 425 | 0,157 | 0,163 |
| 6 | 17408 × 5120 | 157 | 442 | 0,151 | 0,158 |
| 6 | 248320 × 5120 | 155 | 418 | 2,283 | 2,335 |

Le seuil de 400 Go/s est atteint sur ces six médianes locales. Rejouer sur
GPU au repos pour confirmer la marge et comparer au profil initial avec le
même protocole :

```sh
cargo test --lib qmv_aligned_bandwidth_probe -- --ignored --nocapture --test-threads=1
cargo test --lib affine_qmv_u -- --include-ignored --test-threads=1
cargo test --lib u56 -- --ignored --test-threads=1
cargo test --lib affine_offsets_fast_kernels_bit_exact -- --ignored --test-threads=1
cargo test --lib affine_qmm_u6_matches_qmv_bitwise_and_routes -- --ignored --test-threads=1
```

Pour le prefill, mesurer M=32 et M=524 aux trois formes : temps total de la
déquantification, conversion des activations et GEMM, pic mémoire, puis latence
du prefill réel. Comparer le u6 par défaut à `SARAGOSSA_RUST_QMM_U6=1` ; aucun
vainqueur entre ces deux chemins n'est établi par la sonde M=1. Les oracles
prefill couvrent des N impairs, M=32/33/524, l'arrondi bf16 et les offsets,
sans charger les grandes matrices d'un modèle.
