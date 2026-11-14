# Prefill MoE BM32 et gated delta hv32

Le GEMM groupé u4 gs64 BM32 est accessible aux routeurs softmax, y compris
aux experts routés du chemin partagé. Le softmax garde BM16 lorsque
`rows × top_k < 16 × experts` : le padding BM32 double sur les petits lots
uniformes et les sondes mesurent une régression. Le routeur sigmoïde garde
sa sélection existante. Les overrides de test forcent les deux variantes
pour continuer à les comparer sous le seuil. Le grouping, le padding et les deux
GEMM utilisent la même hauteur de tuile. Les formes non alignées, les poids
u8 et les pipelines manquants gardent le chemin 16 × 32. Le kill-switch
historique `SARAGOSSA_RUST_FLINT_MOE_COOP_TILED=0` couvre aussi le softmax.

Le gated delta NAX sélectionne les instanciations hk16/hv32 et hk16/hv48,
dk=dv=128, et dimensionne la grille sur Hv. Les autres formes et les pipelines
absents suivent le repli existant. L'état bf16 reste hors du chemin NAX.
hv32 utilise trois produits compensés (hi×hi + hi×lo + lo×hi) dans le
layout MPP existant : la précision relâchée du
port hv48 dépasse la borne de 5e-4 sur la sonde à T=512. hv48 conserve
son mode historique. Le kernel conserve l'écrêtage `log(max(g, 1e-6))` de la variante hv48.
`SARAGOSSA_RUST_PREFILL_LINEAR_NAX=0` désactive les deux instanciations.

## Oracles

Compiler hors verrou, puis lancer les tests GPU en série sous le verrou
commun. Les tests ignorés de modèle exigent un chemin local explicite ;
ils ne téléchargent aucun poids.

```sh
rtk cargo test --release --all-features --lib --no-run
rtk proxy python3 tools/gpulock.py run --owner prefill-oracles --prio 10 -- \
  cargo test --release --all-features --lib -- --test-threads=1 --nocapture \
  softmax_coop_tiled_padding_boundary \
  gated_delta_nax_matches_naive_oracle \
  gated_delta_nax_unsupported_shape_or_pipeline_falls_back \
  moe_routed_coop_tiled_matches_16x32 \
  moe_shared_coop_tiled_matches_16x32 \
  moe_routed_rows_coop_full_matches_cpu \
  moe_routed_rows_coop_matches_gather_30b_multitile
```

NAX est comparé au CPU et au scan séquentiel pour y et l'état final,
avec état initial non nul, à T=1,15,16,17,55,300, et aussi T=512 pour hv32.
La tolérance absolue
préexistante de 5e-4 est inchangée. Les oracles MoE exigent l'identité des
bits, avec routages déséquilibrés, experts vides, queues et résidu.

Le modèle est comparé sous les quatre combinaisons BM16/BM32 × repli/NAX.
Le cache de préfixe est coupé ; les compteurs doivent prouver les chemins
utilisés. Les tokens greedy doivent être strictement identiques ; les hidden
BM32/BM16 également, à récurrence identique.

```sh
# MODEL désigne le dossier local Qwen3.6-35B-A3B-4bit.
rtk proxy python3 tools/gpulock.py run --owner prefill-greedy --prio 10 -- \
  env SARAGOSSA_RUST_PREFIX_CACHE=0 SARAGOSSA_RUST_MTP_DECODE=0 \
  SARAGOSSA_TEST_PREFILL_MODEL="$MODEL" \
  SARAGOSSA_TEST_PREFILL_PROMPT=tests/fixtures/rowsexact/code4k-1.txt \
  SARAGOSSA_TEST_PREFILL_LENGTHS=512,4096 SARAGOSSA_TEST_PREFILL_STEPS=32 \
  cargo test --release --all-features --lib -- --ignored --exact \
  qwen_loader::prefill_variants_tests::qwen_moe_prefill_variants_greedy_identity \
  --test-threads=1 --nocapture
```

`SARAGOSSA_TEST_PREFILL_PROMPT` accepte un fichier texte externe. Ses tokens
sont répétés ou tronqués aux longueurs demandées. Sans fichier, le test utilise
une phrase synthétique sur Rust. Pour étendre la validation, changer le corpus,
les longueurs et le nombre de tokens ; la réussite ne vaut que pour les cas
exécutés.

## Sondes et A/B

Les sondes reprises de kbench sont ignorées par défaut. `KB_REPS` est borné
entre 1 et 31 (défaut 5). `kb_moe_coop` compare le softmax à 256 experts top-8
et 512 experts top-10 sur 64,128,512,4096 tokens, et rapporte les lignes
réellement padées à partir des indices GPU. Il exige l'identité des bits.
`kb_gated_delta` compare seq f32, seq bf16 et NAX à Hv=32,48 et
T=512,4096,8192 ; y et l'état hv32 NAX/seq f32 doivent rester dans 5e-4.
Le contrôle hv48 long rapporte l'écart sans ajouter de borne au kernel
historique ; son oracle CPU court garde sa tolérance de 5e-4.

```sh
rtk proxy python3 tools/gpulock.py run --owner prefill-sondes --prio 10 -- \
  env KB_REPS=5 cargo test --release --all-features --lib -- \
  --ignored --test-threads=1 --nocapture \
  metal_backend::tests::kb_moe_coop metal_backend::tests::kb_gated_delta
```

L'A/B intégré ci-dessous isole C1, B1 et leur combinaison. Il utilise le
même binaire, le greedy AR et l'identité des ids. Les quatre blocs ABBA par
comparaison sont à lancer séparément des validations courtes.

```sh
rtk cargo build --release --all-features
for lot in c1 b1 ensemble; do
  case "$lot" in
    c1) bm=1; nax=0 ;;
    b1) bm=0; nax=1 ;;
    ensemble) bm=1; nax=1 ;;
  esac
  rtk proxy python3 tools/gpulock.py run --owner "prefill-$lot" --prio 10 -- \
    python3 bench/ab.py run ".tmp/ab-prefill-$lot" \
    --a target/release/saragossa --b target/release/saragossa \
    --model-dir "$MODEL" --prompt-tokens 4096 --max-tokens 128 \
    --temperature 0 --blocks 4 --order abba --primary prefill_ms --identity ids \
    --env-a SARAGOSSA_RUST_PREFIX_CACHE=0 --env-b SARAGOSSA_RUST_PREFIX_CACHE=0 \
    --env-a SARAGOSSA_RUST_MTP_DECODE=0 --env-b SARAGOSSA_RUST_MTP_DECODE=0 \
    --env-a SARAGOSSA_RUST_FLINT_MOE_COOP_TILED=0 \
    --env-a SARAGOSSA_RUST_PREFILL_LINEAR_NAX=0 \
    --env-b "SARAGOSSA_RUST_FLINT_MOE_COOP_TILED=$bm" \
    --env-b "SARAGOSSA_RUST_PREFILL_LINEAR_NAX=$nax"
  rtk proxy python3 bench/ab.py report ".tmp/ab-prefill-$lot"
done
```

## Validation courte du 14 novembre 2026

M5 Max, trois répétitions, routage synthétique uniforme, verrou partagé.
Ces durées murales sont une sonde de padding, pas une campagne de débit :
les temps sont nettement plus élevés que les mesures kbench initiales.
Le seuil reste conservateur ; la campagne ABBA ci-dessus doit confirmer
le gain intégré sur un corpus représentatif.

| Experts/top-k | Tokens | Slots utiles | Lignes BM16/BM32 | Médiane BM16/BM32 (ms) |
|---|---:|---:|---:|---:|
| 512/10 | 64 | 640 | 6912 / 13824 | 23,94 / 31,97 |
| 512/10 | 128 | 1280 | 8192 / 16384 | 26,96 / 44,85 |
| 512/10 | 512 | 5120 | 8192 / 16384 | 48,71 / 57,95 |

Les sorties des huit cas MoE (256/8 et 512/10, quatre longueurs) sont
identiques bit à bit. À exactement 16 slots/expert, les mesures dépendent
du routage : le cas uniforme de cette sonde régresse, tandis que kbench
mesurait un gain sur 256/8 à 512 tokens. BM32 reste sélectionné à cette
frontière pour préserver ce cas déjà mesuré.

La version hv32 compensée passe l'oracle CPU (y et état, écart maximal
2,11e-6) et le scan séquentiel jusqu'à 8192 tokens (écart maximal 1,04e-5),
sans relâcher 5e-4. La sonde longue hv48 historique rapporte environ
1,12e-3 ; son oracle court reste à 3,75e-4 et son calcul n'a pas changé.

Sur Qwen3.6-35B-A3B-4bit, `code4k-1` répété/tronqué à 512 et 4096 tokens :
les 32 tokens greedy sont identiques dans les quatre combinaisons.
Le contrôle vérifie séparément les chemins du hidden et de la génération :
40 appels BM32 et 30 appels NAX hv32 par prefill lorsque ces variantes
sont forcées. Les hidden BM16/BM32 sont bit-identiques. Les huit tests
finaux passent en 18,15 s sous verrou.

Les mesures NAX compensées ont une seule répétition : 4,66 ms à T=512,
6,05 ms à T=4096 et 9,44 ms à T=8192 contre 3,62 / 11,32 / 22,59 ms pour
seq f32. Elles ne constituent pas un verdict de performance, notamment à
T=512. L'A/B intégré et les répétitions supplémentaires restent à lancer.
