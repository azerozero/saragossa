# Qualification du prefill Gemma 4

## Décision du 2026-11-14

Le prefill résident Gemma 4 est **activé par défaut**, sur décision du
mainteneur. `SARAGOSSA_RUST_PREFILL_RESIDENT_GEMMA4=0` rétablit le per-op.
L'attention du prefill per-op est calculée en f32 sur CPU, celle du résident
sur GPU : les deux chemins ont des arrondis valides mais différents.
Les **deux chemins** sont qualifiés sur le corpus KL externe court, avec
les seuils et la provenance inchangés. Un oracle long distinct compare le
résident au per-op ; il ne constitue pas une référence MLX longue.

Le résident est éligible jusqu'à **3000 tokens au total**, enveloppe chat
comprise. `SARAGOSSA_RUST_PREFILL_RESIDENT_GEMMA4_MAX_TOKENS` remplace cette
borne par une valeur positive explicite ; zéro, valeur invalide ou absente
conservent 3000. Au-delà, le résident renvoie `Ok(None)` avant l'embedding
et les allocations : l'appelant utilise son repli per-op habituel, sans
changement de `main`. `SARAGOSSA_RUST_TRACE_PREFILL=1` affiche ce refus.
Le d512 ne supporte pas le prolongement résident par morceaux :
`SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK` ne borne donc pas sa mémoire.

**Ce plafond ne constitue pas une qualification uniforme des prompts courts.**
Les points 256, 512 et 3000 passent, mais 64 et 128 échouent sur le top-1,
et 1024 et 2048 échouent sur les trois gardes dans l'oracle interne. Ces échecs restent documentés et reproductibles ; le
kill-switch Gemma 4 continue de permettre le per-op à toute longueur.
Aucun seuil ni golden n'est modifié.

Mesures de qualification ; les A/B longs et le decode portent sur
200 tokens greedy :

| Contrôle | Résultat (2026-11-14) |
|---|---|
| Prefill q4a8-ls, corpus 1k (1037 tokens chat) | Première divergence au token 161 (index 160) |
| Prefill q4a8-ls, corpus 4k (4109 tokens chat) | Première divergence au token 62 (index 61), sans repli |
| Decode seul, même cache et état de prefill | Identique sur 200 tokens |
| Prompt français court 4bit, 28 tokens chat | Identique sur 32 tokens générés (budget historique) |
| Prompt court q4a8-ls, 45 tokens chat (vérifié le 2026-11-14) | Première divergence au token 4 (index 3) sur 200 tokens |
| Oracle chat KL, snapshot 4bit, per-op | Ratio KL 0,11 ; ratio p99 Δp 0,36 ; top-1 99,61 % |
| Oracle chat KL, snapshot 4bit, résident | Ratio KL 0,09 ; ratio p99 Δp 0,34 ; top-1 99,61 % (bf16 : 98,83 %) |
| Oracle long per-op/résident, snapshot 4bit, 4109 tokens, 128 positions | **ÉCHEC** : KL 0,0035143076 (ratio 2,535884) ; p99 Δp 0,1014514138 (ratio 1,451535) ; top-1 98,4375 % |

Le passage initial de l'oracle long sous verrou a confirmé cinq dispatches
`causal_attention_prefill_long_d512_f32` à 4109 tokens. Le KL dépasse la
limite 0,0020787474 (ratio maximal 1,5) ; le p99 reste sous 0,1048387415.
Le top-1 est également sous le garde bf16 de 98,828125 %. **La qualification
longue n'est donc pas acquise**. Le test reste ignoré en CI ordinaire et
échoue lorsqu'on le lance explicitement sur ce checkpoint. Aucun seuil
n'a été ajusté. L'ancien plafond 4096 limitait les allocations ; il ne prouvait
pas la qualité numérique de cette plage. Les mesures supplémentaires et la
réduction du plafond sont détaillées ci-dessous.


Les tests `gemma4_prefill_reference_kld_per_op` et
`gemma4_prefill_reference_kld_resident` partagent le calcul et les seuils de
`golden::kld::external_kld_gemma4` : KL moyen et p99 Δp au plus 1,5 fois le
bruit bf16, top-1 au moins égal à celui de la référence bf16. Leurs morceaux
de 216–224 tokens, dont 128 positions notées en decode forcé, donnent des
**prefills de 88–96 tokens seulement**. Ils couvrent le kernel court, ni mid
(>256), ni long d512 (>2048). Ils exigent les poids locaux, vérifient leur
SHA et refusent un chemin résident indisponible. Ils ne remplacent donc pas
la couverture longue des anciennes assertions d'identité 1k/4k.

`gemma4_prefill_long_per_op_kld_resident` utilise un prompt de 4096 tokens
de corpus avant enveloppe chat (~4100 au total). Il enregistre en mémoire
128 distributions du chemin per-op (attention CPU f32), puis rejoue les
mêmes tokens après le prefill résident, sans repli. Le decode per-op est
commun aux deux parcours afin de comparer l'effet du prefill et de son KV.
C'est l'équivalent interne d'un record/compare, avec les métriques du golden :
partition top-64 + reste, KL moyen et p99 des écarts maximaux de probabilité
par position. Les limites sont `MAX_NOISE_RATIO` (1,5) fois
`mlx_bf16_noise.mean_kl` et `mlx_bf16_noise.p99_delta_p` du payload JSON
`tests/fixtures/goldens/kld/gemma4_meta.bin`, authentifié par
`gemma4_meta.json` via le chargeur golden ; le même garde top-1 est conservé.
Ce test accepte le checkpoint local indiqué par `SARAGOSSA_GEMMA4_DIR` :
la référence per-op et le candidat partagent les mêmes poids. Il ne compare
pas ces poids au golden MLX. Une trace locale au thread, enregistrée après
le dispatch et également affichable sous trace, doit prouver l'exécution de
`causal_attention_prefill_long_d512_f32` à la longueur entière du prompt.
Les tests par longueur ci-dessous couvrent aussi le chemin mid sur poids
réels, avec les mêmes gardes.
Aucun seuil ni golden n'est modifié.

Les diagnostics `gemma4_prefill_ab_1k/4k` conservent la comparaison des
200 tokens et affichent la première divergence ; leur succès ne qualifie
pas la qualité. L'identité stricte reste testée pour le decode seul sur
200 tokens. Pour le prompt court (32 tokens de corpus, 45 avec l'enveloppe
chat), une tentative d'identité stricte sur 200 tokens échoue au token 4.
Le test `gemma4_prefill_ab_short` en conserve le diagnostic sur les
200 tokens, sans seuil de tolérance numérique.
L'oracle historique
`qwen_loader::gemma4_tests::gemma4_resident_prefill_tokens_match_per_op`
conserve, lui, son assertion d'identité stricte sur le prompt français
« Explique en français, en deux phrases, pourquoi le ciel est bleu. »
avec son budget inchangé de 32 tokens générés (identité vérifiée sur les
32 tokens pour un prompt de 28 tokens) ; il vérifie désormais aussi
que le prompt, enveloppe chat comprise, reste sous 64 tokens.

Le kernel causal long couvre les têtes globales d512 : chaque thread
accumule deux colonnes indépendantes. Cette variante est séparée du kernel
historique d128/d256. Elle matérialise le score et le poids en f32 comme le
kernel mid pour éviter les contractions de fast-math à ces frontières ;
leur égalité en bits est testée à 256 et 257 tokens. Les oracles CPU
conservent le seuil de 5e-4 aux longueurs 2048, 2049 et 4109.
Toute erreur d'exécution résidente Gemma 4, y compris d512, remonte à
l'appelant ; elle ne relance pas silencieusement le per-op. Le test
`gemma4_resident_prefill_propagates_execution_error` injecte une incohérence
et vérifie aussi le dispatch qui pourrait déclencher ce repli.
`gemma4_resident_prefill_token_limit_before_embedding` vérifie l'admission
à la borne, le refus juste au-delà avant embedding, puis le repli per-op.
`gemma4_prefill_max_tokens_defaults_and_override` couvre le défaut 3000 et
le parsing de la surcharge.

## Diagnostic par longueur et par variante

Mesures sur le snapshot `mlx-community/gemma-4-26b-a4b-it-4bit`
`efbeee6e582ebfd06abc9d65e90839c4b5d2116b`, sous verrou GPU, le
2026-11-14. Les SHA de configuration et de tokenisation correspondent au
manifeste du golden. Les longueurs indiquées incluent les 13 tokens de
l'enveloppe chat ; les tests vérifient leur longueur exacte. Le nouveau
point 1024 diffère donc du diagnostic greedy historique à 1037 tokens.
L'oracle à 4109 conserve exactement le prompt du test long initial.

Chaque oracle note 128 positions après le prefill, rejoue les tokens du
per-op, et conserve le decode per-op des deux côtés. Les seuils sont
inchangés : KL moyen <= 0,002078747419919901 ; p99 delta de probabilité
<= 0,10483874146608464 ; top-1 >= 98,828125 %. La calibration vient du
payload authentifié `kld/gemma4_meta`, sans remplacement de golden.

Pour les ablations de 4109 tokens, une seule référence per-op est enregistrée
et réutilisée entre candidats, avec les bits f64 des logprobabilités,
les ids top-64 et les tokens forcés conservés. Le prompt et les flags de
la référence sont contrôlés avant relecture. Ce transport temporaire de
diagnostic ne fait pas partie du test livré : celui-ci recalcule toujours
sa référence. Le cas GEMM f32 des deux côtés possède sa propre référence.

- Mid étendu : seul le corps de `causal_attention_prefill_long_d512_f32`
  est remplacé par celui du mid via `SARAGOSSA_RUST_KERNELS_PATH`, avec
  `scores[4128]` et le garde correspondant. Les autres sources sont identiques.
- KV f32 : `SARAGOSSA_RUST_PREFILL_KV_BF16_STORE=0`.
- Découpage demandé : `SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK=1024`,
  en conservant `SARAGOSSA_RUST_PREFILL_CHUNK=0` pour la référence.
- GEMM f32 candidat : portée stricte limitée au prefill candidat, sans
  changer la référence ni le decode.
- GEMM f32 des deux côtés : `SARAGOSSA_RUST_QMM_NA=0`,
  `SARAGOSSA_RUST_QMM_NA_FUSED_TILED=0` et
  `SARAGOSSA_RUST_QMM_NA_GS128=0` ; le decode mono-ligne reste en f32.
- Globales per-op : cinq couches globales d512 en per-op, 25 couches
  glissantes résidentes ; contrôle entièrement résident soumis lui aussi
  couche par couche.

| Longueur totale / variante | KL moyen | Ratio KL | Ratio p99 Δp | Top-1 | Résultat |
|---|---:|---:|---:|---:|---|
| 64 | 0.0006723507 | 0.485160 | 0.684431 | 98.4375 % | ÉCHEC |
| 128 | 0.0011947210 | 0.862097 | 0.445878 | 98.4375 % | ÉCHEC |
| 256 | 0.0006214619 | 0.448440 | 0.528758 | 100.0000 % | OK |
| 512 | 0.0007905451 | 0.570448 | 0.711504 | 99.2188 % | OK |
| 1024 | 0.0084968743 | 6.131246 | 2.613435 | 98.4375 % | ÉCHEC |
| 1024, GEMM f32 candidat seul | 0.0037073627 | 2.675190 | 1.903402 | 98.4375 % | ÉCHEC |
| 2048 | 0.0041084255 | 2.964592 | 1.574993 | 96.0938 % | ÉCHEC |
| 3000 | 0.0014868029 | 1.072860 | 0.896341 | 99.2188 % | OK |
| 4109 | 0.0035143076 | 2.535884 | 1.451535 | 98.4375 % | ÉCHEC |
| 4109, mid étendu | 0.0035143076 | 2.535884 | 1.451535 | 98.4375 % | ÉCHEC |
| 4109, KV bf16 désactivé | 0.0035143076 | 2.535884 | 1.451535 | 98.4375 % | ÉCHEC |
| 4109, découpage demandé à 1024 | 0.0035143076 | 2.535884 | 1.451535 | 98.4375 % | ÉCHEC |
| 4109, contrôle par couche | 0.0035143076 | 2.535884 | 1.451535 | 98.4375 % | ÉCHEC |
| 4109, globales d512 per-op | 0.0028323905 | 2.043820 | 1.258485 | 98.4375 % | ÉCHEC |
| 4109, GEMM f32 candidat seul | 0.0080728107 | 5.825246 | 3.119315 | 96.8750 % | ÉCHEC |
| 4109, GEMM f32 des deux côtés | 0.0000296535 | 0.021398 | 0.102223 | 100.0000 % | OK, référence modifiée |

Les ratios ne sont pas monotones en longueur : chaque longueur termine le
corpus à un autre endroit et détermine sa propre continuation forcée.
Les échecs à 64 et 128 portent uniquement sur le top-1 (126/128).
Le succès à 256, 512 ou 3000 ne qualifie donc pas uniformément tous les prompts
plus courts. Aucune tolérance supplémentaire n'est appliquée à ces cas.
Le mid étendu et le contrôle par couche reproduisent exactement les
métriques du résident standard à 4109. Le passage des seules cinq couches
globales en per-op réduit le KL, mais échoue encore sur KL et top-1 : ces
couches ne suffisent donc pas à expliquer ou corriger la dérive.
Le succès du GEMM f32 des deux côtés
est un diagnostic distinct : référence **et continuation** changent ; il
ne suffit pas à qualifier un candidat contre la référence standard.
Avec cette référence inchangée, le passage du seul candidat en f32 aggrave
les trois métriques : ce changement n'est pas retenu comme correctif.
À 1024, il réduit le KL mais échoue encore aux trois gardes.

Les traces du défaut, du KV bf16 désactivé et du découpage demandé indiquent
`rows=4109 chunk_rows=4109 chunks=1 kv_bf16_store=false` : ces deux flags
ne modifient pas le parcours effectif. Le stockage bf16 ne s'applique qu'au
prefill découpé, que d512 refuse actuellement. Ils ne peuvent donc pas
être la cause de cet échec. Le kernel long accumule déjà en f32 ; sa
substitution par le mid étendu ne corrige rien.

La cause précise n'est pas isolée à un opérateur. Les mesures montrent une
sensibilité au parcours GEMM/aux arrondis du prefill complet, présente aussi
avant le seuil long de 2048, et non une croissance monotone avec la longueur.
Aucun correctif local testé ne satisfait l'oracle avec sa référence inchangée.
Le plafond est donc ramené à 3000, le plus grand point passant mesuré dans
cette série ; le défaut et le parsing de sa surcharge sont testés. L'oracle
long reste ignoré avec son échec connu, ainsi que les points courts en échec.

## Validation sous verrou GPU, sans benchmark

Après changement du défaut à 3000, un rejeu sans surcharge de borne et avec
référence recalculée reproduit exactement les métriques du point 3000.
Les 12 tests ciblés de flags, admission/refus à 3000/3001, propagation
d'erreur, kernels Gemma et garde KL passent également sous verrou.
`cargo fmt --all -- --check`, Clippy tous targets/features avec
`-D warnings`, `cargo build` et la compilation release des tests passent.
Les échecs numériques ignorés du tableau restent des limites connues.

Compiler avant de prendre le verrou :

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build
cargo test --release --lib --no-run
```

Pour KL, définir `SARAGOSSA_GEMMA4_DIR` vers le snapshot **4bit de référence**
identifié dans `tests/fixtures/goldens/kld/gemma4.json`. Le q4a8-ls n'a pas la
même provenance : ne pas contourner le contrôle SHA pour l'utiliser ici.
Ces deux tests qualifient uniquement les prefills courts :

```sh
python3 -I -B tools/gpulock.py run --owner gemma4-kl --prio 10 -- \
  env SARAGOSSA_REQUIRE_REAL_MODELS=1 SARAGOSSA_RUST_PREFILL_RESIDENT=1 \
  SARAGOSSA_RUST_PREFILL_CHUNK=0 SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK=0 \
  cargo test --release --lib \
  decoder::tests::gemma4_oracles::gemma4_prefill_reference_kld_ \
  -- --ignored --nocapture --test-threads=1
```

L'oracle long s'exécute une seule fois, sans benchmark. La borne est relevée
explicitement à 4128 pour laisser passer l'enveloppe chat ; cela ne modifie
pas le défaut 3000. Le test échoue si son prompt sort de [4096, 4128] ou si
le kernel long d512 n'est pas dispatché. Définir `SARAGOSSA_GEMMA4_DIR` vers
les poids à qualifier (par exemple le même snapshot 4bit) :

```sh
python3 -I -B tools/gpulock.py run --owner g4fix --prio 5 -- \
  env SARAGOSSA_RUST_PREFILL_RESIDENT=1 SARAGOSSA_RUST_PREFILL_CHUNK=0 \
  SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK=0 SARAGOSSA_RUST_TRACE_PREFILL=1 \
  SARAGOSSA_RUST_PREFILL_RESIDENT_GEMMA4_MAX_TOKENS=4128 \
  cargo test --release --lib \
  decoder::tests::gemma4_oracles::gemma4_prefill_long_per_op_kld_resident \
  -- --ignored --exact --nocapture --test-threads=1
```

Pour reproduire la série par longueur, reprendre la commande longue en
remplaçant son filtre par
`decoder::tests::gemma4_oracles::gemma4_prefill_per_op_kld_resident_N`,
avec `N` dans `64 128 256 512 1024 2048 3000`. Conserver la surcharge
à 4128 pour les diagnostics au-delà de 3000. Pour vérifier le défaut livré,
lancer le point 3000 **sans** cette surcharge. Les échecs numériques de la
table sont attendus ; ne pas les confondre avec un refus d'admission.

Les ablations persistantes utilisent la même commande longue, en ajoutant
un seul réglage à la fois : `SARAGOSSA_GEMMA4_ORACLE_GEMM_F32=1`,
`SARAGOSSA_GEMMA4_ORACLE_GLOBAL_PER_OP=1` (ou `control`), ou
`SARAGOSSA_RUST_PREFILL_KV_BF16_STORE=0`. Pour le découpage, remplacer
`SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK=0` par `1024`. Les flags
`SARAGOSSA_GEMMA4_ORACLE_*` ne concernent que les tests ; aucune sélection
de couches ni surcharge GEMM supplémentaire n'est livrée en production.
La variante mid étendue est une substitution locale de source Metal
décrite plus haut, sans modification des kernels livrés.

L'oracle externe reste également utilisable dans deux processus distincts :

```sh
for resident in 0 1; do
  python3 -I -B tools/gpulock.py run --owner gemma4-kl --prio 10 -- \
    env SARAGOSSA_REQUIRE_REAL_MODELS=1 SARAGOSSA_RUST_PREFILL_RESIDENT=1 \
    SARAGOSSA_RUST_PREFILL_RESIDENT_GEMMA4="$resident" \
    SARAGOSSA_RUST_PREFILL_CHUNK=0 SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK=0 \
    cargo test --release --lib golden::kld::external_kld_gemma4 \
    -- --ignored --exact --nocapture --test-threads=1
done
```

L'identité du prompt français court utilise également le snapshot 4bit
historique (résolu dans le cache HF local) :

```sh
python3 -I -B tools/gpulock.py run --owner gemma4-short --prio 10 -- \
  env SARAGOSSA_REQUIRE_REAL_MODELS=1 SARAGOSSA_RUST_PREFILL_RESIDENT=1 \
  SARAGOSSA_RUST_PREFILL_CHUNK=0 SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK=0 \
  cargo test --release --lib \
  qwen_loader::gemma4_tests::gemma4_resident_prefill_tokens_match_per_op \
  -- --ignored --exact --nocapture --test-threads=1
```

Définir ensuite `SARAGOSSA_GEMMA4_DIR` vers le checkpoint **q4a8-ls** local
pour l'identité et les diagnostics A/B. Les tests exécutent chaque prefill
une fois, sans instantané ni cache de préfixe, puis exactement 200 tokens
sans arrêt anticipé sur EOS. Les prompts longs sont construits à partir du
corpus KL versionné, tronqué à 1024/4096 tokens avant l'enveloppe chat.
Le decode est comparé séparément au per-op à partir du même cache et du
même état final de prefill. Les filtres séparent les deux provenances : ne
pas lancer tout le module avec un unique checkpoint q4a8-ls.

```sh
for test in gemma4_decode_identity_200 gemma4_prefill_ab_short \
            gemma4_prefill_ab_1k gemma4_prefill_ab_4k; do
  python3 -I -B tools/gpulock.py run --owner gemma4-greedy --prio 10 -- \
    env SARAGOSSA_RUST_PREFILL_RESIDENT=1 SARAGOSSA_RUST_PREFILL_CHUNK=0 \
    SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK=0 \
    SARAGOSSA_RUST_PREFILL_RESIDENT_GEMMA4_MAX_TOKENS=4128 \
    cargo test --release --lib "decoder::tests::gemma4_oracles::$test" \
    -- --ignored --exact --nocapture --test-threads=1
done

python3 -I -B tools/gpulock.py run --owner gemma4-kernel --prio 10 -- \
  cargo test --release --lib causal_attention_prefill_long_d512 \
  -- --nocapture --test-threads=1

python3 -I -B tools/gpulock.py run --owner gemma4-error --prio 10 -- \
  cargo test --release --lib gemma4_resident_prefill_ \
  -- --nocapture --test-threads=1
```

Ces commandes vérifient la qualité, l'identité ciblée et les chemins
exécutés ; elles ne lancent aucun benchmark.
