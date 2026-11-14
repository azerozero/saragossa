# Qualification du prefill Gemma 4

## Décision du 2026-11-14

Le prefill résident Gemma 4 est **activé par défaut**, sur décision du
mainteneur. `SARAGOSSA_RUST_PREFILL_RESIDENT_GEMMA4=0` rétablit le per-op.
L'attention du prefill per-op est calculée en f32 sur CPU, celle du résident
sur GPU : les deux chemins ont des arrondis valides mais différents.
Les **deux chemins** sont qualifiés sur le corpus KL externe court, avec
les seuils et la provenance inchangés. Un oracle long distinct compare le
résident au per-op ; il ne constitue pas une référence MLX longue.

Le résident est éligible jusqu'à **4096 tokens au total**, enveloppe chat
comprise. `SARAGOSSA_RUST_PREFILL_RESIDENT_GEMMA4_MAX_TOKENS` remplace cette
borne par une valeur positive explicite ; zéro, valeur invalide ou absente
conservent 4096. Au-delà, le résident renvoie `Ok(None)` avant l'embedding
et les allocations : l'appelant utilise son repli per-op habituel, sans
changement de `main`. `SARAGOSSA_RUST_TRACE_PREFILL=1` affiche ce refus.
Le d512 ne supporte pas le prolongement résident par morceaux :
`SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK` ne borne donc pas sa mémoire.

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

L'unique passage de l'oracle long sous verrou a confirmé cinq dispatches
`causal_attention_prefill_long_d512_f32` à 4109 tokens. Le KL dépasse la
limite 0,0020787474 (ratio maximal 1,5) ; le p99 reste sous 0,1048387415.
Le top-1 est également sous le garde bf16 de 98,828125 %. **La qualification
longue n'est donc pas acquise**. Le test reste ignoré en CI ordinaire et
échoue lorsqu'on le lance explicitement sur ce checkpoint. Aucun seuil
n'a été ajusté. La borne 4096 limite les allocations ; cet essai à 4109 ne
prouve pas la qualité numérique à 4096 ni aux autres longueurs voisines.


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
Il ne qualifie pas séparément le chemin mid de prefill sur poids réels.
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
`gemma4_prefill_max_tokens_defaults_and_override` couvre le défaut 4096 et
le parsing de la surcharge.

## Validation sous verrou GPU, sans benchmark

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
pas le défaut 4096. Le test échoue si son prompt sort de [4096, 4128] ou si
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
