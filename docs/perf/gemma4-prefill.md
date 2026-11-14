# Qualification du prefill Gemma 4

## Décision du 2026-11-14

Le prefill résident Gemma 4 est **activé par défaut**, sur décision du
mainteneur. `SARAGOSSA_RUST_PREFILL_RESIDENT_GEMMA4=0` rétablit le per-op.
L'attention du prefill per-op est calculée en f32 sur CPU, celle du résident
sur GPU : les deux chemins ont des arrondis valides mais différents.
Le critère de qualité est désormais le passage des **deux chemins** contre
l'oracle KL de référence, avec les seuils et la provenance inchangés.

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

Les anciennes assertions d'identité longue `gemma4_prefill_identity_1k/4k`
sont remplacées explicitement par `gemma4_prefill_reference_kld_per_op` et
`gemma4_prefill_reference_kld_resident`. Ces tests partagent le calcul et
les seuils de `golden::kld::external_kld_gemma4` : KL moyen et p99 Δp au plus
1,5 fois le bruit bf16, top-1 au moins égal à celui de la référence bf16.
Ils exigent les poids locaux, vérifient leur SHA et refusent un chemin
résident indisponible. Aucun seuil ni golden n'est modifié.

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
Les deux tests remplacent les anciennes exigences d'identité longue :

```sh
python3 -I -B tools/gpulock.py run --owner gemma4-kl --prio 10 -- \
  env SARAGOSSA_REQUIRE_REAL_MODELS=1 SARAGOSSA_RUST_PREFILL_RESIDENT=1 \
  SARAGOSSA_RUST_PREFILL_CHUNK=0 SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK=0 \
  cargo test --release --lib \
  decoder::tests::gemma4_oracles::gemma4_prefill_reference_kld_ \
  -- --ignored --nocapture --test-threads=1
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
    cargo test --release --lib "decoder::tests::gemma4_oracles::$test" \
    -- --ignored --exact --nocapture --test-threads=1
done

python3 -I -B tools/gpulock.py run --owner gemma4-kernel --prio 10 -- \
  cargo test --release --lib causal_attention_prefill_long_d512 \
  -- --nocapture --test-threads=1

python3 -I -B tools/gpulock.py run --owner gemma4-error --prio 10 -- \
  cargo test --release --lib gemma4_resident_prefill_propagates_execution_error \
  -- --nocapture --test-threads=1
```

Ces commandes vérifient la qualité, l'identité ciblée et les chemins
exécutés ; elles ne lancent aucun benchmark.
