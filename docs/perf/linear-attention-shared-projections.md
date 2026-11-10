# Projections linear-attention partagées

Les projections compatibles u4 gs64 occupent désormais trois allocations par
couche : les codes packed, les scales bf16 et les biases bf16. Dans chacune,
les lignes suivent l'ordre `qkv | z | b | a`. Les quatre poids originaux et les
deux concats résidents utilisent des références et des offsets dans ce stockage.
Aucun kernel Metal ni oracle/golden n'est modifié.

## Construction et consommateurs

Références de cette révision ; les numéros indiquent les points d'entrée.
Les noms `b` et `a` correspondent respectivement à beta et gate dans les
structures résidentes.

| Chemin | Source | Utilisation |
|---|---|---|
| Chargement single-copy | `src/qwen_loader/quantized.rs:325` | Upload packed u32 et lecture directe des paramètres bf16 dans des MTLBuffer partagés ; pas de mmap de fichier |
| Lecture bf16 | `src/metal_backend/buffers.rs:442` | `new_buffer(StorageModeShared)`, puis lecture du fichier dans `contents()` |
| Assemblage | `src/decoder/attention_ops.rs:140` | Les quatre `Linear` originaux deviennent les propriétaires des poids |
| Rattachement du runtime | `src/decoder.rs:2409`, `src/decoder.rs:2457`, `src/linear_attention.rs:259` | Single-copy, puis regroupement avant résolution des caches |
| Matérialisation avant drop CPU | `src/decoder.rs:1892`, `src/decoder.rs:2558` | Résout encore les quatre originaux et les variantes résidentes |
| Construction full | `src/metal_backend/buffers.rs:138` | `resident_la_in_proj_concat`, ordre qkv/z/b/a |
| Construction variantes | `src/metal_backend/buffers.rs:162`, `src/metal_backend/buffers.rs:235` | `z_beta_gate`, paires qkv/z et b/a, repli split pour single-copy |
| Concat générique | `src/metal_backend/matmul.rs:830` | Reconnaît les plages contiguës avant tout cache ou allocation ; ancien constructeur conservé |
| Setup decode | `src/decoder/resident/setup.rs:834`, `src/decoder/resident/setup.rs:859` | Résout les poids des couches LA MoE et denses |
| Decode mono résident | `src/decode_resident/layers.rs:1966`, `src/decode_resident/layers.rs:2162` | Consomme `MetalLinearAttnResidentDenseWeights` |
| Projections decode | `src/metal_backend/linear_attention.rs:1826`, `src/metal_backend/linear_attention.rs:1922` | Full, puis qkv + z/b/a, puis paires concat/split |
| Decode en lignes | `src/decode_resident/layers.rs:2060`, `src/decode_resident/layers.rs:2236`, `src/decode_resident/layers.rs:2281` | Même stockage pour les appels en lignes et captures récurrentes |
| Calcul LA en lignes | `src/metal_backend/linear_attention.rs:2746` | Full ou paires ; conv/SSM séquentiels, captures et arbres conservés |
| Prefill résident | `src/metal_backend/prefill/global_attention.rs:361`, `src/metal_backend/linear_attention.rs:1044` | Projette les quatre originaux via `encode_matmul_weight` ; ne dépend pas du concat full |
| Validation prefill | `src/metal_backend/prefill/scratch.rs:558` | Vérifie biais absents et formes des originaux |
| Wrappers Metal mono/batch | `src/metal_backend/linear_attention.rs:355`, `src/metal_backend/linear_attention.rs:534`, `src/metal_backend/linear_attention.rs:1532` | Résolution résidente ou projections séparées des originaux |
| CPU et per-op | `src/linear_attention.rs:655`, `src/linear_attention.rs:685`, `src/linear_attention.rs:709` | Lecture CPU inchangée des vues, `project_biasless` ou matmul individuel |
| Lightbatch duo | `src/decode_resident/duo.rs:275`, `src/metal_backend/lightbatch_duo.rs:255` | Concat full, qmm2 ; offset de base du full égal à zéro |
| Lightbatch multi | `src/decode_resident/multi.rs:382`, `src/metal_backend/lightbatch_multi.rs:75` | Full, z/b/a ou paires ; dispatch commun des matmuls résidents |
| DFlash cible | `src/decoder/resident/dflash.rs:569`, `src/decoder/resident/verify.rs:337` | Vérification en lignes via les mêmes poids ; restauration des états, pas de stockage de poids supplémentaire |
| MTP cible | `src/decoder/resident/mtp.rs:1682`, `src/decoder/resident/mtp.rs:2419`, `src/decoder/resident/mtp_block.rs:249` | Couches LA du tronc et captures via le même encodeur ; la couche propre au draft MTP est full-attention |
| Flash-Next | `src/decode_resident/flash_next/weights.rs:405`, `src/decode_resident/flash_next/session.rs:2589` | Charge ses originaux puis utilise le résolveur commun ; b/a denses, donc chemin mixte historique conservé |
| Oracle CPU expérimental | `src/qwen4_exp/forward.rs:143` | Originaux qkv/z quantifiés et b/a denses, sans regroupement |
| Profilage | `src/metal_backend/linear_attention.rs:2233`, `src/metal_backend/linear_attention.rs:2490` | Résolveur commun, mesure des variantes full et z/b/a |
| Tests existants | `src/linear_attention/tests.rs:201`, `src/linear_attention/tests.rs:244`, `src/metal_backend/tests/rows_exact.rs:397`, `src/metal_backend/prefill/tests.rs:1151` | Fixtures CPU/Metal, oracle en lignes, prefill synthétique ; conservés |
| Tests et fixtures de chargement | `src/decoder/tests.rs:1847`, `src/qwen_loader/test_fixtures.rs:632`, `src/decode_resident/flash_next/tests.rs:4842` | Noms, formes et cas de projections ; conservés |

Les autres références aux noms de ces poids sont la sélection des tenseurs
(`src/qwen_loader.rs:34`), la validation des formes
(`src/qwen_loader/shape_validation.rs:164`, `src/qwen_loader/qwen4_exp.rs:452`),
et les règles/fixtures du convertisseur (`src/devtools/convert/rules.rs:167`,
`src/devtools/convert/tests.rs:105`). Elles ne lisent pas les buffers résidents.

Les matmuls communs sont dans `matmul.rs` (per-op), `matmul_weight.rs`
(originaux encodés) et `matmul_resident.rs` (handles résolus). Les bindings
spécialisés sont dans `matmul_encode.rs` et `matmul_align64.rs`.

## Disposition et choix numériques

Pour une matrice `[out, in]` u4 gs64, une ligne occupe `in/2` octets de codes
et `2 × in/64` octets dans **chacun** des paramètres bf16. Une concaténation
sur les sorties conserve donc les groupes de quantification et toutes les
lignes. Les sources doivent avoir le même `in`, des buffers autonomes à offset
zéro, et une taille multiple de 256 octets dans les trois composantes.
L'alignement de 256 est volontairement plus strict que l'alignement scalaire
de 4 octets des bindings de codes ; aucun padding n'est introduit.

Pour le 27B (`in=5120`), les offsets sont :

| Vue | Lignes avant la vue | Offset packed | Offset scales/biases, chacun |
|---|---:|---:|---:|
| qkv | 0 | 0 | 0 |
| z et z/b/a | 10 240 | 26 214 400 | 1 638 400 |
| b | 16 384 | 41 943 040 | 2 621 440 |
| a | 16 432 | 42 065 920 | 2 629 120 |

Tous sont multiples de 256. Un blit initial copie les octets des trois
composantes sans conversion. Après complétion, les poids adoptent les vues ;
les références sources et celles du command buffer sont relâchées par couche
(pool d'autorelease local). Les IDs de poids restent stables. Les vues gardent
les buffers vivants et restent lisibles sur CPU après le drop des payloads.

Un offset non nul forçait auparavant les matmuls vers le kernel générique.
Le marqueur interne `relocated_u4` indique que la source était à offset zéro
avant regroupement : seuls ces poids conservent leur sélection habituelle.
Les bindings u4 qmv, qmm2/qmm8, skinny, qmv-T et NAX transmettent les offsets
sans changer leurs réductions, tailles de groupes ni conversions. Les autres
vues (MoE, pools) gardent leur ancien dispatch. Les paires single-copy restent
split pour ne pas changer l'algorithme choisi.

## Gain attendu

Pour 48 couches compatibles, scales **et** biases bf16 inclus :

| Allocation supprimée | Octets | Mio (2^20 octets) |
|---|---:|---:|
| Copie full qkv/z/b/a | 2 278 195 200 | 2 172,65625 |
| Copie z/b/a | 862 617 600 | 822,65625 |
| Total | **3 140 812 800** | **2 995,3125** |

Cela représente **3 140,8 Mo décimaux**, soit environ 3,14 Go. Les « Mo » de
la trace initiale sont des Mio. Le supplément transitoire du regroupement est
borné à une couche : 47 462 400 octets, soit 45,26 Mio. Ce sont des longueurs
de buffers ; les arrondis du driver peuvent légèrement changer le compteur
physique. Aucun modèle réel n'a été exécuté pour cette validation.

## Retour arrière et limites

`SARAGOSSA_RUST_LA_SHARED_PROJECTIONS=0`, défini avant le démarrage, restaure
le stockage et le dispatch historiques. Le flag est documenté dans
`src/runtime_flags.rs`. `RETI_RUST_SINGLE_COPY_WEIGHTS=0` reste inchangé et
désactive aussi ce regroupement, qui intervient dans la préparation single-copy.

Les projections denses, quantifications mixtes, autres groupes/bits, frontières
non alignées, et vues préexistantes de pools/shards conservent l'ancien chemin.
Les fichiers ne sont ni remappés ni réécrits. La résolution après regroupement
ne copie plus les concats, même si le cache de concat est désactivé. Les
décisions adaptatives de création des variantes full/split restent celles du
résolveur existant ; près du seuil de budget, une mémoire disponible différente
peut influencer cette politique, comme avant.

La validation bit à bit couvre une couche synthétique et les kernels concernés,
pas une génération complète 27B, ni un benchmark de latence de chargement.
Le regroupement ajoute un blit et une attente par couche au rattachement du
runtime ; il remplace les conversions et uploads des concats ultérieurs.

## Validation

Les tests `shared_la_*` comparent les bits des sorties et des états conv/SSM
pour 1, 2, 5 et 17 lignes : decode full, variante z/b/a, split et encodeur de
prefill résident. Ils comparent aussi les projections originales CPU/per-op,
les kernels à offsets (y compris NAX si disponible), les longueurs des buffers
uniques, la survie des handles après drop, l'idempotence et les replis mixtes,
non alignés ou poolés. L'absence de Metal fait échouer ces nouveaux tests.

Commandes exécutées avec Cargo hors ligne et sous `taskpolicy -b` :

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build
cargo test --all-features --lib -- shared_la linear_attention::tests \
  quantization::tests metal_backend::matmul::cache_key_tests \
  metal_backend::prefill::tests \
  metal_backend::tests::affine_matmul_matches_cpu \
  metal_backend::tests::affine_qmv_u4_gs64_align64_matches_generic_and_routes \
  metal_backend::tests::affine_qmm_na_fused_tiled_u4_align64_matches_generic_and_routes \
  --test-threads=1 --nocapture
```

Les passes plus larges ont été interrompues dans les grandes fixtures
synthétiques Flash-Next et l'oracle CPU du GEMM flint pour limiter le travail.
Leurs résultats incomplets ne sont pas comptés comme une suite réussie.

Résultats sur cette révision : fmt, Clippy avec `-D warnings` et build réussis.
La suite ciblée termine en 26,70 s :

```text
test metal_backend::tests::shared_la_allocated_bytes_and_lifetime ... ok
test metal_backend::tests::shared_la_decode_and_prefill_bit_exact ... ok
test metal_backend::tests::shared_la_offset_kernels_bit_exact ... ok
test metal_backend::tests::shared_la_relocation_releases_sources_and_skips_incompatible ... ok
test result: ok. 36 passed; 0 failed; 1 ignored; 0 measured; 832 filtered out
```

Les branches NAX du nouveau test ont été exécutées. Le test ignoré préexistant
est `prefill_resident_capture_tail_matches_full`. La fixture de mémoire passe
de 18 à 3 buffers uniques, de 3 124 224 à 1 262 592 octets logiques : gain
exact de **1 861 632 octets**, soit le full plus z/b/a. Le compteur global du
device, seulement informatif, inclut aussi les ressources du blit et les
arrondis du driver ; il ne sert pas à une assertion dépendant des autres tests.

Aucun téléchargement, installation Python, modification de `models/` ou modèle
réel exécuté. Cargo signale seulement l'avertissement préexistant de
compatibilité future de la dépendance `block 0.1.6`.
