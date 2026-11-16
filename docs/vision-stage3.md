# Qualification image → LLM — étape 3

Base : `2752d0f`, branche `feat/vision-mrope`. Embeddings Python fournis,
sans ViT Rust. Les transports média restent refusés.

La calibration Transformers 5.8.1 / torch 2.12.0 du Qwen dense BF16 a été
figée dans `12f9626` avant la première comparaison des logits Rust.
Elle couvre six prompts, dont trois multi-images, puis 16 tokens imposés :
17 états × 248 320 logits par prompt. Chaque seuil RMS, max-abs et erreur
maximale de probabilité vaut exactement deux fois le bruit BF16 observé.
SHA256 du manifeste :
`aeb004947b672651f35d1510fa2794cd6376e67d54234cf65beddfaebec59b41`.
Les 851 tenseurs LLM, les embeddings et les 146 sources exécutées sont
empreintés ; les douze payloads sont des logits synthétiques, pas des poids.

| Contrôle | Périmètre | Résultat |
|---|---|---|
| Insertion et positions | 24 prompts Python, quatre checkpoints | Exactes |
| Cache SSD v5 | Delta, reprise avant/après image, falsifications | 3 tests réussis |
| Cache RAM | Contenu, paramètres et frontières des médias | Hit/miss conformes |
| A/B réel Qwen3.6 | Une et plusieurs images, 16 tokens AR/MTP/DFlash | Identiques, acceptation et rollback couverts |
| Texte Qwen3.8 u4ls | Oracle strict, 96 états | 0 désaccord, 0 dépassement |
| Texte Qwen3.6 BF16 | Oracle historique, 96 états | Réussi, 2 désaccords autorisés par sa porte existante |
| Logits vision f32 | Per-op, résident entier et fractionné | 306/306 états, trois portes respectées |

Les pires rapports erreur/seuil sont **0,242916** (RMS), **0,245259**
(max-abs) et **0,922630** (probabilité), tous inférieurs à 1. Les maxima
observés sont respectivement 0,021756, 0,165569 et 0,011912 ; chaque état
est comparé à ses propres seuils, sans recalibrage.

`cargo fmt --all -- --check`, Clippy toutes cibles/features avec
`-D warnings`, `cargo build` et `cargo test --release --all-features`
terminent avec le code 0 : **1 301 tests réussis, 268 ignorés**. Les trois
oracles GPU ignorés sont exécutés explicitement et passent ; les six tests
Python des fixtures passent aussi. Les exécutions GPU utilisent le verrou
du projet, priorité 5, avec des tickets de moins de 30 minutes.

Le prefill vision conserve Q/K/V en f32 et utilise la récurrence linéaire
séquentielle. Les chemins NAX texte restent inchangés. M-RoPE ne s'applique
qu'aux couches full-attention ; les adresses KV suivent les tokens et la
position RoPE suit `tokens + rope_delta`.

L'oracle texte Qwen3.6 **f32 reste rouge sur la base et le candidat** :
1 désaccord greedy et 1 dépassement métrique sur 96 états, KL maximale
0,026721 pour une porte à 0,02. Les 96 tokens et métriques affichées sont
identiques entre les deux binaires. Ni cette capture ni sa porte ne sont
modifiées. Pour Qwen3.8, les poids u4ls et le tokenizer local correspondant
au SHA attendu ont été réunis par des liens temporaires en lecture seule ;
une autre copie 4 bits locale contient des poids différents.

L'A/B qualifie les draft/verifiers ordinaires de l'entrée interne. Le
prototype MTP à contrôle GPU, les lots vision et les transports média
restent hors qualification. Les performances et le pic mémoire ne font
pas l'objet d'une mesure dans cette étape. Aucun téléchargement ni écriture
dans les dossiers de poids.
