# Quantification mixte par sensibilité

Outils Python qui produisent les quantifications mixtes 3/4 bits publiées pour
ce moteur, et mesurent leur qualité. Annexes au moteur Rust : ils servent à
fabriquer des poids, pas à les exécuter.

| Fichier | Rôle |
|---|---|
| `mixed_quant.py` | Alloue deux largeurs de bits par couche selon la sensibilité, écrit un modèle MLX |
| `kl_eval.py` | Mesure KL(bf16 ‖ quantifié) et l'accord top-1, en teacher forcing |
| `test_allocation.py` | Tests de la logique d'allocation, sans dépendance à MLX |
| `convert_ls.py` | Re-quantifie en affine gs64 (3/4/5/6/8 bits, selon le template) par moindres carrés, éventuellement pondérés par l'imatrix |
| `gptq.py` | Quantification de second ordre, Hessienne complète par module/expert, mêmes largeurs et format que le template |
| `gptq_capture.py` | Collecte des Hessiennes par lots bornés, routage MoE trié ou non trié |
| `test_gptq.py` | Auto-tests stdlib et tests CPU GPTQ, MoE, GatedDeltaNet et checkpoint miniature |
| `opt_quant.py` | Ajustement d'échelle et de biais par groupe utilisé par `convert_ls.py` |
| `imatrix.py` | Capture BF16 de E[x²] par canal d'entrée, corpus indépendant |
| `sensitivity.py` | Allocation calibrée 3/4/5/6/8 bits sous budget réel, appelée par `mixed_quant.py` |
| `test_quant.py` | Auto-tests stdlib et contrôle facultatif MLX CPU sur de petits tenseurs |
| `tokenizer_sidecar.py` | Recopie `tokenizer.json` de la source à l'identique et refuse une regex de pré-tokenizer réécrite par transformers (`test_tokenizer_sidecar.py`) |
| `chat_kl.py` | KL en régime chat contre les poids officiels bf16 (référence transformers, candidat mlx-lm) ; `--prompts`, `--json` par prompt, `--full-vocab`, `--reverse` |
| `prompt_sets.py` | Ensembles figés `sets/selection.json` et `sets/validation.json` (aveugle), disjonction et SHA-256 |
| `paired.py` | Différences appariées par prompt, IC 95 % bootstrap, plan factoriel 2×2 |
| `audit_bytes.py` | bpw langage et total depuis les en-têtes safetensors, refus au-delà d'une limite |

Sans `--imatrix`, `mixed_quant.py --target-bpw` classe en espace des poids
(taille · ‖ΔW‖²/‖W‖² du fit uniforme, sans corpus). `convert_ls.py --uniform-fit`
remplit un template imatrix sans pondération. Résultats de la campagne ≤ 4,2 bpw :
[docs/perf/quant-campagne-42bpw.md](../../docs/perf/quant-campagne-42bpw.md).

## Qwen3.8-27B : alternative GPTQ à 4,2 bpw

GPTQ collecte `H = XᵀX / N` sur les entrées BF16, puis quantifie colonne par
colonne en compensant l'erreur dans les colonnes restantes avec le facteur
supérieur de `H⁻¹`. La normalisation par N ne change pas cette compensation.
Aucune rétropropagation : les couches GatedDeltaNet restent compatibles.
L'extraction, le mélange texte/chat et le refus des `PROMPTS` réservés sont
ceux de `imatrix.py`.

Avec `mlx` et `mlx-lm` compatibles avec `qwen3_5` dans l'environnement Python,
depuis la racine (et des destinations encore inexistantes) :

```bash
MODEL_ROOT="/chemin/vers/modeles"
SRC="$MODEL_ROOT/Qwen3.8-27B-BF16"
CALIB="/chemin/vers/corpus-independant.txt"  # au moins 65 536 tokens
IMATRIX="$MODEL_ROOT/qwen38-imatrix.safetensors"
TEMPLATE="$MODEL_ROOT/Qwen3.8-27B-imatrix42-template"
DST="$MODEL_ROOT/Qwen3.8-27B-GPTQ42"

python3 -I -B tools/quant/test_gptq.py
python3 -I -B tools/quant/test_gptq.py --mlx-cpu

# Omettre cette collecte si une imatrix de même provenance existe déjà.
python3 -I -B tools/quant/imatrix.py --model "$SRC" --calib "$CALIB" \
    --out "$IMATRIX" --samples 128 --length 512 --chat
python3 -I -B tools/quant/mixed_quant.py --input "$SRC" --output "$TEMPLATE" \
    --imatrix "$IMATRIX" --base-bits 3 --target-bpw 4.2 \
    --high-bits 6 --force-high 'embed|lm_head'
python3 -I -B tools/quant/gptq.py --src "$SRC" --template "$TEMPLATE" --dst "$DST" \
    --calib "$CALIB" --samples 128 --length 512 --chat \
    --dampening 0.01 --act-order --hessian-memory-mib 2048
```

Le budget est celui de `mixed_quant.py` : **au plus 4,2 bpw langage**, avec
échelles/biais et tenseurs non quantifiés compris. GPTQ conserve exactement les
largeurs, formes, dtypes et fichiers d'index du template. L'imatrix ne sert ici
qu'à l'allocation ; GPTQ recalcule les Hessiennes complètes depuis le corpus.
`allocation.json` reste le rapport de l'allocation LS du template, pas une mesure
de qualité GPTQ. `gptq.json` enregistre le SHA-256 du corpus, les paramètres, le
nombre de lignes observées par expert et les erreurs LS/GPTQ sur la calibration.

`--dampening` ajoute une fraction strictement positive de la moyenne de la
diagonale à H (défaut 0,01). Les canaux jamais activés sont régularisés avant
inversion. `--act-order`, désactivé par défaut, visite les colonnes par énergie
décroissante ; les grilles LS BF16 des **groupes d'origine de 64 colonnes**
restent fixes. Les codes sont remis dans l'ordre d'origine avant l'emballage,
sans `g_idx`, permutation des activations ni nouveau kernel côté MLX/saragossa.
`--block-size` (128 par défaut) regroupe les compensations hors bloc en matmul.

Pour chaque ligne, le résultat emballé GPTQ est comparé aux fits LS uniforme
et pondéré par `diag(H)` selon l'erreur complète `eᵀHe`. Le meilleur est gardé,
ce qui protège l'erreur sur la calibration, sans garantir la KL sur d'autres
textes. Les embeddings, dont l'entrée est un identifiant, utilisent le fit LS
uniforme. Les tests CPU vérifient aussi une amélioration du candidat GPTQ brut,
avant tout repli LS, sur des activations corrélées en 3/4/5/6/8 bits.

Chaque expert possède sa propre Hessienne, construite uniquement avec les
activations qui lui sont routées, y compris celles de `down_proj` et après le
tri des tokens par mlx-lm. Un module ou expert jamais observé fait échouer la
conversion : élargir le corpus. Aucune Hessienne partagée ou uniforme implicite.
Les experts exposés comme linéaires individuels sont capturés séparément aussi.

Le budget `--hessian-memory-mib` borne les **accumulateurs H**, pas la mémoire
totale. Une matrice exige `4 × colonnes²` octets ; le modèle BF16, les activations,
les copies de factorisation CPU et les poids de travail s'ajoutent. Une H qui
dépasse ce budget est refusée. Chaque lot relit tous les extraits en forward
BF16 avec un état récurrent vierge ; réduire le budget augmente le nombre de
passes. Les matrices quantifiées intermédiaires sont stockées temporairement
sur disque, à côté de la destination ; prévoir aussi cet espace et celui du
checkpoint final. La destination n'apparaît qu'après succès complet.

Les tests ne chargent aucun modèle externe : matrices 6×128, petits experts,
petite GatedDeltaNet et checkpoint miniature rechargé avec les modules MLX.
Sur MLX 0.32, le fixture BF16 MoE CPU convertit temporairement les opérandes de
`gather_mm` en f32, seule précision de ce kernel CPU ; le routage réel SwitchGLU
est aussi testé directement en f32, trié et non trié. La conversion réelle garde
le forward BF16 de mlx-lm sur le périphérique par défaut. Aucun résultat 27B ni
gain KL de cette recette n'est revendiqué.

## Qwen3.8-27B : recette imatrix à 4,8 bpw

Recette expérimentale, **aucun gain KL encore mesuré**. Objectif : battre
Jundot/Qwen3.8-27B-oQ4e-mtp (KL chat 0,0164, même top-1 94,66 %, 4,84 bpw
langage), contre mix46 ≈ 0,021 et u4ls ≈ 0,031 à 4,5 bpw. Le corpus doit
être indépendant des huit `PROMPTS` de `chat_kl.py` : leur présence est refusée
avant chargement, après découpage et après application du chat template,
y compris après normalisation de casse, espaces et ponctuation. Ce contrôle
ne détecte pas les paraphrases ni la contamination sémantique du corpus.

Depuis la racine du dépôt, avec `mlx`, `mlx-lm`, `torch` et `transformers`
compatibles avec `qwen3_5` dans l'environnement Python actif :

```bash
MODEL_ROOT="$HOME/workspace/ai_model"
SRC="$MODEL_ROOT/Qwen3.8-27B-BF16"
CALIB="/chemin/vers/corpus-independant.txt"  # à fournir, au moins 65 536 tokens
IMATRIX="$MODEL_ROOT/qwen38-imatrix.safetensors"
REF="$MODEL_ROOT/qwen38-chatref.json"

python3 -I -B tools/quant/test_quant.py
python3 -I -B tools/quant/test_quant.py --mlx-cpu  # petits tenseurs, aucun poids externe
python3 -I -B tools/quant/imatrix.py --model "$SRC" --calib "$CALIB" \
    --out "$IMATRIX" --samples 128 --length 512 --chat

# B : imatrix brute, allocation 4/5/6/8 à 4,8 bpw (défaut).
python3 -I -B tools/quant/mixed_quant.py --input "$SRC" \
    --output "$MODEL_ROOT/Qwen3.8-27B-imatrix48-B-template" \
    --imatrix "$IMATRIX" --target-bpw 4.8 --high-bits 6 --force-high 'embed|lm_head'
python3 -I -B tools/quant/convert_ls.py --src "$SRC" \
    --template "$MODEL_ROOT/Qwen3.8-27B-imatrix48-B-template" \
    --imatrix "$IMATRIX" --dst "$MODEL_ROOT/Qwen3.8-27B-imatrix48-B"

# C : même budget, fit et allocation avec mélange imatrix/amplitude.
python3 -I -B tools/quant/mixed_quant.py --input "$SRC" \
    --output "$MODEL_ROOT/Qwen3.8-27B-imatrix48-C-template" \
    --imatrix "$IMATRIX" --imatrix-blend --target-bpw 4.8 \
    --high-bits 6 --force-high 'embed|lm_head'
python3 -I -B tools/quant/convert_ls.py --src "$SRC" \
    --template "$MODEL_ROOT/Qwen3.8-27B-imatrix48-C-template" \
    --imatrix "$IMATRIX" --imatrix-blend --dst "$MODEL_ROOT/Qwen3.8-27B-imatrix48-C"

python3 -I -B tools/quant/chat_kl.py ref --model "$SRC" --out "$REF"
python3 -I -B tools/quant/chat_kl.py eval --model "$MODEL_ROOT/Qwen3.8-27B-imatrix48-B" --ref "$REF"
python3 -I -B tools/quant/chat_kl.py eval --model "$MODEL_ROOT/Qwen3.8-27B-imatrix48-C" --ref "$REF"
python3 -I -B tools/quant/chat_kl.py eval \
    --model "$MODEL_ROOT/Qwen3.8-27B-oQ4e-mtp-Jundot" --ref "$REF"
```

Si l'imatrix existe déjà, réutiliser le fichier et sa provenance sans relancer
`imatrix.py` : seules les quatre commandes B/C sont nécessaires à la production.
Les destinations doivent être nouvelles. Réutiliser le même `REF` pour tous
les candidats (8 réponses greedy, 192 tokens maximum, KL top-64 + reste).
Conserver le corpus, son SHA-256 enregistré dans l'imatrix, les versions des
bibliothèques et `allocation.json`. Le template est un vrai checkpoint MLX
min-max avec une entrée `config.json.quantization` par module ; prévoir
l'espace du template et du candidat final.

`--chat` enveloppe un extrait sur deux ; les extraits disjoints sont répartis
dans le corpus, sans cache partagé. Les tokens du chat template s'ajoutent
aux 512 tokens et comptent dans E[x²]. Le forward utilise les poids BF16,
sauf les exceptions prévues par mlx-lm, notamment `A_log`.

Le fit minimise Σ E[x_j²] (W_quant − W)², en gardant le meilleur groupe entre
min-max et ajustement après arrondi BF16. Le chemin sans `weight` conserve
les opérations historiques 4/8 bits. L'allocation mesure cette erreur totale
après fit en **3/4/5/6/8 bits**, puis promeut par pas **3→4→5→6→8**,
au meilleur gain par bit supplémentaire. Un pas sans gain est refusé, même
si une largeur ultérieure améliorerait le score. C'est un glouton, sans
garantie d'optimum global. Le budget réel inclut les
0,5 bpw d'échelles/biais gs64, les tenseurs langage non quantifiés et les
épinglés ; vision et MTP retirés par mlx-lm ne comptent pas. Il est contrôlé
avant sauvegarde et consigné dans `allocation.json`.

Dans ce mode, `--high-fraction` est ignoré, `--base-bits` vaut **4 par défaut**
et fixe la largeur de départ. `--low-bits`, s'il est explicite, doit lui être égal.
`--base-bits` doit être inférieur ou égal à `--high-bits`, et
`--high-bits` fixe le minimum de `--force-high` (6 par défaut, 5 ou 8 possibles).
Les trios experts gate/up/down restent uniformes, même si un seul membre
est épinglé ; `--moe` active le regex structurel historique. Un budget
infaisable est refusé. L'embedding reçoit une entrée one-hot : sans fréquences
mesurées, chaque token a une probabilité **1/vocab**, de somme 1. Son score est
donc `sum((W_quant-W)²) / vocab`, y compris lorsque le scan découpe la table
en blocs ; il ne grossit plus artificiellement avec le vocabulaire. Tout
linéaire sans vecteur valide est refusé.

### Base 3 bits : budget de 4,2 bpw

Pour viser moins que les **4,5 bpw effectifs de u4**, partir de 3 bits avec
`--base-bits 3`. Avec l'imatrix déjà capturée, depuis la racine du dépôt :

```bash
MODEL_ROOT="$HOME/workspace/ai_model"
SRC="$MODEL_ROOT/Qwen3.8-27B-BF16"
IMATRIX="$MODEL_ROOT/qwen38-imatrix.safetensors"
TEMPLATE="$MODEL_ROOT/Qwen3.8-27B-imatrix42-template"
DST="$MODEL_ROOT/Qwen3.8-27B-imatrix42"

python3 -I -B tools/quant/mixed_quant.py --input "$SRC" --output "$TEMPLATE" \
    --imatrix "$IMATRIX" --target-bpw 4.2 --base-bits 3 \
    --high-bits 6 --force-high 'embed|lm_head'
python3 -I -B tools/quant/convert_ls.py --src "$SRC" --template "$TEMPLATE" \
    --imatrix "$IMATRIX" --dst "$DST"
```

Le budget est un plafond : les promotions portent sur des modules entiers,
donc `allocation.json.actual_bpw` peut être inférieur à 4,2. Les échelles,
biais, tenseurs fixes et modules épinglés sont inclus. Le rapport conserve
aussi `base_bits`. Les codes 3 bits sont emballés en flux little-endian
contigu, 32 codes dans 3 mots u32, validés contre MLX sur CPU.

`convert_ls.py` lit seulement les en-têtes des checkpoints, puis les poids
source par mmap et blocs de 256 lignes (`--block-rows`). Les groupes de 64
et les lignes du mélange restent entiers. Le calcul de `opt_quant.fit` est
inchangé ; les payloads quantifiés sont écrits directement aux offsets des
shards du template, y compris pour un module réparti entre plusieurs shards.
Les tenseurs non quantifiés sont copiés octet pour octet. Seul l'ordre de
cumul du diagnostic RMSE global change.

La conversion refuse une destination existante et s'arrête sous 30 Gio libres
(`--min-free-gib`). Une interruption laisse des shards `.partial` à supprimer
avant de recommencer ; ils ne constituent pas un checkpoint utilisable. Aucun
chargement du modèle entier ni accumulation des sorties n'est nécessaire.
`--workers N` permet d'ajuster jusqu'à N blocs indépendants du même tenseur
en parallèle sur CPU. La file est bornée à N blocs ; la mémoire augmente
proportionnellement à N. Chaque processus écrit une plage distincte et le
diagnostic est cumulé dans l'ordre des lignes. Le défaut reste un processus.
Tests du flux, des experts fusionnés et des payloads SHA-256 sur CPU :

```bash
python3 -I -B tools/quant/test_tensor_stream.py
python3 -I -B tools/quant/test_tensor_stream.py --mlx-cpu
```

L'objectif qualité est **KL < 0,031 (u4ls)** ; il n'est pas encore mesuré
pour cette recette. L'erreur pondérée utilisée par l'allocation ne garantit
pas la KL. Comparer le candidat avec `chat_kl.py eval` sur la même référence
BF16 que u4ls avant de conclure à un gain.

### Mélange imatrix/amplitude

`--imatrix-blend` est facultatif et doit être passé **aux deux étapes**, scan
et conversion. Il applique, pour chaque élément d'un linéaire :

```text
sigma2_i = moyenne_j(W_ij²)               # ligne entière, pas groupe de 64
importance_ij = imatrix_j * sqrt(sigma2_i + W_ij²)
```

Ce mélange, inspiré de llama.cpp, fait intervenir l'amplitude des poids dans
l'importance d'activation. Le fit, le choix du meilleur groupe après arrondi
BF16 et les scores d'allocation utilisent tous ces poids par élément. Les
RMSE sont normalisées par leur somme ; le rapport restitue l'erreur totale
non normalisée. L'embedding conserve sa fréquence uniforme sans mélange.
Sans l'option, l'imatrix brute reste le défaut. `allocation.json` enregistre
les largeurs, `imatrix_blend`, la métrique et le score distinct de l'embedding ;
le convertisseur refuse un mode blend différent de celui du rapport du template.
Comparer B/C avec la même imatrix et la même référence KL : aucun gain du
mélange n'est présumé avant cette mesure.

Pour le Qwen dense 64 couches, les noms attendus sont
`language_model.model.layers.N.linear_attn.in_proj_{qkv,z,a,b}` et `out_proj`,
`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`, puis
`language_model.lm_head` : normalement **497 vecteurs**. Ils sont découverts
via `named_modules()` et interceptés par classe, sans renommage. Vérifier
ces noms avec la version installée de [qwen3_5 dans mlx-lm](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py) ;
un module jamais appelé fait échouer la capture. Le proxy diagonal ignore
les corrélations et l'amplification SSM ; en MoE, le vecteur est partagé entre
experts, sans pondération par fréquence de routage. Le gain KL et la capture
27B restent à mesurer. Les auto-tests stdlib couvrent l'emballage LE contigu
5/6 bits ; `test_quant.py --mlx-cpu` vérifie les mots contre `mx.quantize`,
la restitution par `mx.dequantize`, les fits 4/5/6/8 brut/blend, le score de
l'embedding sur plusieurs blocs et une conversion miniature complète.
Ces tests CPU ne prouvent ni la qualité 27B ni les performances du moteur.

## Méthode historique sans calibration

Pour chaque couche linéaire, on mesure l'erreur relative de la largeur basse :

```text
rel_err(W) = ‖W − dequant(quant(W))‖_F / ‖W‖_F
```

Les couches sont classées par erreur décroissante et promues en largeur haute
tant qu'un budget en paramètres n'est pas épuisé. Aucune donnée de calibration
n'intervient, donc le résultat est déterministe et reproductible.

Sur les architectures hybrides GatedDeltaNet/SSM, ce classement en espace des
poids bat le classement pondéré par activations (façon imatrix) : les
projections de porte récurrentes sont systématiquement sous-protégées par les
proxys d'activation d'entrée. Mesuré sur le 27B : KL 0,065 contre 0,18 à 0,20
pour trois variantes imatrix à bpw égal. Les méthodes à gradient (DWQ) ne
s'appliquent pas ici, MLX n'ayant pas de rétropropagation sur les opérateurs
SSM.

Deux garde-fous structurels comptent autant que le classement :

- `--force-high` épingle hors budget ce dont l'erreur se propage. Le défaut
  suit l'architecture : en dense seuls `embed` et `lm_head` sortent du budget,
  tandis que `--moe` y ajoute la récurrence SSM, le routeur, les experts
  partagés et l'attention pleine. Appliquer la recette MoE à un modèle dense
  épingle plus de la moitié des paramètres et fait grimper le bpw sans que
  rien ne le signale (mesuré sur un 0,6B : 55,7 % épinglés, 4,35 bpw au lieu
  de 4,06).
- `--moe` décide **par trio de couche** (`gate`/`up`/`down` ensemble). Les
  kernels experts fusionnés exigent des largeurs identiques ; un trio
  dépareillé perd tout chemin rapide. Mesuré : 10 couches sur 40 dépareillées
  faisaient tomber le décodage de ≥140 à 95 tok/s, sans la moindre erreur
  visible.

## Prérequis

```bash
pip install mlx mlx-lm
```

Apple Silicon requis. Prévoir la place disque du modèle bf16 source (~52 Go
pour un 27B, ~67 Go pour un 35B) et autant de mémoire unifiée que possible :
`mixed_quant.py` charge le modèle entier avant de le réécrire.

## Reproduire les modèles publiés

Récupérer d'abord le modèle bf16 de référence depuis Hugging Face. Les
commandes ci-dessous s'exécutent depuis ce répertoire (`tools/quant/`) :

```bash
# Dense 27B, mixte 3/4 bits -> 3,85 bpw, 12 Go
# 0,25 est la valeur du modèle publié (9,5 % épinglés + 25,2 % promus = 34,6 %).
python3 mixed_quant.py \
    --input Qwen3.6-27B-bf16 --output Qwen3.6-27B-mixed34 \
    --low-bits 3 --high-bits 4 --high-fraction 0.25

# MoE 35B-A3B : décision par trio, garde-fous structurels -> 3,85 bpw, 16 Go
# 0,28 est la valeur du modèle publié (7,1 % épinglés + 27,9 % promus).
python3 mixed_quant.py \
    --input Qwen3.6-35B-A3B-bf16 --output Qwen3.6-35B-A3B-mixed34 \
    --low-bits 3 --high-bits 4 --high-fraction 0.28 --moe
```

Le second produit 12 couches expertes en 4 bits sur 40, le partage annoncé sur
la carte du modèle.

`--high-fraction` est un budget **hors modules épinglés** : la part promue par
sensibilité, à laquelle s'ajoute celle de `--force-high`. La part totale en
largeur haute est donc la somme des deux, et c'est elle que rapporte la ligne
de synthèse affichée par le script.

Cette convention explique un écart apparent avec la carte du 27B, qui annonce
un « 30 % parameter budget » : le script d'origine comptait `embed` et
`lm_head` **dans** le budget, si bien que 0,30 y produisait 34,6 % de
paramètres en 4 bits. Ici les épinglés sont hors budget, donc `0,25 + 9,5 %`
donne le même 34,6 % et le même modèle. Les deux chiffres décrivent la même
quantification sous deux conventions différentes.

Puis mesurer l'écart au bf16 :

```bash
python3 kl_eval.py \
    --reference Qwen3.6-27B-bf16 --candidate Qwen3.6-27B-mixed34
```

Valeurs attendues : KL ≈ 0,065 pour le 27B, ≈ 0,162 pour le 35B-A3B. Repères
d'interprétation : KL < 0,05 excellent, < 0,15 bon, > 0,3 dégradation notable ;
un accord top-1 > 0,95 vaut quasi-sans-perte.

Le jeu de prompts est figé dans `kl_eval.py` pour que deux exécutions restent
comparables. Le remplacer via `--prompts` change les valeurs absolues, pas
seulement leur bruit : ne comparez que des mesures issues du même jeu.

## Gemma 4 26B-A4B : attention en 8 bits

Le checkpoint `mlx-community/gemma-4-26b-a4b-it-4bit` (attention, experts et
embeddings en 4 bits ; MLP dense et routeur en 8) perd nettement face aux poids
Google bf16. Mesure en régime chat avec `chat_kl.py` : 8 réponses greedy de
192 tokens au plus générées par transformers bf16, top-64 par position, KL sur
la partition top-64 + reste (1 505 positions). Le texte brut ne convient pas :
sur un extrait de Sherlock Holmes précédé du BOS, les poids officiels dans
transformers 5.22.2 ont une perplexité d'environ 10⁵, sur CPU comme sur MPS,
alors qu'ils répondent correctement en chat.

Chaque ligne « + … » modifie un seul groupe de modules par rapport au
checkpoint mlx-community ; les lignes ne se cumulent pas.

| variante (mlx-lm) | KL | même top-1 | taille |
|---|---|---|---|
| poids Google bf16 (plancher mlx/transformers) | 0,0037 | 99,1 % | bf16 |
| mlx-community 4 bits | 0,134 | 92,0 % | 13,5 Gio |
| + embeddings 8 bits | 0,129 | 92,6 % | 13,8 Gio |
| + k/v 8 bits | 0,083 | 93,4 % | 13,6 Gio |
| + attention 8 bits | 0,046 | 95,1 % | 14,0 Gio |
| + attention non quantifiée | 0,046 | 95,2 % | 15,0 Gio |
| + experts 8 bits | 0,107 | 92,2 % | 24,1 Gio |
| attention et embeddings 8 bits (q4a8) | 0,041 | 95,4 % | 14,3 Gio |
| q4a8 + `convert_ls.py` | **0,034** | **96,0 %** | 14,3 Gio |

Dans le moteur Rust, q4a8 + moindres carrés donne KL 0,031 et 96,2 % de top-1,
contre 0,132 et 91,7 % sur le checkpoint mlx-community. Les moindres carrés
seuls sur le checkpoint mlx-community ne changent rien (KL 0,137, dans le bruit) : l'erreur
dominante vient de la largeur de l'attention, pas de l'arrondi. Coût en
decode, A/B ABBA sur 5 blocs (prompt de 1 024 tokens, 256 générés, M5 Max) :
18,35 contre 18,95 tok/s, soit −3,2 %.

```bash
python3 mixed_quant.py --input gemma-4-26B-A4B-it --output gemma4-q4a8-minmax \
    --low-bits 4 --high-bits 8 --high-fraction 0 \
    --force-high 'self_attn|embed_tokens|\.mlp\.|router'
python3 convert_ls.py --src gemma-4-26B-A4B-it --template gemma4-q4a8-minmax \
    --dst gemma4-q4a8-ls
python3 chat_kl.py ref --model gemma-4-26B-A4B-it --out chatref.json  # transformers
python3 chat_kl.py eval --model gemma4-q4a8-ls --ref chatref.json
```

`convert_ls.py` suit la largeur de chaque tenseur du template (3, 4, 5, 6 ou 8 bits) et
découpe les experts fusionnés `gate_up_proj` des poids Google comme mlx-lm.

## Référence KL du moteur Rust

Avec les features `metal,devtools` (actives par défaut), sur deux modèles locaux :

```bash
saragossa run /chemin/reference --prompt "Bonjour" --max-tokens 32 --kl-record /tmp/kl-ref
saragossa run /chemin/candidat --kl-compare /tmp/kl-ref
```

Les mêmes options fonctionnent avec `--model-dir` à la place de `run`.
`--kl-record` et `--kl-compare` sont exclusifs. Aucun téléchargement : le modèle
est local. Le répertoire de sortie doit être nouveau ; il contient `ref.json`
(version 1, prompt, tokens greedy imposés, dimensions, empreintes et transformations)
et `ref.bin` (log-softmax f32 little-endian, matrice `[n, vocab]`). La ligne `i`
prédit `forced_ids[i]` après le prompt et `forced_ids[..i]`. Seule la dernière
position du prefill est évaluée, puis les `n−1` pas de decode GPU résident.
L'enregistrement fait exactement `--max-tokens` pas, y compris après EOS.

Les logits sont pris après softcapping, avant température/top-k/top-p ; ces
options de sampling n'affectent pas le diagnostic. Le KV du decode est f32.
La comparaison relit le prompt et les tokens enregistrés, refuse le repli du
decode résident, les versions inconnues, vocabulaires différents et fichiers
tronqués. Le JSON est borné à 1 Mio, le binaire à 256 Mio et les séquences à
65 536 tokens. L'empreinte SHA-256 de config et celle d'un échantillon de poids
(embeddings exécutés premier/milieu/dernier et norme finale) sont informatives : ce ne
sont ni une preuve d'identité de tous les poids ni un contrôle du tokenizer.
Conserver le même vocabulaire et la même correspondance token/id entre modèles.

`--kl-compare` écrit une ligne JSON : `kl_mean`, `kl_max` en nats,
`top1_agreement`, `forced_delta_p_mean` (écart signé candidat moins référence),
`forced_delta_p_max` (maximum absolu), `forced_abs_delta_p_mean` (moyenne absolue
du token imposé) et `mean_abs_delta_p` (moyenne absolue sur les entrées `[n, vocab]`). Les log-probabilités f32 des deux côtés sont
renormalisées en f64 pour les métriques. La KL couvre tout le vocabulaire ;
ce format est propre au moteur, sans compatibilité binaire llama.cpp.

## Modèles publiés avec ces outils

- [Qwen3.6-27B-mixed-3-4bit-3.85bpw-mlx](https://huggingface.co/destynova002/Qwen3.6-27B-mixed-3-4bit-3.85bpw-mlx)
- [Qwen3.6-35B-A3B-mixed-3-4bit-3.85bpw-mlx](https://huggingface.co/destynova002/Qwen3.6-35B-A3B-mixed-3-4bit-3.85bpw-mlx)

## Tests

```bash
python3 -I -B test_allocation.py
python3 -I -B test_quant.py
```

Les tests couvrent le classement, le budget, les couches épinglées et
l'invariant des trios MoE. Ils n'importent pas `mlx`, donc ils tournent sur
n'importe quelle machine.
