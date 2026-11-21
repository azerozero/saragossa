# Campagne quantification ≤ 4,2 bpw — Qwen3.8-27B (21 nov. 2026)

Plan v2 relu par Astra, critères d'arrêt écrits avant chaque mesure. Résultat
mesuré : **D (base 3 bits, 4,200 bpw) échoue au critère runtime** (ratio ≥ 0,97,
prefill ≤ 1,10 × u4ls). Il passe la gate kld moteur et l'identité greedy.
D a donc déclenché l'arrêt pré-enregistré de la campagne en E0.

Inférence, non démontrée : avec ce runtime, une base 4 bits coûte au moins
4,5 bpw (codes plus 0,5 bpw d'échelles et biais gs64). La base 3 bits passe
par qmv faute de GEMM de prefill u3. Un autre candidat ≤ 4,2 bpw buterait donc
probablement sur le même chemin. Seul D a été mesuré. Rien n'est à porter en
Rust côté allocation.

Données brutes (SHA-256 des prompts, pas leur texte ; chemins anonymisés) :
[`data/quant-campagne-42bpw/`](data/quant-campagne-42bpw/).

## Protocole figé

| Élément | Contenu |
|---|---|
| Sélection | `tools/quant/sets/selection.json`, 24 prompts (8 historiques de `chat_kl.py` + 16 rédigés), code 8 / prose 8 / multilingue 8, SHA-256 `acc78140…` |
| Validation aveugle | `tools/quant/sets/validation.json`, 16 prompts : no_robots test (11) et aya_dataset test (5 langues), règle déterministe `prompt_sets.py build-validation`, SHA-256 `6892204e…`. **Non ouverte** (E4 non atteint) |
| Disjonction | aucune fenêtre commune de 32 caractères alphanumériques normalisés (NFKC, casse) ; les prompts des deux ensembles sont refusés dans un corpus de calibration (`imatrix.reject_eval_text`) |
| Référence | transformers 5.22.2 BF16 sur MPS, greedy, 192 tokens par prompt, top-64 ; 4 608 positions, aucune sous le minimum de 128 ; vocabulaire complet pour 2 prompts |
| Données | [ensembles et références](data/quant-campagne-42bpw/sets.json), [KL par prompt et par modèle, D mesuré deux fois](data/quant-campagne-42bpw/kl_per_prompt.json), [comparaisons appariées](data/quant-campagne-42bpw/paired.json), [audit d'octets](data/quant-campagne-42bpw/audit_bytes.json) |
| Mesure | `chat_kl.py eval --json` par prompt : KL moyenne, p99, max, top-1, NLL ; `paired.py` : IC 95 % par bootstrap sur les prompts, refus si une catégorie régresse de plus de 10 % |
| Budget | `audit_bytes.py` : octets `language_model.*` sérialisés / 26 895 998 464, refus au-delà de 4,200 |

Tokenizer : le `tokenizer.json` de D et celui de `ai_model/Qwen3.8-27B-u4ls`
sont des rendus de transformers (regex Qwen2, SHA `0997f410…`). main les refuse
désormais (`tokenizer_sidecar.py`). Les KL n'en dépendent pas, car le candidat est
évalué sur les ids de la référence BF16. Le contrôle kld u4ls a été lancé sur le
tokenizer d'origine (`.bak`). `convert_ls.py` et `gptq.py` recopient maintenant le
tokenizer de la source BF16.

Limite connue : le corpus de l'imatrix existante (calib_sha256 `e1e20f0f…`) est
introuvable. Sa disjonction n'est vérifiée que contre les 8 prompts historiques.

## E0 — contrôles

| Modèle | bpw langage | KL sélection | même top-1 | PPL | p99 moyen par prompt |
|---|---|---|---|---|---|
| u4ls (référence) | 4,501 | 0,0330 | 93,73 % | 1,328 | 0,372 |
| D, imatrix 3/4/5/6/8, fit pondéré | 4,200 | 0,0559 | 91,45 % | 1,366 | 0,639 |
| mxfp4 (contrôle hors budget) | 4,251 | 0,0720 | 90,84 % | 1,380 | 0,716 |

- D − u4ls apparié : **+0,0229**, IC 95 % [0,0175 ; 0,0288]. Il régresse dans les trois catégories.
- Répétabilité : D et D en ordre inverse sont identiques au bit sur les 24 prompts.
- KL top-64 + reste contre vocabulaire complet : écart < 2 % (0,0632 contre 0,0633, 0,0951 contre 0,0966).
- Gate kld moteur, oracle `external_kld_qwen38` inchangé, référence mlx-lm de D capturée hors dépôt via `SARAGOSSA_GOLDEN_ROOT` : **réussie**, ratio KL 0,04, ratio p99 Δp 0,17, même top-1 99,61 %. Contrôle u4ls avec les fixtures du dépôt : ratio 0,03 et 0,18.
- Identité greedy AR = MTP = DFlash sur D : md5 identiques à 1 k et à 32 k.

Sorties des gates kld : [`gates.json`](data/quant-campagne-42bpw/gates.json).

Runtime, `bench/ab.py`, ABBA, 4 blocs (16 passes par mode et par contexte), 256 tokens,
même binaire (main 7cb2f3e). Ratio D / u4ls, rapports complets et md5 d'identité dans
[`runtime.json`](data/quant-campagne-42bpw/runtime.json) :

| Mode | decode 1 k | decode 32 k | débit prefill 1 k / 32 k |
|---|---|---|---|
| AR | 1,007 | 1,005 | 0,109 / 0,114 |
| MTP | 0,699 | 0,763 | 0,112 / 0,115 |
| DFlash | 0,700 | 0,769 | 0,113 / 0,115 |

Seuils ≥ 0,97 et prefill ≤ 1,10 × u4ls échoués par D : selon le critère
pré-enregistré, la base 3 bits est suspendue pour ce checkpoint, et E1 à E4
sont arrêtés (voir l'inférence en tête).

Cause probable, d'après la trace de dispatch
([`dispatch_trace_prefill512.json`](data/quant-campagne-42bpw/dispatch_trace_prefill512.json)) (`SARAGOSSA_RUST_TRACE_DISPATCH_PATH`, prefill
de 512 tokens) : les 247 modules u3 passent par
`affine_qmv_fast_aligned_u3_gs64_f32` à M = 32, 44 ou 480 (1 720 dispatchs), là
où u4 utilise `gemm_nax_coop_qb_tiled_u4`. Les u5/u6 passent par
`dequant_u*_to_bf16` puis `qmm_na`. La vérification MTP/DFlash (M > 1) emprunte
le même chemin qmv u3, d'où la perte de 23 à 30 %. Le decode AR (M = 1) est à
parité.

## Information hors décision

A1F0 = allocation de D remplie en fit LS uniforme (`convert_ls --uniform-fit`) :
KL 0,0691. Écart apparié avec D : +0,0132, IC 95 % [0,0091 ; 0,0173]. À
allocation égale, le fit pondéré par E[x²] vaut donc environ −0,013 de KL. Ce
n'est pas l'ablation 2×2 complète : les cellules à allocation poids n'ont pas
été converties.

## Suite

1. Prérequis noyau, hors de cette campagne : un prefill u3 GEMM (au minimum
   dequant u3 → bf16 + `qmm_na`, comme u5/u6) et un chemin de vérification
   u3 à M petit. Ensuite, refaire E0 sur D avec les mêmes critères. Les scripts
   et la référence de sélection sont réutilisables tels quels.
2. Pas de gagnant après E4, donc aucun portage Rust d'une stratégie
   d'allocation. Le seul acquis transférable à `feat/convert-natif` est
   l'intérêt mesuré du fit LS pondéré par imatrix à allocation fixe.
3. Conversions utilisées : 1 sur 9.
