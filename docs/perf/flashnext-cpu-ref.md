# Qwen3.8-Flash-Next : loader q4e8 et référence CPU f32

Étape 1 du portage `qwen4_exp`. Le loader (`src/qwen_loader/qwen4_exp.rs`) lit
la sortie de `convert-qwen4exp`. La référence CPU f32 (`src/qwen4_exp/`)
transcrit couche par couche les modules transformers 5.20.2 (`sdpa`).

Il s'agit d'une **référence sans cache** : un seul passage sur toute la
séquence. Le decode incrémental, le prefill découpé et le chemin Metal viendront
avec le chemin résident.

## Ce que la référence implémente

- **Gated residual.** La norme est groupée par `hidden`, avec un gain `1 + w`.
  Les poids de mélange valent `a = σ(W_up·SiLU(W_down·n/hc))`, l'entrée de la
  branche est `x = mean_flux(a ⊙ n)`, et l'injection `2σ(W_inject·n/hc)`
  s'applique au résiduel non normalisé. Le mixer final ne fait pas d'injection,
  et aucune norme n'est ajoutée avant le lm_head.
- **Gated DeltaNet.** Cette variante est propre à Flash-Next :
  - normalisation l2 des têtes `x/√(Σx²+ε)`, au lieu de l'unité RMS des autres
    familles, qui calcule `x/√(Σx²+d·ε)` ;
  - gate de sortie `σ(z)` ;
  - `linear_attn.norm` à gain `w` direct.
- **QSA.**
  - Indexeur : requêtes normées par `q_layernorm` puis tournées par la RoPE ;
    clés brutes moyennées par blocs de 4, puis normées par `k_layernorm`, puis
    tournées par la RoPE au premier token du bloc.
  - Score `Σ_têtes ReLU(q·k)/√d`, top-`budget/4` blocs, plus la queue :
    `4·min(⌊V/4⌋, 512) + V mod 4` tokens.
  - Attention GQA avec un `q_proj` entrelacé par tête (query | gate), et
    `σ(gate)` appliquée avant `o_proj`.
  - mRoPE sur 64 dimensions, θ = 1e7, avec les mêmes positions sur les trois
    axes en texte seul.
- **PLE.**
  - Hash n-gram : XOR des produits `token × multiplicateur` en u64, modulo un
    nombre premier par tête, plus l'offset de la tête ; EOS en bord de segment.
  - Gather E4M3 multiplié par l'échelle per-tensor.
  - Gate `σ(sign(s)·√max(|s|, 1e-6))`.
  - Convolution depthwise de noyau 4 et de dilation 3.
  - Le résultat est ajouté aux flux avant l'hyper-connection d'attention de la
    couche d'indice 1.
- **MoE.** Softmax f32 sur tous les experts, top-k, renormalisation, puis expert
  partagé multiplié par `σ(gate)`. Les experts sont déquantifiés un par un et
  seulement s'ils sont routés.
- **Loader.**
  - Vérifie la config, en ne refusant que les invariants supposés par le code.
  - Vérifie l'inventaire exact des tenseurs, les triplets affines et leurs bits,
    les plages de chaque tenseur, la table n-gram et ses buffers de hachage
    (recalculés par SplitMix64 avec seed 1234).
  - Ne valide le MTP que sur ses métadonnées.
  - Les lectures sont bornées et la déquantification se fait à la demande.

## Oracles

Tolérances :
- par couche : norme relative `‖a−b‖/‖b‖` ≤ 1e-4, mesurée sur l'entrée torch de
  la couche précédente (ablation locale) ;
- logits : KL(oracle‖rust) ≤ 1e-3, en moyenne comme au maximum par token ;
- top-1 identique.

Quasi-égalités : un token dont le routeur a une marge inférieure à 1e-5 au seuil
du top-k, ou une requête QSA à scores ReLU nuls ex æquo au seuil, est listé à
part. Pour QSA, la sélection torch est alors imposée afin de comparer les
sorties. La dérive cumulée (couches enchaînées) est mesurée, sans servir de
porte.

### N1 : modèle synthétique, dans le dépôt, sans poids

La fixture `tests/fixtures/qwen4exp_mini` fait 987 288 octets. Elle contient un
petit modèle de 4 couches (3 DeltaNet, dont une avec PLE, puis 1 QSA), avec
hidden 64, hc 2, 4 experts top-2 et un budget QSA de 16 tokens par blocs de 4.
La séquence fait 40 tokens, avec un EOS en position 17.

Tous les poids sont aléatoires mais exactement représentables en q4e8 : la
conversion est sans perte, vérifiée sur ses 151 tenseurs. Le test convertit la
source, ouvre la sortie, exécute la référence et la compare aux sorties
officielles.

| mesure | valeur |
|---|---|
| couches 0 à 2 (locale / chaînée) | ≤ 5,1e-7 / ≤ 2,6e-7 |
| mixer final | 8,5e-8 (local), 2,2e-7 (chaîné) |
| logits | KL moyen 3,1e-15, max 1,8e-14, top-1 identique |
| routage | identique sur toutes les couches |
| QSA | 3 requêtes à ex æquo de ReLU nulles, toutes au seuil ; les autres sont identiques |

Régénération : `make_mini_fixture.py` (hors dépôt, sha256
`974eb8bf…603c`), avec la graine 20261109, torch 2.14.1 et transformers 5.20.2.
Les paramètres sont dans `RECIPE.json` et les sha256 sont figés dans le test.

### N2 : vrais poids q4e8 v2

Ces tests sont ignorés par défaut. Ils s'activent avec `SARAGOSSA_QWEN4EXP_DIR`
et `SARAGOSSA_QWEN4EXP_GOLDEN`.

Les goldens viennent de `oracle_dump.py` (hors dépôt, sha256 `e81c392f…ab84`) :
- couches officielles `Qwen4ExpTextDecoderLayer`, en f32, sdpa et mode eval ;
- poids q4e8 déquantifiés, experts placés dans le module officiel
  `Qwen4ExpTextExperts` (le MoE n'est pas réimplémenté) ;
- empreintes des poids copiées dans les métadonnées des goldens.

Pour les couches 0 à 3, une seule séquence de 8000 tokens suffit : comme tout
est causal, ses préfixes donnent les longueurs 64, 2047 à 2052 et 3000.

| couche | locale, hors quasi-égalités (tous préfixes) | chaînée (8000) | quasi-égalités |
|---|---|---|---|
| 0 (DeltaNet) | ≤ 1,7e-6 | 1,6e-6 | aucune |
| 1 (DeltaNet + PLE) | ≤ 7,6e-7 | 2,1e-6 | aucune |
| 2 (DeltaNet) | ≤ 7,1e-7 | 1,1e-4 | 1 bascule de routage dans le chemin chaîné (token 313) |
| 3 (QSA) | ≤ 5,0e-7 | 1,6e-4 | routage du token 159 (marge < 1e-5) ; 0 ex æquo QSA |

Frontières QSA : de V = 2047 à 2051, la sélection est dense et identique à
torch. À V = 2052, la sélection réelle retient 2048 tokens, identiques à torch.
La taille `4·min(⌊V/4⌋, 512) + V mod 4` est vérifiée pour chacune des 8000
requêtes.

Sans exclusion du token 159, l'erreur locale brute de la couche 3 monte à
1,5e-4 sur les préfixes de 2047 à 3000 tokens : c'est l'effet du seul token dont
le routage est quasi égal.

Modèle complet (48 couches, mixer, lm_head) sur 4 prompts de 14 à 32 tokens :
KL moyen au plus 6,8e-11, KL max au plus 5,3e-10, logits à une erreur relative
de 2,7e-6 à 5,8e-6, top-1 identique partout.

Empreintes des goldens N2 : `layers.safetensors` `0bea534b…e90f`,
`full.safetensors` `f030ea98…15ed`.

## Coût mesuré (M5 Max, CPU)

| passage | durée | pic (`peak memory footprint`) |
|---|---|---|
| modèle complet, 4 prompts courts | 280 s | 0,99 Go |
| couches 0 à 3 sur 8000 tokens, test compris | 378 s | 5,1 Go |

Le second pic inclut les goldens chargés et quatre résiduels `[8000, 10240]`
f32 (328 Mo chacun). Les activations de la référence croissent avec la longueur
de la séquence ; les poids, eux, restent bornés à une couche à la fois.
