# Qwen3.8-Flash-Next : loader q4e8 et référence CPU f32

Étape 1 du portage `qwen4_exp`. Le loader (`src/qwen_loader/qwen4_exp.rs`) lit
la sortie de `convert-qwen4exp`. La référence CPU f32 (`src/qwen4_exp/`)
transcrit couche par couche les modules transformers 5.20.2 (`sdpa`).

Il s'agit d'une **référence sans cache** : un seul passage sur toute la
séquence. Le decode incrémental, le prefill découpé et le chemin Metal viendront
avec le chemin résident.

## Ce que la référence implémente

- **Gated residual.**
  - Norme groupée par `hidden`, avec un gain `1 + w`.
  - Poids de mélange `a = σ(W_up·SiLU(W_down·n/hc))` et entrée de branche
    `x = mean_flux(a ⊙ n)`.
  - Injection `2σ(W_inject·n/hc)` sur le résiduel non normalisé.
  - Le mixer final ne fait pas d'injection, et aucune norme n'est ajoutée avant
    le lm_head.
- **Gated DeltaNet.** Cette variante est propre à Flash-Next :
  - normalisation l2 des têtes `x/√(Σx²+ε)`, au lieu de l'unité RMS des autres
    familles, qui calcule `x/√(Σx²+d·ε)` ;
  - gate `σ(z)` ou `SiLU(z)` selon `output_gate_type` ;
  - `linear_attn.norm` à gain `w` direct.
- **QSA.**
  - Indexeur : requêtes normées par `q_layernorm` puis tournées par la RoPE ;
    clés brutes moyennées par blocs de `compress` tokens, puis normées par
    `k_layernorm`, puis tournées par la RoPE au premier token du bloc.
  - Score `Σ_têtes ReLU(q·k)/√d`, top-`budget/compress` blocs, plus la queue :
    `compress·min(⌊V/compress⌋, budget/compress) + V mod compress` tokens.
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
  - Config : invariants vérifiés dans l'ordre, dimensions bornées à 2^20,
    arithmétique vérifiée, `group_size` nul refusé.
  - Inventaire exact des tenseurs du tronc **et du sidecar MTP** : formes,
    dtypes, triplets affines et leurs bits, plages physiques.
  - Table n-gram, avec des buffers de hachage recalculés par SplitMix64 (seed
    1234).
  - Lectures bornées et déquantification à la demande.

## Oracles et quasi-égalités

Tolérances :
- par couche : norme relative `‖a−b‖/‖b‖` ≤ 1e-4, mesurée sur l'entrée torch de
  la couche précédente (ablation locale) ;
- logits : KL(oracle‖rust) ≤ 1e-3, en moyenne comme au maximum par token ;
- top-1 identique.

Les goldens figent les **scores de l'oracle** : logits du routeur et scores de
l'indexeur pour chaque requête.

Une quasi-égalité n'est admise que si l'oracle la justifie :
- **Expert échangé** : sa probabilité torch doit être au seuil torch du top-k, à
  1e-5 près. Les échanges sont journalisés.
- **Bloc QSA échangé** : même règle, avec les scores torch. De plus, les scores
  Rust de l'indexeur doivent égaler ceux de torch à 1e-4 près, sur toutes les
  requêtes.

Un test négatif vérifie qu'un indexeur dont les scores Rust sont tous nuls est
bien rejeté.

Deux passes sont mesurées :
- **Ablation** : la sélection torch est imposée à toute couche QSA. C'est elle
  qui sert de porte sur les sorties.
- **Passe chaînée native** : sélection propre de l'indexeur, sur l'entrée Rust
  enchaînée. Elle est mesurée, avec un garde-fou large de 1e-2 en norme
  relative.

### N1 : modèle synthétique, dans le dépôt, sans poids

La fixture `tests/fixtures/qwen4exp_mini` fait 825 490 octets. Elle contient un
modèle de 3 couches (DeltaNet, DeltaNet + PLE, QSA), avec hidden 64, hc 2,
4 experts top-2, un indexeur à 4 têtes non alignées et un budget QSA de 16
tokens par blocs de 4. La séquence fait 40 tokens, avec un EOS en position 17.

Tous les poids sont aléatoires mais exactement représentables en q4e8 : la
conversion est sans perte. Le test enchaîne conversion, loader et forward, puis
compare aux sorties officielles. La sortie de la dernière couche est capturée
par un hook.

| mesure | valeur |
|---|---|
| couches 0 à 2 (locale / chaînée imposée / chaînée native) | ≤ 4,4e-7 / ≤ 2,8e-7 / ≤ 2,8e-7 |
| mixer final | 2,1e-7 |
| logits | KL moyen 2,6e-15, max 1,2e-14, top-1 identique ; idem en passe native |
| routage | identique ; marge minimale de l'oracle au seuil : 4,2e-3 |
| QSA (indexeur aléatoire) | scores Rust égaux aux scores torch ; 0 ex æquo |
| golden QSA à marge nette (marge 2,8e-2) | sélection identique sans quasi-égalité, sortie 2,4e-7 |

Régénération : `tools/flashnext/make_mini_fixture.py`, graine 20261110,
torch 2.14.1 et transformers 5.20.2 (voir `tools/flashnext/README.md` et
`RECIPE.json`). Les sha256 sont figés dans le test, et la fixture est sous la
licence du dépôt.

### N2 : vrais poids q4e8 v2

Ces tests sont ignorés par défaut. Ils s'activent avec `SARAGOSSA_QWEN4EXP_DIR`
et `SARAGOSSA_QWEN4EXP_GOLDEN`.

Les goldens viennent de `tools/flashnext/oracle_dump.py` : couches officielles
`Qwen4ExpTextDecoderLayer`, experts placés dans le module officiel. Avant
l'inférence, les tests vérifient `provenance.json`, qui contient une entrée par
golden :
- l'empreinte du golden ;
- les empreintes de `config.json` et de chaque fichier de poids réellement
  ouvert (shards, MTP, table n-gram), **recalculées sur disque** ;
- la version de transformers, le backend sdpa et le dtype f32.

`oracle_dump.py` refuse de mélanger dans un même dossier des goldens issus
d'autres poids, d'une autre config ou d'un autre environnement. Un test Rust
modifie tour à tour un shard, la table, la config et le golden, et exige le
refus.

Pour les couches 0 à 3, une seule séquence de 8000 tokens suffit : comme tout
est causal, ses préfixes donnent les longueurs 64, 2047 à 2052 et 3000. La porte
locale est vérifiée sur chacun de ces préfixes.

| couche | locale (tous préfixes, hors échanges justifiés) | chaînée native (8000) | échanges et écarts natifs |
|---|---|---|---|
| 0 (DeltaNet) | ≤ 1,7e-6 | 1,6e-6 | aucun |
| 1 (DeltaNet + PLE) | ≤ 7,6e-7 | 2,1e-6 | aucun |
| 2 (DeltaNet) | ≤ 7,1e-7 | 1,1e-4 | passe native : 1 routage basculé |
| 3 (QSA) | ≤ 5,0e-7 | 1,6e-4 | token 159 : experts 79 ↔ 373, justifiés par l'oracle ; passe native : 2 routages et 36 sélections QSA ≠ torch |

Frontières QSA : de V = 2047 à 2051, la sélection est dense et identique à
torch. À V = 2052, la sélection réelle retient 2048 tokens, identiques à torch.
Sur les vrais poids, les scores Rust de l'indexeur sont égaux à ceux de torch,
sans aucun ex æquo.

Dans la passe native, la dérive d'entrée fait basculer quelques routages et
sélections, d'où une erreur relative de 1e-4 en enchaînant les couches. Avec la
sélection imposée, l'erreur locale de la couche 3 reste à 4,2e-7, hors token
159.

Modèle complet (48 couches, mixer, lm_head) sur 4 prompts de 14 à 32 tokens :
KL moyen au plus 6,8e-11, KL max au plus 5,3e-10, logits à une erreur relative
de 2,7e-6 à 5,8e-6, top-1 identique partout.

## Coût mesuré (M5 Max, CPU, binaire de test seul)

| passage | durée | pic (`peak memory footprint`) |
|---|---|---|
| modèle complet, 4 prompts courts | 286 s | 0,96 Go |
| couches 0 à 3 sur 8000 tokens, test compris | 394 s | 5,1 Go |

Le second pic inclut les goldens chargés et plusieurs résiduels `[8000, 10240]`
f32 (328 Mo chacun) conservés par le test. Il ne mesure donc pas isolément la
référence. Celle-ci garde les poids bornés à une couche à la fois, mais ses
activations croissent avec la longueur de la séquence.
