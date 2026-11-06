# Qwen3.8-Flash-Next : decode résident ≤ 2051 tokens (étape 4)

Chemin Metal « un token = un command buffer » de `qwen4_exp`
(`src/decode_resident/flash_next/`), en régime QSA dense : jusqu'à
2051 tokens visibles, l'indexeur retient tous les blocs, et la session refuse
d'aller au-delà. Les primitives existantes sont réutilisées sans
l'orchestrateur des couches Qwen.

## Briques et oracles CPU (tests ordinaires, aux formes réelles)

| brique | oracle | écart |
|---|---|---|
| DeltaNet Flash-Next (l2 de q/k, ε non mis à l'échelle, gate σ), 16 K / 48 V, d128 | `naive_flash_next` : sorties et état récurrent | ≤ 1e-5 ; la variante Qwen s'écarte de plus de 1e-2 |
| qmv hc Down `[320, 10240]` u8, Up `[10240, 320]` u8 align64, down partagé 640, lm_head u6 `[248320, 2560]` | CPU déquantifié | ≤ 1e-5, route vérifiée |
| MoE 512 experts / top-10, intermédiaire 640, expert partagé | CPU (softmax f32, top-10 renormalisé) | ≤ 1e-5 ; mutation top-k − 1 détectée |
| hyper-connections (mix, injection, combinaison, mixer final) | `GatedResidual` | ≤ 1e-5 |
| PLE : hachage incrémental, lectures `pread` parallèles avant commit, décodage E4M3 sur GPU, anneau de 9 colonnes | `ple` sans cache sur 14 pas, puis reset | ≤ 1e-5 ; mutation de l'anneau détectée |
| attention QSA dense (q_proj entrelacé, RoPE 64, θ = 1e7, KV f32) | `qsa` sans cache sur 40 pas | ≤ 1e-5 ; position 2051 refusée |

## N2 résident sur les vrais poids (test ignoré)

`n2_resident_decode_matches_cpu_reference` : le GPU avance token par token
(teacher forcing sur les 2051 premiers tokens du texte du golden N2) contre la
référence CPU f32 sans cache. L'écart relatif est mesuré par couche et par
token, la KL et l'argmax par token. Les préfixes de 64, 256 et 2051 tokens sont
tirés d'un même passage : tout est causal.

### Redéfinition de la porte

Sur la chaîne native, une seule quasi-égalité du routeur suffit à faire
diverger la suite de la séquence (diagnostic ci-dessous). La porte porte donc
sur une **passe à routage imposé**, même logique que la justification par
l'oracle de l'étape 1 :

- aux points `(token, couche)` où la marge du routeur CPU au seuil du top-10
  vaut au plus 1e-5, le MoE GPU reçoit les experts choisis par le CPU ;
- l'imposition passe par `force_next_topk` (`cfg(test)` seulement, jamais en
  production) : un kernel de top-k imposé recalcule les poids softmax sur les
  experts imposés, comme le kernel normal ; la session vérifie que chaque
  imposition a été consommée ;
- le mécanisme est validé sur le MoE synthétique : le top-10 naturel imposé
  redonne la sortie au bit près, et un autre ensemble suit le CPU à 1e-5.

Porte, sur la passe imposée, à 64, 256 et 2051 tokens : écart ≤ 1e-3 par
couche et par token, KL max ≤ 1e-3, argmax identique sauf quasi-égalité
(écart de probabilité ≤ 1e-5). La chaîne native reste mesurée comme témoin,
sans porte dure. Un test de mutation impose un mauvais expert sur un token à
marge nette et exige que la porte passe au rouge.

### Résultats (2051 tokens, 2871 impositions sur 2051 × 48 points)

| passe | préfixe | écart max | KL max | argmax ≠ | tokens > 1e-3 |
|---|---|---|---|---|---|
| imposée | 64 | 1,4e-5 | 1,0e-8 | 0 | 0 |
| imposée | 256 | 7,0e-5 | 1,0e-8 | 0 | 0 |
| imposée | 2051 | 9,1e-4 | 1,0e-8 | 0 | 0 |
| native (témoin) | 64 | 1,4e-5 | 1,0e-8 | 0 | 0 |
| native (témoin) | 256 | 2,1e-1 | 1,35e-1 | 1 | 167 |
| native (témoin) | 2051 | 1,25 | 1,35e-1 | 26 | 1962 |
| mutation (token 38, couche 18, marge 8,6e-3) | 46 | 9,4e-2 | 2,15e-3 | — | rouge, comme attendu |

**Porte verte** sur la passe imposée. Écart max par couche sur 2051 tokens :
≈ 1e-5 jusqu'à la couche 13, puis une montée progressive avec des sauts aux
couches QSA 27 et 31, jusqu'à 9,1e-4 (couche 32), avant de redescendre à
3,5e-4 en sortie.

**Bisection de cette montée.** Les oracles de brique, à entrée commune,
tournent aussi sur 2051 positions (`SARAGOSSA_FLASH_NEXT_STEPS=2051`) :

- attention QSA dense : ≤ 1e-5 à chaque pas ;
- DeltaNet : sorties et état final ≤ 1e-5 ;
- MoE et hyper-connections ne dépendent pas de la position.

Aucune primitive ne dérive donc avec la longueur. Les 9e-4 viennent de
l'amplification, le long des 48 couches, d'écarts d'arrondi de 1e-5 (ordre
des sommes GPU et CPU), surtout dans l'attention à long contexte. La KL de
1e-8 et l'argmax identique sur 2051 tokens le confirment.

### Témoin natif : diagnostic

Tout reste sous 1e-3 jusqu'au token 88. Le premier écart apparaît au
**token 89, couche 15** (5,5e-2), où la marge du routeur CPU vaut **2,1e-7** :
un expert bascule sur une quasi-égalité. L'écart passe ensuite aux tokens
suivants par l'état récurrent DeltaNet et le KV des couches suivantes, puis
s'amplifie en profondeur. Le même phénomène existe entre deux implémentations
f32 quelconques ; il ne signale pas de défaut.

## Mémoire

Pic `peak memory footprint` du binaire de test N2 sur 2051 tokens :
**84,3 Go** (passes native, imposée et mutation). Ce pic inclut les poids GPU (~73 Go), la référence CPU (une
couche déquantifiée à la fois) et ses résiduels conservés
(48 × 2051 × 10240 f32, soit 4 Go). Chargement GPU : 33 à 43 s.

## Débit

Le harnais `bench_resident_decode_after_1k_context` est prêt (contexte de
1024 tokens, 256 pas gloutons, médiane, p10/p90, temps CPU de préparation).
Il n'a pas encore été lancé et attend une machine libre.
