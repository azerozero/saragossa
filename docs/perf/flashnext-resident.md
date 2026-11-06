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
(teacher forcing sur le texte du golden N2) contre la référence CPU f32 sans
cache. Les deux chaînes sont natives et l'écart relatif est mesuré par couche
et par token.

| tokens | écart max par couche | KL max | argmax ≠ | verdict |
|---|---|---|---|---|
| 64 | 1,4e-5 | 1,0e-8 | 0 | vert |
| 256 | 2,1e-1 (dès la couche 15) | 1,35e-1 | 1 (token 201) | **rouge** |
| 2051 | 1,25 (dès la couche 2) | 1,35e-1 | 26 | **rouge** |

### Diagnostic de la porte rouge

Sur 256 tokens, tout reste sous 1e-3 jusqu'au token 88. Le premier écart
apparaît au **token 89, couche 15** (5,5e-2). En ce point, la marge du
routeur CPU au seuil du top-10 vaut **2,1e-7** : c'est une quasi-égalité
d'experts, justifiable par la règle des 1e-5.

Une fois l'expert basculé, la sortie de ce token diffère de quelques
pourcents. L'écart passe aux tokens suivants par l'état récurrent DeltaNet et
le KV des couches suivantes : tous les tokens à partir de 90 dépassent 1e-3
dès les couches 16 et 17. Le token 110 rebascule (marge 1,7e-5, mesurée sur
une entrée déjà contaminée). La dérive s'amplifie ensuite en profondeur. Sur
2051 tokens, la première bascule survient dès la couche 2.

La porte « 1e-3 sur la chaîne native », sans dérogation, ne peut donc pas
tenir sur les vrais poids au-delà de quelques dizaines de tokens. C'est une
propriété de la chaîne et non une preuve de bug : une bascule quasi ex æquo
contamine tout le reste de la séquence. Rien ne prouve pour autant l'absence
de bug au-delà de 88 tokens.

**Prochaine mesure proposée** (à décider) : une passe **à routage imposé**.
Le MoE GPU reçoit les experts choisis par le CPU aux tokens où la marge est
≤ 1e-5, et les sélections QSA restent denses. Si le moteur est exact, la
chaîne doit rester au niveau de 1e-5 sur 2051 tokens. Il faut pour cela un
buffer d'indices imposé dans le chemin MoE résident. Variante : ablation
locale par couche, avec une entrée commune et un état CPU injecté.

## Mémoire

Pic `peak memory footprint` du binaire de test N2 sur 2051 tokens :
**83,1 Go**. Ce pic inclut les poids GPU (~73 Go), la référence CPU (une
couche déquantifiée à la fois) et ses résiduels conservés
(48 × 2051 × 10240 f32, soit 4 Go). Chargement GPU : 33 à 43 s.

## Débit

Le harnais `bench_resident_decode_after_1k_context` est prêt (contexte de
1024 tokens, 256 pas gloutons, médiane, p10/p90, temps CPU de préparation).
Il n'a pas encore été lancé et attend une machine libre.
