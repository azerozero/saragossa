# Qwen3.8-Flash-Next : decode résident ≤ 2051 tokens (étape 4)

Chemin Metal « un token = un command buffer » de `qwen4_exp`
(`src/decode_resident/flash_next/`), en régime QSA dense. Tant que la
visibilité reste sous `budget + compress − 1` (2051 tokens, valeur calculée
depuis la config), l'indexeur retient tous les blocs. La session refuse
d'aller au-delà. Les primitives existantes sont réutilisées sans
l'orchestrateur des couches Qwen.

## Contrat de la session

- **Validation à l'ouverture**, avant toute allocation. La session refuse :
  - `norm_topk_prob = false` ;
  - la gate DeltaNet SiLU ;
  - un top-k au-delà de 16 ;
  - un DeltaNet autre que d128 / noyau 4 ;
  - une attention ou un budget QSA incohérents.
- **KV f32 épinglé**, indépendant de `SARAGOSSA_RUST_KV_BF16` et du KV
  quantifié.
- **PLE**. Le hachage se fait sur un clone du contexte n-gram. Les lectures
  `pread` parallèles sont terminées avant le commit. Le contexte n'est engagé
  qu'une fois le pas abouti.
- **Échec d'encodage.** Les longueurs KV et le pas PLE sont restaurés. Rien
  n'est soumis, donc les états GPU restent intacts.
- **Erreur d'exécution GPU.** La session est invalidée jusqu'au prochain
  `reset`.
- **`reset`.** La session est invalidée pendant la restauration et ne
  redevient utilisable qu'après un succès complet. Un échec en cours de route
  (testé) laisse la session refuser le pas suivant.
- **Encodeur.** `EncoderEndGuard` ferme l'encodeur sur tous les chemins.
- **Contenu du `reset`.** Restaure les états DeltaNet initiaux (instantanés
  pris à l'ouverture), vide le KV et remet le PLE à zéro.
- **Garde mémoire au pic de chaque étape.** Les étapes sont, dans l'ordre :
  mixer, lm_head, PLE, chaque couche, puis états et scratch. Chaque budget
  compte :
  - le résident : buffers GPU, plus les sources CPU gardées en vie ;
  - le transitoire : copies CPU des triplets affines, avec échelles et biais
    élargis en f32, et tampon d'empilement des experts.

  Le budget des états et du scratch compte les états DeltaNet deux fois
  (états vivants et instantanés du `reset`), ainsi que le scratch interne de
  l'exécuteur alloué au premier pas.

  Un test sur la fixture N1 vérifie qu'un plafond égal au résident final est
  refusé, et qu'un plafond égal au pic est accepté. Le test de reprise, sur
  les vrais poids, compare les allocations réelles de la session au budget :
  buffers propres, états, instantanés et KV, plus la croissance du device au
  premier pas. Mesure : 241,6 Mo alloués, dont 1,5 Mo au premier pas, pour un
  budget de 241,7 Mo à 16 tokens de capacité.

## Briques et oracles CPU (tests ordinaires)

« Têtes réelles » désigne les géométries de Flash-Next : DeltaNet 16 K / 48 V
en d128, attention 24/2 en d256, hc 4 × 2560 de rang 320, 512 experts.
Certaines projections sont réduites pour garder les tests rapides : hidden 64
pour le DeltaNet, 1024 pour le MoE.

| brique | oracle | écart |
|---|---|---|
| DeltaNet Flash-Next (l2 de q/k, ε non mis à l'échelle, gate σ), têtes réelles, hidden 64 | `naive_flash_next` : sorties et état récurrent | ≤ 1e-5 ; la variante Qwen s'écarte de plus de 1e-2 |
| qmv hc Down `[320, 10240]` u8, Up `[10240, 320]` u8 align64, down partagé 640, lm_head u6 `[248320, 2560]` | CPU déquantifié | ≤ 1e-5, route vérifiée |
| MoE 512 experts / top-10, intermédiaire 640, expert partagé, hidden 1024 | CPU (softmax f32, top-10 renormalisé) | ≤ 1e-5 |
| routage imposé (oracles) | même MoE | top-10 naturel imposé : sortie identique au bit près ; ensemble top-9 + un expert extérieur : CPU à 1e-5 |
| hyper-connections (mix, injection, combinaison, mixer final), dimensions réelles | `GatedResidual` | ≤ 1e-5 |
| PLE : hachage incrémental, octets E4M3 décodés sur GPU, anneau de 9 colonnes | `ple` sans cache sur 14 pas, puis reset | ≤ 1e-5 |
| attention QSA dense (q_proj entrelacé, RoPE 64, θ = 1e7, KV f32), dimensions réelles | `qsa` sans cache sur 40 pas | ≤ 1e-5 ; position hors régime dense refusée |

Les oracles DeltaNet et QSA acceptent `SARAGOSSA_FLASH_NEXT_STEPS=2051` :
sur 2051 positions à entrée commune, ils restent ≤ 1e-5.

## N2 résident sur les vrais poids (tests ignorés)

`n2_resident_decode_matches_cpu_reference` fait avancer le GPU token par
token, en teacher forcing sur les 2051 premiers tokens du texte du golden N2,
contre la référence CPU f32 sans cache. Il mesure :

- l'écart relatif par couche et par token ;
- la KL et l'argmax par token ;
- les experts routés par le GPU à chaque couche (relus en test seulement).

Les préfixes de 64, 256 et 2051 tokens viennent d'un même passage : tout est
causal. Toute valeur non finie fait échouer la porte. Un test unitaire le
vérifie avec un NaN sur un token tardif.

### Porte

Une seule quasi-égalité du routeur suffit à faire diverger la suite de la
séquence en chaîne native. La porte est donc double.

1. **Passe native, contrôlée jusqu'à la première bascule de routage.**
   - Pourquoi la porte s'arrête à cette bascule : sur un MoE, une
     quasi-égalité du routeur fait choisir un autre expert. L'écart passe
     ensuite par l'état récurrent et le KV à tous les tokens suivants. Au-delà
     de la première bascule, la chaîne native compare donc deux trajectoires
     différentes, et une porte sur toute la séquence y serait physiquement
     intenable. L'exactitude du reste de la séquence est portée par la passe
     imposée, gatée de bout en bout.
   - Sur toute la séquence native : sorties et métriques finies (NaN et
     infinis refusés).
   - Avant la bascule : écart ≤ 1e-3 et argmax identique.
   - La bascule tombe sur un point dont la **marge de probabilité** du
     routeur CPU, au seuil du top-10, vaut au plus 1e-6.
   - Le GPU n'y échange que la paire au seuil (k-ième ↔ (k+1)-ième expert du
     CPU).
   - Au-delà, la passe native est un témoin publié.
2. **Passe à routage imposé.** Aux points de marge de probabilité ≤ seuil, le
   MoE GPU reçoit les experts du CPU. Cela passe par `force_next_topk`,
   compilé et activé seulement en `cfg(test)`.
   - Le seuil retenu est le plus strict de {1e-7, 1e-6, 1e-5} qui garde la
     porte verte.
   - Conditions à 64, 256 et 2051 tokens :
     - écart ≤ 1e-3 par couche et par token ;
     - KL ≤ 1e-3 ;
     - argmax identique, sauf quasi-égalité (écart de probabilité ≤ 1e-5) ;
     - routage GPU identique au CPU sur tous les points non imposés.

Deux **mutations** imposent un mauvais expert, au point de plus grande marge
CPU, dans la passe imposée au seuil retenu. L'une tombe tôt (tokens 8 à 63),
l'autre tard (dans les 64 tokens qui suivent la première bascule native).
Chacune doit faire passer la porte au rouge.

### Résultats (2051 tokens, 98 448 points token × couche)

Points de marge de probabilité :
- ≤ 1e-7 : **26** ;
- ≤ 1e-6 : **293** ;
- ≤ 1e-5 : **2871**.

| passe | préfixe | écart max | KL max | argmax ≠ | tokens > 1e-3 | routage ≠ hors imposition |
|---|---|---|---|---|---|---|
| imposée ≤ 1e-7 | 64 | 1,4e-5 | 1,0e-8 | 0 | 0 | 0 |
| imposée ≤ 1e-7 | 256 | 2,1e-1 | 1,35e-1 | 1 | 167 | 650 : **rouge** |
| imposée ≤ 1e-6 | 64 | 1,4e-5 | 1,0e-8 | 0 | 0 | 0 |
| imposée ≤ 1e-6 | 256 | 7,0e-5 | 1,0e-8 | 0 | 0 | 0 |
| imposée ≤ 1e-6 | 2051 | 9,1e-4 | 1,0e-8 | 0 | 0 | 0 : **verte** |
| native (témoin) | 64 | 1,4e-5 | 1,0e-8 | 0 | 0 | — |
| native (témoin) | 256 | 2,1e-1 | 1,35e-1 | 1 | 167 | — |
| native (témoin) | 2051 | 1,25 | 1,35e-1 | 26 | 1962 | — |
| mutation précoce (token 38, couche 18, marge 8,6e-3) | 46 | 9,4e-2 | 4,2e-3 | — | — | rouge, comme attendu |
| mutation tardive (token 132, couche 22, marge 8,9e-3) | 140 | 1,48e-1 | 1,22e-1 | — | — | rouge, comme attendu |

- **Seuil retenu : 1e-6.** À 1e-7, le point du token 89 (marge 2,1e-7) n'est
  pas imposé. Il bascule, et la chaîne diverge comme en natif.
- **Passe native : contrôle vert.** Elle reste sous 1e-3, argmax identique,
  jusqu'à la première bascule, au **token 89, couche 15**. La marge y vaut
  **2,1e-7**, et l'échange est exactement la paire au seuil.
- **Au-delà de cette bascule, le témoin seulement.** On compte
  12 480 points de routage différents, dont 5998 limités à la paire au seuil.
  Leurs entrées sont déjà contaminées : l'écart passe par l'état récurrent
  DeltaNet et le KV des couches suivantes, puis s'amplifie en profondeur. Ce
  témoin ne prouve rien au-delà du contrôle ci-dessus. L'exactitude de cette
  zone repose sur la passe imposée.
- **Passe imposée ≤ 1e-6, écart par couche sur 2051 tokens.** Il reste vers
  1e-5 jusqu'à la couche 13. Il monte ensuite, avec des sauts aux couches QSA
  27 et 31, jusqu'à 9,1e-4 (couche 32), puis redescend à 3,5e-4 en sortie.
  Les oracles de brique à entrée commune restent ≤ 1e-5 sur 2051 positions,
  et le MoE et les hyper-connections ne dépendent pas de la position. La
  montée vient donc de l'amplification, le long de la chaîne, d'écarts
  d'arrondi de 1e-5. La KL reste de 1e-8 et l'argmax identique.

`resident_session_recovers_after_ple_and_encode_failures` injecte, au
token 6, une lecture n-gram en échec, puis un encodage en échec après la
couche 20. La session garde sa position. Elle redonne ensuite, au bit près,
les logits d'une session sans échec, puis rejoue le début de la séquence à
l'identique après `reset`. Une mutation qui supprime la restauration du KV
fait échouer ce test.

## Mémoire

Pic `peak memory footprint` du binaire de test N2 sur 2051 tokens :
**84,5 Go**, sur les passes native et imposées, les deux mutations et les
essais de seuil. Il
comprend :
- les poids GPU (~73 Go) ;
- la référence CPU, une couche déquantifiée à la fois ;
- les résiduels CPU conservés (48 × 2051 × 10240 f32, soit 4 Go).

Chargement GPU : 33 à 43 s.

## Débit

Le harnais `bench_resident_decode_after_1k_context` est prêt. Il fait
3 répétitions séparées par un `reset`, chacune avec un contexte de 1024
tokens puis 256 pas gloutons. Il publie :
- le débit mural `tokens / durée totale` par répétition ;
- la médiane, p10/p90 et le maximum par pas ;
- le temps CPU de préparation (embedding et PLE).

La porte des 45 tok/s porte sur la répétition la plus lente et fait échouer
le test en dessous. La mesure n'est pas encore lancée : elle attend une
machine libre.
