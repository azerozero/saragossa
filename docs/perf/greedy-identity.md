# Greedy spéculatif et AR : état de l'identité (4 nov. 2026)

En greedy, MTP, DFlash et lookup doivent produire le texte de l'AR. Pour cela,
une ligne vérifiée parmi R lignes doit donner exactement les bits du pas
mono-ligne. Ce n'est pas encore le cas. Le texte reste identique tant que la
dérive ne fait pas basculer une quasi-égalité top-1/top-2. Sur le bench
concurrents, cela arrive pour 1 requête sur 66 sur le 27B, et pour 2 prompts
sur 5 avec DFlash sur le 35B, chemin non exposé (voir [mtp-pairwise.md](mtp-pairwise.md),
« DFlash 2 sur le 35B-A3B »).

## Sources d'écart mesurées

Mesures : ligne r d'un lot de R lignes contre la même entrée en mono-ligne,
comparées en bits, sur des formes synthétiques du 27B et du 35B.

| Kernel en lignes | Égal au mono | Écart |
|---|---|---|
| normes RMS en lignes | tout R | — |
| matmul u8 gs64, dense f32 | tout R | — |
| matmul u4 gs64 (qmm2, qmv répliqué) | R ≤ 4 | — |
| matmul u4 gs64 skinny (défaut à partir de 5 lignes) | non | 2,4-3,3e-6 relatif |
| bloc linear-attn batché | non, dès R = 2 | 2e-7 (35B), 1e-6 (27B) |
| préfixe NAX de l'attention en lignes (≥ 16k, KV bf16) | non | Q en bf16 |
| queue MoE de vérification, queue dense | R ≤ 4 | skinny ensuite |

Cas réel, Qwen3.8-27B 4 bits, prompt prose de 32 768 tokens :
- L'AR choisit « The » avec une marge de 0,0089.
- Une trajectoire entièrement vérifiée en lignes (3 ou 9 lignes) choisit
  « It ».
- Le préfixe NAX déplace ce logit de 1e-3 à 3,5e-3 par vérification. Sans
  NAX, la trajectoire garde « The ».

## Conséquence

Dans les conditions mesurées, la promesse greedy est donc « identique à l'AR,
sauf sur une quasi-égalité ». Le plan de correction passe par trois étapes :
- des oracles en bits par kernel et sur trajectoire ;
- une bisection du bloc linéaire ;
- des mesures du coût d'un skinny exact et du retrait de NAX en greedy.

Cette page sera mise à jour avec ses résultats.
