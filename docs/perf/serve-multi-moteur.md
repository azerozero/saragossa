# serve multi-moteur (`--engines N`)

Un seul `saragossa serve --engines N` charge une fois le LLM, Whisper, le TTS
et la diarisation, puis fait tourner N moteurs en parallèle, chacun avec son KV,
son scratch Metal et son cache de préfixe par blocs.

## Mesure (M5 Max 128 Go, 25 nov. 2026)

Bench vocal : N chaînes diarisation → STT → LLM streamé → TTS streamé, 12 tours
par chaîne, `temperature 0`. Modèles : Qwen3.6-35B-A3B-4bit (MTP, poids mmap),
whisper-large-v3-turbo, Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit et
Nemotron-3-Diarization. Cache de préfixe coupé, porte de calme loadavg < 2,
secteur, verrou GPU. Empreinte = `footprint` du process serve (arrondie au Go) ;
mémoire système = anonyme + wired + compressé (`vm_stat`).

| N | empreinte à chaud / pic | mémoire système pic | first_sound p50/p95 | tts_rtf p50/p95 | decode tok/s par chaîne |
|---|---|---|---|---|---|
| 1 (avant) | 11 / 12 Go | 65,0 Go | 1,48 / 1,58 s | 0,218 / 0,241 | 124 |
| 1 | 11 / 12 Go | 65,0 Go | 1,45 / 1,61 s | 0,215 / 0,242 | 126 |
| 2 | 12 / 12 Go | 65,9 Go | 2,01 / 2,30 s | 0,326 / 0,372 | 87 |
| 3 | 12 / 13 Go | 67,2 Go | 2,77 / 3,13 s | 0,435 / 0,493 | 60 |
| 4 | 12 / 13 Go | 67,4 Go | 3,52 / 3,87 s | 0,587 / 0,642 | 46 |
| 6 | 13 / 15 Go | 68,8 Go | 5,26 / 5,57 s | 0,854 / 0,949 | 32 |

« Avant » : binaire de la PR #352 (bb61467, sans `--engines`) ; « 1 » à « 6 » :
binaire 47b417f de cette branche avec `--engines N`, même session, même
environnement (`SARAGOSSA_RUST_PREFIX_CACHE=0`, `SARAGOSSA_RUST_MMAP_WEIGHTS=1`,
sidecar MTP officiel), clips FLEURS fr 0, 1, 2, 4 répétés, graine TTS 7.
Chaque chaîne enregistre par tour la transcription, la réponse et le SHA-256
du PCM reçu. Identité : à chaque tour,
transcription, réponse et hash du PCM sont identiques entre N=1 et l'ancien
binaire, et entre chaque N et N=1. À N=6, un trou de lecture de 0,36 s apparaît
(p95 `tts_rtf` 0,95).

## Ancien schéma : un serve par chaîne

Même bench, un process par chaîne (24 nov., binaire antérieur à la PR #352,
mémoire système) : environ +16 Go par chaîne (62,8 → 78,9 → 94,7 Go, coupure à
110 Go dès N=4) ; first_sound p50 1,70 s à N=3 et 2,14 s à N=4.

## Lecture

- Mémoire : environ +0,5 Go par chaîne supplémentaire, contre ~16 Go.
- Latence : le multi-moteur est plus lent que l'ancien schéma au même N
  (2,77 s contre 1,70 s à N=3). Le débit decode cumulé ne passe que de 125 à
  ~190 tok/s : la file de commandes Metal du prefill, de Whisper, de la
  diarisation et du codec TTS est partagée entre moteurs (les arènes decode ont
  déjà la leur). L'ablation du 27 nov. (plus bas) montre plutôt un GPU
  partagé dans le temps, dominé par le TTS.
- Coupés dès N > 1 : cache de préfixe interne du décodeur, mémo de prefill TTS,
  lots ; `--prefix-cache-ssd-dir` refusé ; STT streamée sérialisée entre
  moteurs.

## Ablation à N=6 (27 nov. 2026)

Même machine, mêmes modèles et même environnement, binaire de `main` (862f917).
Le débit par chaîne du tableau ci-dessus est un decode actif : il ne mesure pas
le débit global. Ici, chaque chaîne fait le même travail fixe (8 tours, clips
FLEURS fr 0, 1, 2, 4 en rotation décalée d'un cran par chaîne, pour que les
arrivées ne soient pas synchrones). Le débit global vaut tokens LLM (ou secondes
audio) émis par toutes les chaînes, divisés par le temps mur entre le départ
commun et la fin du dernier tour. Le trou intra-requête est calculé par
morceau HTTP reçu, avec une lecture continue simulée.

Trois cellules :

- complet : diarisation, STT, LLM et TTS ;
- LLM seul : même transcription, sans TTS ;
- TTS seul : les mêmes clauses que le complet.

La porte loadavg < 2 et le secteur sont vérifiés à chaque passe. Le complet a
4 passes, les deux autres 3.

| Cellule | durée (s) | tokens LLM / s global | audio global (× temps réel) | decode actif p50 par chaîne | first_sound p50 |
|---|---|---|---|---|---|
| complet | 113,5 à 115,6 (moy. 114,3) | 22,7 à 23,2 | 7,81 à 7,95 | 39 à 44 tok/s | 3,86 à 3,94 s |
| LLM seul | 16,9 à 17,0 | 154,5 à 155,7 | — | 29,3 à 29,5 tok/s | — |
| TTS seul | 97,3 à 98,1 | — | 9,20 à 9,27 | — | 1,20 à 1,26 s |

- Les temps s'additionnent : LLM seul + TTS seul = 114,7 s, contre 114,3 s pour
  le complet. C'est compatible avec un partage du GPU dans le temps sans
  recouvrement utile, sans le démontrer. Sous cette hypothèse de sérialisation,
  le TTS prend environ 85 % du temps et le LLM 15 %, et un LLM en lignes parfait
  (s → ∞) plafonnerait le débit global à environ ×1,17.
- Aucun trou intra-requête sur 48 tours par passe. L'attente cumulée de lecture
  reste sous 1 s par passe complète.
- Identité : réponse et SHA-256 du PCM identiques par clip à la passe de
  référence, pour toutes les passes et cellules (le TTS seul reproduit le PCM
  du complet).

### Coût des matmuls u6 du TTS en lignes

Bench `tts_u6_rows_cost_bench` (`--ignored --release`) : poids synthétiques aux
formes du talker (28 couches, 1,15 Go) et du code predictor (5 couches,
64 Mo). Il mesure la durée GPU d'un command buffer, en médiane de 7, sur 3
exécutions.

| R flux | talker : qmv en lignes (défaut) | R × qmv mono | `QmmBatchedU6` | predictor : lignes | `QmmBatchedU6` |
|---|---|---|---|---|---|
| 1 | 2,63 à 2,71 ms | — | — | 0,196 à 0,201 ms | — |
| 2 | 3,00 ms (×1,1) | 4,39 ms | 16,4 ms | 0,22 ms (×1,1) | 1,06 ms |
| 4 | 5,39 ms (×2,0) | 7,87 ms | 18,4 ms | 0,42 ms (×2,1) | 1,32 ms |
| 6 | 7,40 à 7,45 ms (×2,8) | 11,33 ms | 22,0 ms | 0,55 ms (×2,8) | 1,62 ms |

- Le qmv en lignes (bit-exact contre le mono dès R ≥ 3) fait 6 flux en 2,8 ×
  T(1), soit s ≈ 2,1 contre 6 × T(1) ; contre les 6 qmv mono mesurés
  (11,33 ms), le gain vaut ×1,52. Le `QmmBatchedU6` est 3 fois
  plus lent que lui à R = 6 et ne convient pas.
- Borne Amdahl d'un TTS en lignes entre moteurs à N=6 : ×1,2 si seule la part
  matmul (~1/3 d'une trame mesurée à N=1) accélère, ×1,8 si toute la trame
  suit le facteur 2,1. Cette borne reste à confirmer par un prototype.

#### Référence révisée : R passes complètes sur R files (28 nov. 2026)

Le « R × qmv mono » ci-dessus relit chaque poids R fois de suite, donc chaud en
cache. Le serve fait R passes complètes, une par moteur, sur des files
distinctes et des sorties distinctes. Le bench mesure ce cas et le qmv en
lignes avec la même horloge (temps mur, du commit à la fin du dernier command
buffer), sur 3 exécutions (porte loadavg < 2, secteur) :

| R | talker : R files | talker : lignes | ratio | predictor : R files | predictor : lignes | ratio |
|---|---|---|---|---|---|---|
| 2 | 3,47 à 3,50 ms | 3,31 à 3,33 ms | 1,05 | 0,51 à 0,55 ms | 0,38 ms | 1,33 à 1,46 |
| 4 | 6,58 à 6,66 ms | 5,70 à 5,71 ms | 1,15 à 1,17 | 0,81 à 0,82 ms | 0,58 à 0,62 ms | 1,32 à 1,42 |
| 6 | 9,71 à 9,88 ms | 7,73 à 7,74 ms | 1,26 à 1,28 | 1,16 à 1,18 ms | 0,75 à 0,76 ms | 1,52 à 1,57 |

- Les files concurrentes se recouvrent déjà : 6 passes talker prennent
  9,7 à 9,9 ms de mur, contre 16,0 ms en séquence (durée GPU). Le gain des
  lignes sur le schéma actuel se réduit à ×1,27 (talker) et ×1,55 (predictor)
  à R = 6.
- Profil de trame à N=1, codec non recouvert (`SARAGOSSA_TTS_CODEC_OVERLAP=0`,
  `SARAGOSSA_TTS_INTERNAL_PROFILE=1`, `SARAGOSSA_RUST_DECODE_PROFILE=1`,
  CustomVoice échantillonné, 118 trames, 2 passes), durée GPU par trame :
  talker 3,37 à 3,40 ms, trame CP fusionnée 5,15 à 5,18 ms (891 dispatches),
  codec 2,10 ms. Le recouvrement par défaut mêle le GPU du codec aux autres
  phases. Il n'y a donc pas de mesure GPU propre à N=6, et aucune borne de bout
  en bout n'est retenue ici.
