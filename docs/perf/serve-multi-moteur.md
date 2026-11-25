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
  déjà la leur). Prochaine étape : une file par moteur.
- Coupés dès N > 1 : cache de préfixe interne du décodeur, mémo de prefill TTS,
  lots ; `--prefix-cache-ssd-dir` refusé ; STT streamée sérialisée entre
  moteurs.
