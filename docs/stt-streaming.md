# STT en flux dans `saragossa serve`

Trois usages partagent le même modèle Whisper, chargé avec `--stt-model`.

## Fichier complet : `POST /v1/audio/transcriptions`

Requête `multipart/form-data` compatible OpenAI : `file` (WAV), `language`
(`auto` par défaut), `response_format`, `stream`, `timestamp_granularities[]`.

| `response_format` | Décodage | Réponse |
|---|---|---|
| `json` (défaut) | `<\|notimestamps\|>` | `{"text": …}` |
| `text` | `<\|notimestamps\|>` | texte brut |
| `verbose_json` | tokens `<\|t.tt\|>` | `text`, `language`, `duration`, `segments[]` |

Chaque segment porte `id`, `seek`, `start`, `end`, `text`, `tokens`,
`temperature` (toujours 0), `avg_logprob`, `compression_ratio` et
`no_speech_prob`. Les règles de
timestamps sont celles d'OpenAI (premier timestamp ≤ 1 s, paires croissantes,
masse des timestamps contre le meilleur token texte). `avg_logprob`,
`compression_ratio` (zlib) et `no_speech_prob` sont calculés par fenêtre de
30 s, comme dans `whisper`.
En `auto`, `language` est le code ISO détecté après `<|startoftranscript|>`.

`timestamp_granularities[]=segment` exige `verbose_json`. `word` répond 501 :
l'alignement par mot demanderait un DTW sur les têtes de cross-attention.
Le mode horodaté décode une autre séquence de tokens : son texte peut
s'écarter légèrement de `json`.

L'audio long suit le découpage existant en fenêtres d'au plus 30 s, coupées au
minimum d'énergie entre 29 et 30 s.

### Sortie en flux : `stream=true`

Réponse `text/event-stream`, au format des modèles de transcription récents
d'OpenAI :

```text
data: {"type":"transcript.text.delta","delta":"Bonjour"}

data: {"type":"transcript.text.delta","delta":" à tous."}

data: {"type":"transcript.text.done","text":"Bonjour à tous."}
```

Un delta part dès qu'un token complète du texte visible. Les blancs finaux
et un caractère UTF-8 incomplet attendent le token suivant. La concaténation
des deltas égale `text`, qui égale la réponse `json`. `verbose_json` n'accepte
pas `stream`. Une erreur après les en-têtes produit un événement
`{"type":"error",…}` puis la fermeture.

## Entrée en flux : `POST /v1/audio/transcriptions/stream`

La route STT lit des corps à `Content-Length` avant le passage au thread moteur
séquentiel : elle ne lit pas de requête chunked et ne répond pas pendant
la réception (le débruitage DFN3 dispose de son propre transport duplex). L'entrée en flux passe donc par une session : une requête par
morceau audio, une réponse JSON par requête. Le client reste bloquant et
simple, comme le client HTTP de reti.

- Corps : PCM s16le mono 16 kHz brut (même format que la sortie de
  `/v1/audio/speech` avec `stream: true`). Un corps vide est permis.
- `X-Saragossa-STT-Session` : absent, le serveur ouvre une session ; sinon,
  identifiant renvoyé par la première réponse.
- `X-Saragossa-STT-Language` : langue, lue à l'ouverture (`auto` par défaut).
- `X-Saragossa-STT-Final: true` : dernier morceau, ferme la session.

```json
{"session":"stt-3","final":false,"committed":"Le romantisme présentait","partial":"un important","audio_seconds":2.5}
{"session":"stt-3","final":true,"committed":"…","partial":"","text":"…","audio_seconds":9.4}
```

### Stabilisation

Chaque morceau relance la transcription de la fenêtre courante (≤ 30 s). Le
modèle et ses poids GPU restent résidents ; l'encodeur repasse sur toute la
fenêtre, car Whisper n'a pas d'encodeur causal et complète toujours à 30 s :
le coût d'un morceau ne dépend presque pas de sa taille. Les mots communs
aux deux dernières hypothèses deviennent `committed` (LocalAgreement-2, comme
whisper_streaming). Une hypothèse qui contredit l'accord ne retire rien de
`committed`, et `partial` contient le reste de la dernière hypothèse.

La fenêtre se ferme dès que 30 s + 10 ms sont reçues : à ce moment, la coupe
batch au minimum d'énergie est connue sans audio ultérieur. Son texte est
transcrit une fois et remplace l'accord local de cette fenêtre dans
`committed`. De même, le texte final fait autorité sur la fenêtre ouverte :
`committed` est un aperçu stabilisé, pas une promesse. Le test de mesure
compte les mots stabilisés révisés ainsi. Le texte final est donc exactement celui du batch sur
le même PCM. Après le dernier morceau, seule la fenêtre ouverte reste à
décoder ; si aucun audio n'est arrivé depuis la dernière hypothèse, elle est
réutilisée sans nouveau calcul.

### Limites

Les refus d'admission répondent 400 avant toute inférence et laissent la
session intacte.

- Morceau d'au plus 30 s (960 000 octets), refusé avant copie.
- Au plus 30 min d'audio, 18 000 morceaux et 7 328 inférences par session.
- Un morceau vide, ou qui n'apporte pas 250 ms d'audio depuis la dernière
  hypothèse, renvoie le dernier état sans inférence.
- La fenêtre garde au plus deux fenêtres de capacité (60 s de f32, 3,8 Mo)
  après chaque fermeture ; le texte des fenêtres closes est borné par la
  durée maximale.
- Au plus 8 sessions ; ouvrir la neuvième évince la moins récemment utilisée.
  Une session inactive depuis 5 min est fermée.
- Une erreur d'inférence abandonne la session : le client en rouvre une.
  Une écriture de réponse échouée (client parti) la conserve, déjà avancée.
- Les identifiants sont séquentiels et protégés seulement par le bearer.
- L'identité avec le batch vaut pour le même PCM : un client qui quantifie en
  s16 doit comparer au batch du WAV s16, comme le fait reti.
- Le client règle la cadence : un envoi toutes les 0,5 à 1 s convient.

## Tests et mesures

Les tests CPU couvrent les règles de timestamps, les segments, les deltas,
LocalAgreement, les coupes de fenêtre et le format SSE. Sur poids réels :

```text
SARAGOSSA_WHISPER_DIR=<snapshot> SARAGOSSA_FLEURS_DIR=<fleurs-wer-v1> \
  cargo test --release --test stt_streaming -- --ignored --nocapture --test-threads=1
```

`fleurs_stream_latency_and_wer_report` simule un client temps réel (horloge
audio virtuelle, traitements mesurés) et affiche les latences et le WER.
