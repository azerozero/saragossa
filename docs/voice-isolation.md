# Isolation et nettoyage de la voix

Ce document couvre trois modèles évalués en novembre 2026 :

| Modèle | Rôle | État dans saragossa |
|---|---|---|
| DeepFilterNet3 | débruitage temps réel, 48 kHz | porté en Rust (CPU), oracle PyTorch, CLI `saragossa enhance`, `POST /v1/audio/enhance` |
| MossFormer2_SS_16K | séparation de deux locuteurs, 16 kHz | spécification et mesures sur la référence PyTorch (CPU et MPS) ; pas encore porté |
| NVIDIA-NemotronLabs-RE-USE | restauration universelle, hors ligne | spécification et estimation de coût ; pas porté |

Les mesures ont été faites sur un M5 Max chargé par d'autres travaux (charge
moyenne 35 à 55) : les médianes sont fiables, les queues (p99) sont
pessimistes.

## Licences

- **DeepFilterNet3** : code et poids `Rikorose/DeepFilterNet`, double licence
  MIT ou Apache-2.0 au choix. Le port Rust de `src/denoise/` suit
  `libDF/src/lib.rs` et `df/deepfilternet3.py`. Redistribuer les poids exige de
  joindre l'une des deux licences et la notice de copyright
  (« Copyright (c) 2021 Hendrik Schröter »).
- **MossFormer2_SS_16K** : poids `alibabasglab/MossFormer2_SS_16K` et code
  ClearerVoice-Studio sous Apache-2.0. Une redistribution doit garder la
  licence et signaler les modifications. Le modèle dépend aussi de
  `rotary-embedding-torch` (MIT) pour les positions rotatives.
- **RE-USE** : `nvidia/NVIDIA-NemotronLabs-RE-USE` sous OpenMDW-1.1, usage
  commercial permis. Garder la licence et les notices d'origine à la
  redistribution ; la licence tombe en cas d'action en contrefaçon contre le
  modèle. Ne pas confondre avec `nvidia/RE-USE` et `nvidia/Real-time_RE-USE`
  (NSCLv1, non commerciaux).

## DeepFilterNet3

### Spécification

Tous les paramètres viennent de `config.ini` du checkpoint officiel
(`model_120.ckpt.best`, 2 167 954 paramètres flottants).

**Signal et STFT** (`libDF`) :

- 48 kHz mono, FFT et fenêtre de 960 échantillons (20 ms), pas de 480 (10 ms) ;
- fenêtre de Vorbis `w[n] = sin(π/2 · sin²(π(n + ½)/960))`, dont la somme des
  carrés vaut 1 à 50 % de recouvrement ;
- STFT « temps réel » : la trame `t` concatène les 480 échantillons du hop
  précédent et ceux du hop `t`, sans centrage. L'analyse multiplie par
  `2H/N² = 1/960` ; la synthèse est une iFFT non normalisée, fenêtrée puis
  sommée par overlap-add ;
- 481 bins.

**Features d'entrée**, avec `α = round(exp(−H/sr/τ), 3) = 0,99` (τ = 1 s) :

- 32 bandes ERB (largeurs `libDF::erb_fb`, au moins deux bins par bande,
  vérifiées contre la banque `erb_fb` du checkpoint). Pour chaque bande :
  puissance moyenne, `10·log10(p + 1e-10)`, puis normalisation par moyenne
  exponentielle `s ← α·s + (1 − α)·x`, `(x − s)/40`, état initial
  `linspace(−60, −90)` ;
- 96 premiers bins complexes, normalisés par `x / sqrt(s)` avec
  `s ← α·s + (1 − α)·|x|`, état initial `linspace(0,001, 0,0001)`.

**Réseau** (activations `[canal][fréquence]` par trame, toutes les
convolutions causales en temps, BatchNorm repliée en évaluation, `eps = 1e-5`) :

| Bloc | Détail |
|---|---|
| `pad_feat` | les features entrent avec deux trames d'avance : le pas réseau τ lit `feat[τ + 2]` |
| `erb_conv0` | conv 3×3 (temps × fréquence), 1 → 64, BN, ReLU |
| `erb_conv1..3` | conv en profondeur 1×3 (stride fréquentiel 2, 2, 1) + 1×1, BN, ReLU : 32 → 16 → 8 → 8 bins |
| `df_conv0` | conv 3×3 en 2 groupes, 2 → 64 + 1×1, BN, ReLU, sur 96 bins |
| `df_conv1` | 1×3 en profondeur stride 2 + 1×1 : 96 → 48 bins |
| `df_fc_emb` | linéaire groupée (32 groupes) 3 072 → 512, ReLU, ajoutée à `e3` aplati `[F][C]` |
| `enc.emb_gru` | linéaire groupée 512 → 256 (16 groupes, ReLU), GRU 256, linéaire 256 → 512 (ReLU) |
| `lsnr_fc` | `sigmoid(W·emb) · 50 − 15` (SNR local, non utilisé pour la sortie) |
| `erb_dec` | GRU 2 couches (même enveloppe), puis U-Net : `convt3(conv3p(e3) + emb)`, `convt2(conv2p(e2) + ·)`, `convt1(conv1p(e1) + ·)`, `conv0_out(conv0p(e0) + ·)` ; `convt1/2` sont des convolutions transposées en profondeur 1×3 stride 2 (padding 1, output_padding 1) + 1×1 ; `conv0_out` finit par une sigmoïde |
| masque | 32 gains ERB étendus à 481 bins par la matrice 0/1 `erb_inv_fb` |
| `df_dec` | linéaire groupée 512 → 256 (8 groupes, ReLU), GRU 2 couches, plus `df_skip` (512 → 256) ; `tanh(df_out)` donne 96 × 10 valeurs, auxquelles s'ajoute `df_convp` (conv 5×1 en 2 groupes sur `c0`, + 1×1, BN, ReLU) |
| deep filtering | pour les 96 premiers bins : `Y[τ, f] = Σₙ X[τ − 2 + n, f] · C[n, τ, f]`, n = 0..4, produit complexe sans conjugaison |
| sortie | bins 96 à 480 : `X[τ, f] · gain[f]` ; bins 0 à 95 : sortie du filtre profond |

Le post-filtre (`mask_pf`) est désactivé dans cette configuration.

**Latence algorithmique.** Un hop d'entrée `t` produit la trame de sortie
`τ = t − 2`, dont l'overlap-add rend les échantillons du hop `τ − 1`. Le retard
entrée → sortie vaut donc `N − H + 2H = 1 440` échantillons, soit **30 ms**,
plus l'attente du hop complet (jusqu'à 10 ms) : 40 ms au total, comme annoncé
par l'article. Le temps de calcul s'ajoute à ces 40 ms.

### Implémentation

`src/denoise/` :

- `dsp.rs` : fenêtre, largeurs ERB, STFT et iSTFT incrémentales (rustfft),
  normalisations ;
- `layers.rs` : conv 2D causale par trame (groupée, stride fréquentiel,
  transposée), linéaire groupée, GRU PyTorch, BatchNorm repliée ;
- `network.rs` : graphe DFN3 ;
- `mod.rs` : `DeepFilterNet3::load`, `enhance` (hors ligne, sans retard,
  équivalent de `df.enhance(pad=True)`) et `DfStream::process_hop` (temps réel,
  un hop de 10 ms à la fois) ;
- `stream.rs` : `DenoiseStream::new(&model)`, `push(&[f32])`, `finish()` ;
- `wav.rs` et `resample.rs` : WAV et filtre polyphase incrémentaux.

Le chargeur vérifie la forme de chaque poids, refuse un tenseur inconnu et
contrôle que la banque ERB du checkpoint correspond aux largeurs de libDF.

Conversion des poids (lecture `weights_only=True`, aucun code du checkpoint
n'est exécuté) :

```bash
python3 -I -B tools/voice/convert_dfn3.py \
  DeepFilterNet3/checkpoints/model_120.ckpt.best dfn3.safetensors
saragossa enhance --model dfn3.safetensors --input bruit.wav --output propre.wav
```

### Débruitage continu

L'entrée est moyennée en mono puis rééchantillonnée à 48 kHz, avec le même
filtre FIR polyphase que le batch. `DenoiseStream` conserve les états GRU,
STFT, la phase et le hop partiel entre appels. `push` rend la sortie disponible ;
`finish` compense le retard et rend exactement la fin du signal. Un second
`finish` est vide ; `push` après fermeture renvoie une erreur. Le retard DFN3
est de 30 ms plus le tamponnage d'un hop (jusqu'à 40 ms), auquel s'ajoute
l'anticipation du FIR si la fréquence change.

La limite de cinq minutes est supprimée. Les tampons internes ne dépendent
pas de la durée ; le vecteur rendu par `push` dépend seulement du morceau fourni.
`enhance` reste une commodité batch qui accumule sa sortie. Les poids sont
partagés et chaque flux garde son propre état ; aucun découpage multi-cœur
d'un même flux n'est effectué.

La CLI lit 480 trames à la fois et écrit un WAV mono float32 48 kHz. `--input -`
et `--output -` utilisent stdin/stdout. Sur une sortie seekable, elle réécrit
les tailles à la fin et passe en RF64/ds64 au-delà de 4 Gio. Sur un pipe,
les tailles RIFF/data valent `0xffffffff` (longueur inconnue, fin à EOF).
L'entrée accepte ces flux, les fichiers RIFF/RF64 et le PCM 16/24/32 ou float32 ;
les fréquences doivent rester entre 8 et 192 kHz.

`saragossa serve --enhance-model dfn3.safetensors` expose
`POST /v1/audio/enhance` en duplex :

- corps WAV brut (`Content-Type: audio/wav`) ou multipart avec un champ `file` ;
  les autres champs et les en-têtes multipart sont bornés ensemble à 64 Kio ;
- cadrage `Content-Length` sans plafond de durée, ou `Transfer-Encoding: chunked` ;
  leur combinaison, les doublons et les encodages ambigus sont refusés avec 400 ;
- réponse WAV mono PCM 16 bits 48 kHz en HTTP chunked,
  avec longueurs WAV inconnues. Le premier chunk contient l'en-tête WAV ; les
  suivants contiennent les échantillons disponibles, avant la fin de l'envoi ;
- en-tête WAV validé avant chargement des poids et avant le 200 : une fréquence
  invalide renvoie 400. Une erreur après le 200 ferme la connexion sans chunk
  terminal, pour rendre la troncature détectable ;
- `SARAGOSSA_ENHANCE_IDLE_TIMEOUT_SECS` : inactivité maximale de chaque lecture
  ou écriture socket, défaut 30 s ;
- `SARAGOSSA_ENHANCE_MAX_STREAMS` : flux simultanés, défaut 2 ; le suivant reçoit
  503 avant lecture du corps. La réservation est rendue à la fin ou sur erreur.

Ces réglages sont centralisés dans `runtime_flags.rs`. Le plafond global
`SARAGOSSA_SERVE_MAX_CONNECTIONS` s'applique aussi. L'authentification et la
réservation RAII réutilisent la façade du serveur. Comme les sessions STT
([documentation](stt-streaming.md)), l'admission précède la modification d'état
et une erreur abandonne le flux. Les validateurs WAV sont partagés avec le STT.
Le protocole STT par requêtes successives ne permet pas le duplex : DFN3 utilise
un état lié à la connexion, sans identifiant ni éviction d'un flux actif.

Il n'y a aucune file audio intermédiaire : chaque écriture doit réussir avant
la lecture suivante. Les tampons de transport sont bornés (8 Kio par lecteur,
préfixe HTTP d'au plus 1 Kio, fenêtre multipart d'au plus 74 octets), le WAV
produit au plus 480 échantillons par lecture et le FIR conserve seulement son
historique. Un client duplex doit lire la réponse pendant qu'il envoie ; un
proxy doit désactiver le buffering de requête et de réponse pour préserver ce
comportement. Les autres routes HTTP gardent leur cadrage existant.

```bash
curl --no-buffer -H 'Content-Type: audio/wav' \
  -H "Authorization: Bearer $SARAGOSSA_API_KEY" \
  -H 'Transfer-Encoding: chunked' --data-binary @bruit.wav \
  http://localhost:8080/v1/audio/enhance > propre.wav
```

Les tests ordinaires, sans poids téléchargés, comparent au bit le modèle
synthétique avec le batch historique (morceaux de 1, 7, 480, 4 800 et tailles
pseudo-aléatoires), y compris les hops partiels. Le FIR a son oracle batch.
Un test d'une heure synthétique utilise l'adaptateur de production avec un
noyau de hop substitué : il contrôle le tampon fixe et la dette de sortie,
sans prétendre mesurer une heure d'inférence DFN3. Les tests TCP vérifient
l'émission avant la fin de l'envoi, les deux cadrages HTTP, multipart, le 400,
le 503, la contre-pression et l'inactivité en lecture/écriture.

### WebSocket

La feature `ws` (activée par défaut, impliquant `serve` et `denoise`) expose
`GET /v1/audio/enhance/ws?rate=16000&format=s16` sur le même listener.
Le handshake RFC 6455 exige HTTP/1.1, version 13 et une clé de 16 octets.
Les messages binaires portent du PCM mono little-endian, `s16` ou `f32` ; la
réponse utilise le même format à **48 kHz**. `rate` est obligatoire (8–192 kHz).
Les échantillons peuvent traverser les limites de messages. Les NaN, infinis
et échantillons tronqués sont refusés. Les trames **et messages reconstitués**
sont limités à 16 Kio ; le dépassement ferme avec 1009. Les erreurs de cadrage
ferment avec 1002. Les masques client, continuations, ping/pong et close sont
contrôlés ; aucune extension de compression n'est négociée.

Un **message binaire vide** termine l'entrée : le serveur vide FIR et DFN3,
renvoie la fin audio, un message binaire vide puis close 1000. Un close client
abandonne le traitement et reçoit son écho ; il ne remplace pas la fin d'entrée.
Lire pendant l'envoi est indispensable : le serveur écrit avant de relire,
sans file audio. Les limites de flux (503 avant upgrade), délais d'inactivité
et réservations sont partagés avec le chunked.

Le navigateur ne permet pas d'ajouter `Authorization` au handshake WebSocket.
Le serveur accepte donc aussi `bearer.<clé>` dans les sous-protocoles, et
renvoie le sous-protocole choisi (`saragossa` s'il est proposé, sinon la valeur
`bearer.<clé>`, que le navigateur exige). Un `Origin` non loopback (hors
`http(s)://127.0.0.1`, `localhost`, `[::1]`, tous ports) reçoit 403 avant
l'upgrade, avec ou sans clé. Chaque trame doit arriver en entier avant le délai
d'inactivité, et plus de 32 pings/pongs sans audio ferment la session (1008).
Une erreur PCM après l'upgrade ferme avec 1007 ou 1011. La clé doit être un token
HTTP valide (par exemple une chaîne hexadécimale). Hors boucle locale, placer
le listener derrière un proxy TLS (`wss://`) qui conserve l'upgrade.

```js
// pcmChunks produit des Uint8Array de PCM s16le mono, <= 16 Kio.
const ws = new WebSocket(
  "ws://localhost:8080/v1/audio/enhance/ws?rate=16000&format=s16",
  ["saragossa", `bearer.${apiKey}`],
);
ws.binaryType = "arraybuffer";
ws.onmessage = ({data}) => {
  if (data.byteLength) consumePcm48k(new Uint8Array(data));
};
await new Promise((resolve, reject) => {
  ws.onopen = resolve;
  ws.onerror = reject;
});
for await (const chunk of pcmChunks) {
  while (ws.bufferedAmount > 16384) {
    if (ws.readyState !== WebSocket.OPEN) throw new Error("flux fermé");
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  ws.send(chunk);
}
ws.send(new Uint8Array()); // vider le retard avant close
```

### HTTP/2 h2c

`--features h2` active les dépendances async optionnelles (hors `default`).
Le listener dédié est explicitement activé par
`SARAGOSSA_ENHANCE_H2_ADDR=127.0.0.1:8082` ; le listener HTTP/1 reste disponible.
Le démarrage échoue si l'adresse n'est pas loopback (127.0.0.0/8 ou `[::1]`),
ne peut pas être liée ou si aucune clé API
n'est configurée. Le listener accepte le **prior knowledge h2c**, sans upgrade
HTTP/1 ni TLS intégré. Pour un navigateur, un proxy TLS avec ALPN `h2` doit
terminer HTTPS et transmettre vers ce listener.

```bash
SARAGOSSA_ENHANCE_H2_ADDR=127.0.0.1:8082 \
  saragossa serve --port 8080 --enhance-model dfn3.safetensors
curl --http2-prior-knowledge --no-buffer \
  -H "Authorization: Bearer $SARAGOSSA_API_KEY" \
  --data-binary @mono.s16le \
  'http://localhost:8082/v1/audio/enhance/h2?rate=16000&format=s16' > propre.s16le
```

La requête est `POST`, le corps et la réponse sont du PCM brut comme en WS,
sans préfixe WAV ; la sortie est à 48 kHz. END_STREAM termine l'entrée et vide
le retard. L'admission partage `SARAGOSSA_ENHANCE_MAX_STREAMS` avec les autres
transports et répond 503 avant de consommer le corps. La taille de DATA et la
fenêtre par flux sont de 16 Kio, la fenêtre de connexion de 32 Kio ; une trame
supérieure au maximum négocié produit FRAME_SIZE_ERROR. La capacité reçue
n'est rendue qu'après l'émission de la sortie, et l'envoi attend WINDOW_UPDATE.
Un worker CPU par flux admis communique par deux canaux de capacité un : les
E/S des autres flux continuent pendant le calcul DFN3. Lectures, écritures,
handshake et attente de capacité sont bornés par le délai enhance ; une erreur
après les headers réinitialise le flux, sans toucher aux autres requêtes.

**Limite de [Fetch](https://fetch.spec.whatwg.org/#dom-request-duplex).** Le duplex HTTP/2 est testé avec un client natif qui reçoit
avant END_STREAM. L'API Fetch des navigateurs exige généralement
`duplex: "half"` pour un corps ReadableStream et n'expose la réponse qu'après
la fin de l'envoi : HTTP/2 ne lève pas cette restriction. Pour recevoir le son
pendant une capture navigateur continue, utiliser WS ou WebTransport. Un
exemple Fetch ne constitue donc pas une preuve de duplex navigateur.

### WebTransport / HTTP/3

La feature `webtransport` est opt-in et hors `default`. Elle utilise
[web-transport-quinn](https://github.com/moq-dev/web-transport), une
implémentation HTTP/3 WebTransport sur Quinn, et Rustls/ring. Le certificat
est généré avec rcgen. Ces dépendances optionnelles récentes nécessitent
Rust **1.88 ou supérieur** ; les features par défaut conservent leur périmètre.

```bash
SARAGOSSA_ENHANCE_WEBTRANSPORT_ADDR=127.0.0.1:4433 \
  saragossa serve --port 8080 --enhance-model dfn3.safetensors
```

Le listener UDP accepte une session par connexion QUIC sur
`/v1/audio/enhance/wt?rate=16000&format=s16&token=…`. La clé API est transmise
uniquement dans la requête CONNECT chiffrée ; l'application ne journalise pas
l'URL. Ne pas journaliser cette query dans un proxy. La même admission DFN3
renvoie HTTP **503 avant acceptation** quand le plafond commun est atteint.
La limite globale de connexions s'applique avant le handshake QUIC (refus
QUIC à ce stade, puisqu'aucune requête HTTP n'existe encore).

Chaque session accepte un stream bidirectionnel fiable. Chaque message
applicatif est `u32le longueur + PCM`, limité à **16 Kio** ; aucune allocation
proportionnelle à une longueur non validée. Un préfixe zéro suivi de FIN
termine l'entrée. Le serveur renvoie le PCM à 48 kHz, puis un préfixe zéro et
FIN. Les datagrammes ne sont pas utilisés. Les fenêtres QUIC sont bornées
(16 Kio par stream, 64 Kio par connexion et en émission), et chaque réponse
est écrite avant de lire la suivante. La contre-pression et les délais sont
partagés avec le worker HTTP/2. Un message trop grand, une fin partielle ou
une inactivité ferme le flux avec une erreur, sans marqueur de succès.

Le certificat auto-signé **ECDSA P-256** vit 13 jours plus une marge de
60 secondes au début (donc moins de 14 jours), reste en mémoire et est
renouvelé tous les 12 jours. Aucun fichier de clé n'est écrit. L'endpoint
HTTP authentifié `GET /v1/audio/enhance/certificate` expose l'URL, l'expiration
et `serverCertificateHashes` (SHA-256 du DER, tableau de 32 octets), avec
`Cache-Control: no-store`. Récupérer ces informations avant chaque connexion.

```js
const metadata = await fetch("/v1/audio/enhance/certificate", {
  headers: {Authorization: `Bearer ${apiKey}`},
}).then(r => {
  if (!r.ok) throw new Error(`certificat: ${r.status}`);
  return r.json();
});
const url = new URL(metadata.url);
url.search = new URLSearchParams({rate: "16000", format: "s16", token: apiKey});
const wt = new WebTransport(url, {
  serverCertificateHashes: metadata.serverCertificateHashes.map(h => ({
    algorithm: h.algorithm, value: Uint8Array.from(h.value),
  })),
});
await wt.ready;
const {readable, writable} = await wt.createBidirectionalStream();
const writer = writable.getWriter();
function packet(bytes) {
  if (bytes.length > 16384) throw new Error("morceau PCM trop grand");
  const out = new Uint8Array(4 + bytes.length);
  new DataView(out.buffer).setUint32(0, bytes.length, true);
  out.set(bytes, 4);
  return out;
}
// Les lectures QUIC peuvent couper un préfixe ou regrouper plusieurs messages.
const receiver = (async () => {
  const reader = readable.getReader();
  let pending = new Uint8Array();
  let finished = false;
  for (;;) {
    const {value, done} = await reader.read();
    if (done) {
      if (!finished || pending.length) throw new Error("réponse tronquée");
      return;
    }
    const joined = new Uint8Array(pending.length + value.length);
    joined.set(pending); joined.set(value, pending.length); pending = joined;
    while (pending.length >= 4) {
      if (finished) throw new Error("données après fin");
      const n = new DataView(pending.buffer, pending.byteOffset).getUint32(0, true);
      if (n > 16384) throw new Error("trame réponse trop grande");
      if (pending.length < 4 + n) break;
      const pcm = pending.slice(4, 4 + n);
      pending = pending.slice(4 + n);
      if (n === 0) finished = true;
      else consumePcm48k(pcm);
    }
  }
})();
const sender = (async () => {
  for await (const pcm of pcmChunks) await writer.write(packet(pcm));
  await writer.write(packet(new Uint8Array()));
  await writer.close();
})();
try { await Promise.all([sender, receiver]); }
finally { wt.close(); }
```

L'exemple exige un contexte navigateur sécurisé et une implémentation de
WebTransport avec épinglage de certificat. `consumePcm48k` et `pcmChunks`
sont les fonctions de capture/restitution de l'application. Les tests
ordinaires utilisent de vraies connexions QUIC et vérifient le certificat,
le duplex au bit contre le batch, le 503, les bornes et l'inactivité ; ils ne
constituent pas un test de compatibilité de tous les navigateurs.

### Reproduire les mesures de transport

Les tests ignorés `ws_abba_latency`, `h2_abba_latency` et
`webtransport_abba_latency` alternent six blocs ABBA contre le chunked sur
loopback, après warm-up. Chaque essai contient 64 hops et vérifie `load1 < 2`
avant et après ; sinon la mesure échoue et aucun résultat agrégé n'est publié.
Ils distinguent la première réponse (handshake/headers), le premier PCM après
les quatre hops d'amorçage, puis les allers-retours par hop (p50 et p99).
Pour HTTP/2, la première réponse est mesurée à la disponibilité des headers
par l'API ; pour WebTransport elle inclut TLS/QUIC/CONNECT. Les sorties WS/h2/WT
sont en f32 dans ce banc, contre s16 pour l'endpoint WAV chunked existant.

```bash
CARGO_BUILD_JOBS=4 taskpolicy -b cargo test --release \
  --no-default-features --features ws,h2,webtransport \
  --bin saragossa abba_latency -- --ignored --nocapture --test-threads=1
```

Aucun résultat de latence sous charge n'est présenté comme une mesure valide.
Les comparaisons de fidélité utilisent des poids récurrents synthétiques
communs à l'oracle historique, et ne nécessitent ni téléchargement ni GPU.

### Oracle

`tools/voice/dfn3_ref.py` construit la référence : les classes PyTorch
officielles (`Encoder`, `ErbDecoder`, `DfDecoder`, `DfNet`, `Conv2dNormAct`,
`SqueezedGRU_S`, `GroupedLinearEinsum`, `DF`…) sont extraites telles quelles
par AST du dépôt `Rikorose/DeepFilterNet`, et le state dict est chargé en
`strict=True`. Le paquet `libdf` (extension Rust/PyO3) n'étant pas installé,
ses features et son STFT sont réécrits en numpy à partir de `lib.rs`.

Tolérances figées avant la première comparaison (test
`denoise::tests::dfn3_matches_pytorch_reference`, ignoré sans
`SARAGOSSA_DFN3_DIR`) : features ≤ 1e-4 en absolu, masque ≤ 1e-3, audio
≥ 60 dB de SNR et ≤ 1e-3 en absolu. Justification : tout est en f32, seul
l'ordre des sommes change (huit accumulateurs contre BLAS) sur des produits de
3 072 termes et cinq GRU récurrents sur 300 trames ; l'erreur attendue est de
l'ordre de 1e-5 en relatif, alors qu'une erreur de disposition des poids donne
au mieux −20 dB.

Résultat sur 3 s de parole FLEURS + bruit rose et ronflement à 5 dB :

| Grandeur | Écart max |
|---|---|
| features | 5,7e-7 |
| masque ERB | 1,5e-6 |
| audio de sortie | 6,0e-8 (SNR 134,6 dB contre la référence) |

### Latence par trame de 10 ms

| Chemin | Coût par hop de 10 ms (p50) | Facteur temps réel |
|---|---|---|
| CPU, un thread, port Rust (`dfn3_hop_latency`) | 624 µs (p99 4,5 ms sous charge 55) | 0,062 |
| CPU, fichier entier via `saragossa enhance` (100 clips) | ≈ 500 µs | 0,050 |
| GPU, command buffer vide (commit + attente) | 15 µs | — |
| GPU, sonde à 35 noyaux dépendants de même forme que DFN3 (`dfn3_gpu_hop_latency_probe`) | 306 µs au mur, dont 138 µs de temps GPU | 0,031 |

Sur GPU, le coût n'est donc pas nul. Chaque hop exige environ 35 noyaux
dépendants dans un command buffer, puis une attente : on paie environ 0,3 ms
au mur, soit la moitié du chemin CPU. La sonde ne calcule pas le modèle : elle
reproduit son graphe de dépendances et ses MAC (3,55 M par hop) avec des
produits matrice-vecteur naïfs. Elle donne un ordre de grandeur, pas un port
validé. Dans les deux cas, la latence perçue reste dominée par la latence
algorithmique de 40 ms (30 ms de retard plus le hop en cours de remplissage).
Un port GPU ferait gagner environ 0,3 ms sur 40, et se battrait avec le LLM
pour la file Metal : le CPU reste le bon choix.

### Qualité

Corpus : 50 clips FLEURS (25 en, 25 fr), rééchantillonnés à 48 kHz et mélangés
à un bruit synthétique déterministe (bruit rose plus ronflement à 100 Hz
modulé), à 0 et 5 dB de SNR (`tools/voice/mix_noise.py`). Débruitage par la
CLI Rust. Aucun script PESQ ou DNSMOS n'est disponible localement : la mesure
retenue est le SI-SDR contre le signal propre.

| SNR d'entrée | SI-SDR bruité | SI-SDR débruité | Gain moyen | Gain minimal |
|---|---|---|---|---|
| 0 dB | −0,01 dB | 10,63 dB | +10,64 dB | +4,15 dB |
| 5 dB | 5,00 dB | 13,36 dB | +8,36 dB | −0,24 dB |

WER de Whisper large-v3-turbo (mlx, langue imposée, température 0) avant et
après débruitage, sur les mêmes 100 clips. Le WER de chaque clip est plafonné
à 100 %, car deux sorties débruitées déclenchent des boucles de répétition
(« je ne sais pas… » ×40), qui dominent sinon la moyenne brute.

| SNR | Langue | Propre | Bruité | Débruité DFN3 |
|---|---|---|---|---|
| 0 dB | en | 5,1 % | 21,2 % | 25,9 % |
| 0 dB | fr | 4,5 % | 48,1 % | 69,7 % |
| 5 dB | en | 5,1 % | 8,8 % | 12,5 % |
| 5 dB | fr | 4,5 % | 25,2 % | 35,4 % |

**DFN3 dégrade le WER de Whisper sur ce bruit**, alors qu'il gagne 8 à
10 dB de SI-SDR. Les médianes vont dans le même sens : 24 % → 44 % à 0 dB,
9 % → 18 % à 5 dB. C'est cohérent avec ce qu'on sait des ASR robustes :
Whisper est entraîné sur de l'audio bruité, et les artefacts du masquage
(consonnes atténuées, bandes ERB coupées) le gênent plus que le bruit
stationnaire. Le français souffre davantage. DFN3 sert donc à l'écoute
humaine et à la visio, **pas comme prétraitement de `transcriptions`**.

## MossFormer2_SS_16K

### Spécification

Code de référence : `clearvoice/models/mossformer2_ss/` de ClearerVoice-Studio,
checkpoint `last_best_checkpoint.pt` (55 736 034 paramètres, 1 099 tenseurs,
chargés en `strict=True`).

- **Encodeur** : conv 1D 1 → 512, noyau 16, stride 8, sans biais, ReLU :
  2 000 trames par seconde à 16 kHz.
- **MaskNet** : LayerNorm globale (« ln » sur canaux et temps), conv 1×1
  512 → 512, plus un plongement sinusoïdal mis à l'échelle.
- **24 couches**, chacune composée de :
  1. `FLASH_ShareA_FFConvM` : décalage d'un pas sur la moitié des canaux,
     `to_hidden` (ScaleNorm, linéaire 512 → 2 048, SiLU, `ConvModule` :
     conv en profondeur de noyau 17 avec résiduel) découpé en `v` et `u` de
     1 024 ; `to_qk` 512 → 128, puis `OffsetScale` à quatre têtes (q et k
     quadratiques, q et k linéaires) ; RoPE sur 32 dimensions ; attention
     quadratique par groupes de 256 trames `relu(q·k / 256)²`, plus attention
     linéaire globale `(k·v)/n` ; `(att_u · v) · sigmoid(att_v · u)`, puis
     `to_out` 1 024 → 512 et résiduel ;
  2. `Gated_FSMN_Block_Dilated` : conv 1×1 512 → 256 + PReLU, CLayerNorm,
     FSMN à portes (`to_u` et `to_v` de 256, mémoire FSMN dilatée en conv 2D
     de noyau 39 à connexions denses), CLayerNorm, conv 1×1 256 → 512 et
     résiduel.
- LayerNorm finale, PReLU, conv 1×1 512 → 1 024 (deux locuteurs), porte
  `tanh · sigmoid`, conv 1×1 512 → 512 et ReLU : deux masques.
- **Décodeur** : conv transposée 1D 512 → 1, noyau 16, stride 8.
- Décodage officiel (`decode.py`) : une passe si l'audio tient dans
  `one_time_decode_length`, sinon fenêtres de 2 s avec un pas de 1,5 s sans
  réalignement des locuteurs entre fenêtres. Les sorties sont renormalisées
  au RMS de l'entrée.

Le modèle est non causal : attention linéaire et normalisation globales sur
toute la séquence. Le coût est d'environ 3,3 M de MAC par trame et par couche,
soit environ 160 GMAC par seconde d'audio (0,3 TFLOP).

### Référence et mesures

`tools/voice/moss_ref.py` charge le code officiel tel quel, avec deux
substituts : `rotary_embedding_torch` est le fichier du dépôt
lucidrains/rotary-embedding-torch, et `torchinfo.summary` est un stub sans
effet. Le décodage se fait en une passe (12 paires de 5 à 12 s), suivi de la
renormalisation RMS.

SI-SDR de séparation (meilleure permutation) sur des paires FLEURS
superposées à 0 dB. Exécution sur CPU avec PyTorch, en une passe :

| Paires | SI-SDRi moyen | Remarque |
|---|---|---|
| 6 × en + fr | 20,99 dB (20,24 à 22,01) | |
| 3 × en + en | 6,49 dB (−1,02, 8,80, 11,70) | la paire 8 échoue (−1 dB) : locuteurs probablement proches |
| 3 × fr + fr | 16,97 dB (13,35 à 19,12) | |

Temps de calcul de la référence PyTorch : sur CPU, RTF de 1,24 à 1,48 ; sur
GPU (MPS, f32, une passe, paire 3 de 11,8 s), **0,94 s, soit 80 ms de GPU par
seconde d'audio (RTF 0,080)**, avec le même SI-SDRi (20,76 dB) que sur CPU.

WER de Whisper large-v3-turbo par voix (plafonné à 100 % par clip). Les
références sont les phrases FLEURS entières, alors que le mélange est tronqué
au clip le plus court : le WER « propre » sert donc de plancher (14 %).

| Paires | Voix propre tronquée | Mélange | Voix séparée |
|---|---|---|---|
| en + fr | 16,9 % | 35,1 % | 18,3 % |
| en + en | 15,4 % | 63,9 % | 39,1 % |
| fr + fr | 8,4 % | 61,8 % | 17,4 % |
| toutes | 14,4 % | 49,0 % | 23,2 % |

Hors l'échec en + en (paire 8), la séparation ramène le WER près du plancher.

### Port saragossa

Pas encore fait. Le port demande sur GPU : la LayerNorm globale, la ScaleNorm,
les convolutions en profondeur 1D (noyau 17) et 2D dilatées denses (noyau 39),
l'attention quadratique en groupes de 256 avec `relu²`, l'attention linéaire
globale, la RoPE partielle sur 32 dimensions, PReLU et la conv transposée 1D.
Les GEMM (512 ↔ 2 048, 1 024 → 512) représentent plus de 80 % des MAC et
peuvent réutiliser les matmuls f32/bf16 de `metal_backend`. Estimation : 5 à
8 jours, avec un oracle couche par couche contre `moss_ref.py`.

## RE-USE (NVIDIA-NemotronLabs-RE-USE)

### Spécification

Fichiers locaux lus sans être exécutés : `config.json`, `models/*.py`,
`inference.py` ; `model.safetensors` contient 9 607 747 paramètres f32.

- **STFT à fenêtre fixe (SFI)** : `n_fft = 320 · sr / 8 000`, pas de 40 · sr/8 000
  et fenêtre de Hann centrée, soit 40 ms de fenêtre et 5 ms de pas à toute
  fréquence (1 920 et 240 à 48 kHz, F = 961 bins). Magnitude compressée par
  `log1p`, et phase en entrée ;
- **DenseEncoder** : conv 1×1 2 → 64, InstanceNorm2d et PReLU, DenseBlock de
  4 convs 3×3 dilatées en temps (1, 2, 4, 8), à connexions denses de 64 à 256
  canaux, puis conv 1×3 de stride (4, 2) : 50 trames par seconde et environ F/2 bins ;
- **30 TFMambaBlock** : Mamba bidirectionnel sur l'axe temps pour chaque bin,
  puis sur l'axe fréquence pour chaque trame. Chaque Mamba a `d_model = 64`,
  `expand = 4` (`d_inner = 256`), `d_state = 16`, `d_conv = 4` et
  `dt_rank = 4` ; les deux sens sont concaténés, projetés (128 → 64) et
  normalisés (LayerNorm), avec un résiduel ;
- **décodeurs** magnitude et phase : DenseBlock, deux `SPConvTranspose2d`
  (sous-pixel ×2 en fréquence et ×4 en temps), conv 1×1 ; la phase sort par
  `atan2`. Reconstruction par iSTFT.

### Coût estimé

Par jeton (une case temps-fréquence après l'encodeur), un Mamba coûte environ
71 k MAC (`in_proj` 32 k, `x_proj` 9 k, scan 12 k, `out_proj` 16 k). Un
TFMambaBlock en exécute quatre, soit environ 0,3 M MAC, et les 30 blocs
environ 9 M MAC par jeton.

| sr | Jetons/s (50 trames × F/2) | Mamba | DenseBlocks | Total |
|---|---|---|---|---|
| 16 kHz | 50 × 161 ≈ 8 000 | ≈ 72 GMAC/s | ≈ 30 GMAC/s | ≈ 0,2 TFLOP/s d'audio |
| 48 kHz | 50 × 481 ≈ 24 000 | ≈ 216 GMAC/s | ≈ 90 GMAC/s | ≈ 0,6 TFLOP/s d'audio |

Le scan sélectif est séquentiel le long d'un axe, mais il est parallèle sur
256 × 16 états et sur l'autre axe (481 bins ou trames) ; il tient donc sur le
GPU avec le même schéma par morceaux que le gated delta net de
`linear_attention.rs`. À 2-5 TFLOP f32 effectifs, le RTF attendu est de 0,1 à
0,3 à 48 kHz et de 0,04 à 0,1 à 16 kHz (estimation à mesurer). Le modèle est
non causal : hors ligne seulement, par morceaux pour les fichiers longs
(`inference_chunk.py`).

### État

Pas de référence exécutée : le code dépend de `mamba_ssm` (noyaux CUDA), absent
sur macOS. Un oracle demanderait d'abord un scan Mamba de référence en PyTorch
pur. Estimation du port : 6 à 10 jours (scan bidirectionnel GPU, convs 2D
denses, InstanceNorm, sous-pixel, iSTFT).

## Débruitage multi-cœur (opt-in)

`SARAGOSSA_DENOISE_THREADS=N` (défaut 1 : chemin historique, bit-identique)
découpe le signal en N segments, chacun chauffé sur 1 s de contexte, traités en
parallèle puis recollés par un fondu linéaire de 0,5 s centré sur chaque
frontière. Le fichier entier est gardé en mémoire (pas de flux borné) : environ
0,7 Go par heure en entrée et autant en sortie (f32, 48 kHz), plus les copies par
segment. Seuil de proximité au mono-cœur fixé avant mesure : SNR ≥ 30 dB,
validé seulement sur le modèle synthétique, pas encore mesuré sur le vrai DFN3.
Risque connu : quand le début d'un segment n'est pas multiple de 480, sa grille
STFT est décalée par rapport au mono-cœur (écart borné par le fondu, non mesuré).

Bench de nuit, RTF 1 contre N threads sur le même clip long :

```bash
for n in 1 2 4 8; do SARAGOSSA_DENOISE_THREADS=$n cargo run --release -- enhance \
  --model $SARAGOSSA_DFN3_DIR/dfn3.safetensors --input long.wav --output out_$n.wav; done
```

Chaque run affiche `RTF` ; comparer `out_N.wav` à `out_1.wav` (SNR ≥ 30 dB).
