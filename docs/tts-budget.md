# Budget TTS du serveur audio

`POST /v1/audio/speech` calcule son budget à partir du texte nettoyé :
`min(800, max(160, 2 × nombre de caractères Unicode))`. La règle reprend les
deux trames par caractère de Réti ; le plafond historique du moteur (800)
reste plus strict que celui de Réti (2 000). À 12,5 trames/s, 800 trames
représentent 64 s, contre 12,8 s pour l'ancien défaut de 160.

Une valeur entière positive de `SARAGOSSA_TTS_MAX_FRAMES` remplace le calcul,
y compris sous le plancher de 160, puis reste bornée à 800. Une valeur absente,
nulle, invalide ou hors plage `usize` revient au calcul proportionnel.
`SARAGOSSA_TTS_MAX_FRAMES=160` restaure le budget précédent.

Le WAV est toujours renvoyé en HTTP 200. L'en-tête
`X-Saragossa-TTS-Frame-Limit-Reached: true` indique que le budget effectif a été
consommé sans EOS observé : l'audio peut être incomplet. Il vaut `false` sinon.
Le même indicateur est disponible dans `TtsSynthesisOutput::frame_limit_reached`,
en batch et en streaming, avec prise en compte des plafonds internes du clone.
Les garde-fous anti-répétition restent inchangés ; `false` ne certifie donc pas
une fin par EOS. L'ajout du champ exige d'adapter les constructions littérales
de cette structure dans les clients Rust.

Les budgets explicites de la bibliothèque, le cap adaptatif du clone, les
kill-switches du codec et les replis résident/per-op sont conservés. Le serveur
applique désormais aussi la borne 800 au mode VoiceDesign et aux surcharges.

## Mémoire au plafond : audit statique

Pour le codec 24 kHz, 1 920 échantillons/trame et 16 groupes :

- 800 trames donnent 1 536 000 échantillons, soit 6 144 000 octets PCM f32
  (5,86 Mio), et 3 072 044 octets WAV PCM16. Les codes occupent 51 200 octets
  hors allocations des vecteurs. PCM et WAV coexistent pendant la réponse.
- Le KV talker f32 nécessite `2 × couches × têtes_KV × dimension_tête ×
  (préfixe + 800) × 4` octets. L'allocation résidente utilise une addition
  vérifiée et sa garde mémoire existante conserve le repli per-op.
- L'attention du préfixe codec alloue une ligne de scores à la fois, au plus
  `800 × 4 = 3 200` octets ; son temps est quadratique, pas son stockage des
  scores. Q/K/V et les activations restent linéaires dans le nombre de trames.
- Un tampon du dernier étage codec vaut `6 144 000 × canaux` octets. À titre
  d'échelle, 64 canaux représentent 375 Mio **par tampon**, pas pour le moteur
  entier. Le pool GPU conserve plusieurs tailles et plusieurs temporaires :
  le PCM final n'est donc pas une estimation du pic mémoire. Passer de 160 à
  800 multiplie par cinq les tailles dépendantes de la durée ; les poids sont
  constants. La borne de trames ne borne pas les embeddings du texte d'entrée.

Ces comptes proviennent des allocations de `tts_codec.rs`, `tts_codec_gpu.rs`
et `decoder/resident/setup.rs`, sans charger de modèle. Le pic RSS/Metal dépend
des dimensions du checkpoint et reste à mesurer. Le test ignoré
`tts::tests::codec::codec_at_server_frame_cap` force le codec à 800 trames pour
cette vérification ; il échoue si le snapshot ou le codec Metal manque.

Ce test isolé ne mesure pas le pire cas du serveur. Restent à relever, pour
chaque checkpoint et machine ciblés :

- le pic RSS/empreinte du processus et les allocations Metal au chargement,
  pendant le préfill/talker, au décodage codec et pendant l'encodage/envoi WAV,
  avec poids, KV, pool, temporaires et PCM/WAV simultanément vivants ; en mémoire
  unifiée, ne pas additionner RSS et Metal comme des consommations disjointes ;
- les mêmes pics à froid puis après plusieurs requêtes de tailles différentes,
  aux budgets 160 et 800, avec les chemins résidents puis leurs replis existants,
  pour inclure les buffers conservés par les pools ;
- une synthèse complète atteignant réellement 800 trames (un arrêt EOS précoce
  ne valide pas ce cas), ainsi que le plus long préfixe texte accepté par la
  route. La limite de trames seule ne donne pas de borne mémoire globale.

Consigner le checkpoint, ses dimensions, la longueur tokenisée du préfixe,
le nombre de trames obtenu, les flags actifs, la RAM et les pics par phase.
Le test ci-dessous ne couvre ni le talker/KV ni la coexistence PCM/WAV du
serveur ; sa réussite ne suffit pas à déclarer le plafond sûr en mémoire.

```sh
CARGO_NET_OFFLINE=true SARAGOSSA_REQUIRE_REAL_MODELS=1 taskpolicy -b cargo test --lib tts::tests::codec::codec_at_server_frame_cap -- --ignored --exact --nocapture
```
