# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.35](https://github.com/azerozero/saragossa/compare/v0.3.34...v0.3.35) - 2026-11-24

### Added

- *(loader)* refuse sliding_window inférieur au contexte pour Llama et Mistral
- *(loader)* RoPE llama3, templates Llama 3 et Mistral, oracle KL Mistral 7B

### Fixed

- *(config)* entier RoPE non représentable en f64 comparé tel quel
- *(config)* entiers RoPE comparés exactement, sans passer par f64
- *(config)* facteurs RoPE comparés numériquement (8 et 8.0 équivalents)
- *(test)* helper rope sans rms_norm_eps manquant, flottants JSON, liste golden llama31
- *(config)* rope_parameters remplace rope_scaling nul, biais cohérents, sliding_window refusé, système Mistral en loop.last

### Other

- Merge pull request #339 from azerozero/feat/llama-mistral
- *(loader)* helpers de test sans collision de noms
- *(llama)* rétablit les tests de main, date par version, oracle RoPE llama3 et configs de familles
- fusionne origin/main dans feat/llama-mistral

## [0.3.34](https://github.com/azerozero/saragossa/compare/v0.3.33...v0.3.34) - 2026-11-23

### Fixed

- *(metal)* qmv u3 f32 sous le plancher du GEMM tuilé, tests non périodiques

### Other

- Merge remote-tracking branch 'origin/main' into perf/u3-prefill
- Merge pull request #328 from azerozero/dependabot/cargo/bytes-1.12.2
- Merge pull request #331 from azerozero/dependabot/github_actions/j178/prek-action-3.1.1
- *(deps)* bump bytes from 1.12.1 to 1.12.2
- *(deps)* bump libc from 0.2.186 to 0.2.191
- *(deps)* bump j178/prek-action from 3.1.0 to 3.1.1
- Merge pull request #326 from azerozero/dependabot/github_actions/taiki-e/install-action-2.90.6
- Merge pull request #327 from azerozero/dependabot/cargo/http-1.5.0
- Merge pull request #329 from azerozero/dependabot/cargo/flate2-1.1.11
- Merge pull request #330 from azerozero/dependabot/cargo/quinn-0.11.12
- *(deps)* bump http from 1.4.2 to 1.5.0
- *(readme)* écarts chiffrés, GGUF BF16, identité nuancée, avertissement LINEAR_CHUNKED rétabli
- *(deps)* bump taiki-e/install-action from 2.90.4 to 2.90.6
- *(bench)* nuance l'identité greedy de saragossa dans la page de campagne
- remet le README et la page de campagne à jour pour la vitrine
- *(metal)* GEMM de prefill u3 et qmv en lignes u3/u5/u6

## [0.3.33](https://github.com/azerozero/saragossa/compare/v0.3.32...v0.3.33) - 2026-11-23

### Added

- *(serve)* bridage gpu-share et max-tok-s pour run et serve

### Fixed

- *(serve)* max_tok_s respecte la priorité --set > env > profil
- *(serve)* gpu-share n'altère plus le lookup, sommeil interruptible, précédence env

### Other

- Merge pull request #337 from azerozero/release-plz-2026-11-23T16-36-03Z
- Merge pull request #336 from azerozero/feat/bride-gpu
- Merge remote-tracking branch 'origin/main' into feat/bride-gpu
- *(decoder)* tokens usize dans le test d'identité gpu-share

## [0.3.32](https://github.com/azerozero/saragossa/compare/v0.3.31...v0.3.32) - 2026-11-23

### Added

- *(convert)* experts MoE fusionnés empilés en switch_mlp

### Other

- Merge remote-tracking branch 'origin/main' into feat/convert-moe
- Merge origin/main dans feat/convert-moe

## [0.3.31](https://github.com/azerozero/saragossa/compare/v0.3.30...v0.3.31) - 2026-11-21

### Added

- *(convert)* quantification native HF BF16 vers affine u4/u8 gs64
- *(quant)* fige le protocole de campagne ≤ 4,2 bpw
- *(dflash)* découverte automatique du drafter, journal du décodage choisi et conseils doctor
- *(quant)* ajoute GPTQ avec Hessiennes séparées par expert
- *(quant)* ajoute une base 3 bits à l’allocation imatrix
- *(quant)* ajoute le 5 bits et le mélange imatrix explicite
- *(quant)* ajoute la calibration imatrix et le budget 4/6/8

### Fixed

- *(quant)* fixture tokenizer du test tensor_stream, données brutes et conclusion restreinte à D
- *(convert)* reprise par sha256 des sources, alias language_model.mtp
- *(dflash)* un drafter découvert au chargement invalide laisse place au candidat suivant
- *(dflash)* clés requises du loader partagées avec la découverte, conseil doctor limité au dtype bf16
- *(quant)* fusionne main et recopie le tokenizer source dans gptq
- *(convert)* tête MTP en u4 regroupée, reprise liée aux poids source, init MLX
- *(dflash)* checkpoint complet exigé, identité exacte du modèle, repli MTP et conseil doctor limité au BF16 dense
- *(dflash)* un drafter découvert n'est activé que sur cible dense BF16
- *(doctor)* retire le dépôt deviné du conseil drafter, note sur la conversion native
- *(quant)* conserve les destinations à créer
- *(quant)* borne la mémoire de conversion par blocs
- *(quant)* convert_ls retrouve les tenseurs d'un module répartis sur plusieurs shards

### Other

- Merge remote-tracking branch 'origin/main' into feat/convert-natif
- *(convert)* fixture modifiée en place plutôt que par un paramètre constant
- Merge pull request #322 from azerozero/feat/quant-campagne
- *(quant)* consigne la campagne ≤ 4,2 bpw arrêtée en E0
- Merge origin/main dans feat/auto-sidecars
- *(quant)* couvre le score poids et la référence de candidat hors dépôt
- Merge commit '098aa97' into feat/quant-campagne
- *(quant)* parallélise les blocs avec une file bornée

## [0.3.30](https://github.com/azerozero/saragossa/compare/v0.3.29...v0.3.30) - 2026-11-21

### Fixed

- *(test)* épingle l'historique MTP à l'alignement exact et le contrôle f32 Qwen MoE au qmv

### Other

- Merge pull request #315 from azerozero/test/identite-speculative
- Merge pull request #316 from azerozero/perf/dflash-budget
- *(decoder)* DFlash B=4 charge le décodeur comme serve
- *(dflash)* verrouille les budgets d'arbre par défaut en valeurs littérales
- Merge pull request #318 from azerozero/release-plz-2026-11-21T16-39-56Z
- Merge pull request #317 from azerozero/fix/tests-rouges-gpu
- *(dflash)* arbre B = 3 par défaut en greedy sur cible quantifiée
- *(decoder)* identité spéculative et tête de draft u4

## [0.3.29](https://github.com/azerozero/saragossa/compare/v0.3.28...v0.3.29) - 2026-11-21

### Fixed

- *(mtp)* with_mtp_weights pour toute tête détectée par ModelAssets
- *(mtp)* provenance de la tête et convention du tronc transmises par l'appelant
- *(mtp)* décalage des normes décidé par la convention du checkpoint, non par le préfixe
- *(mtp)* normes 1+w pour la tête dense à clés mtp.* (tête embarquée du 27B BF16)

### Other

- Merge pull request #314 from azerozero/perf/mtp-bf16
- *(perf)* mesure des normes 1+w de la tête MTP BF16

## [0.3.28](https://github.com/azerozero/saragossa/compare/v0.3.27...v0.3.28) - 2026-11-21

### Added

- *(bench)* configurations mlx-mtp et mlx-dflash par mlx-vlm 0.8.0
- *(bench)* saragossa en fin de campagne, drapeau bench-en-cours et arrêt sans blocage
- *(bench)* campagne autonome de bench concurrents en une commande

### Fixed

- *(bench)* un descendant déjà disparu à la capture compte comme arrêté
- *(bench)* PID vérifiés avant kill, gel revérifié à la reprise, commit explicite, docs corrigées
- *(bench)* DONE seulement si le plan est clos, port occupé rejoué, chauffe sans longueur codée en dur
- *(bench)* prompts sans tokens spéciaux, top-k de llmd 20261114, mistral.rs 35B noté non supporté
- *(bench)* verrou GPU tenu pour toute la campagne, refroidissement borné à 5 min

### Other

- Merge pull request #284 from azerozero/feat/campagne-autonome
- *(bench)* données brutes des campagnes du 21 novembre (mlx et finale)
- *(bench)* précise l'alimentation de la campagne 20-dfl
- *(bench)* données brutes des campagnes du 19 et 20 novembre

## [0.3.27](https://github.com/azerozero/saragossa/compare/v0.3.26...v0.3.27) - 2026-11-21

### Fixed

- *(test)* saute proprement les tests sur poids locaux absents, corrige le code mort h2

## [0.3.26](https://github.com/azerozero/saragossa/compare/v0.3.25...v0.3.26) - 2026-11-21

### Fixed

- *(quant)* quantification u4 disponible hors devtools pour la tête de draft MTP

### Other

- Merge pull request #309 from azerozero/perf/mtp-profondeur
- *(mtp)* ne quantifie pas la tête de draft quand un sidecar la remplace
- *(mtp)* profondeur fixe 2 en greedy committed et tête de draft u4 sur cible bf16

## [0.3.25](https://github.com/azerozero/saragossa/compare/v0.3.24...v0.3.25) - 2026-11-21

### Other

- *(metal)* identité au bit du qmv bf16 en lignes pour 5, 7 et 8 lignes
- *(dflash)* arbre B = 4 par défaut en greedy sur cible bf16 dense
- *(metal)* qmv bf16 en lignes à une ligne de poids par simdgroup jusqu'à 8 lignes

## [0.3.24](https://github.com/azerozero/saragossa/compare/v0.3.23...v0.3.24) - 2026-11-20

### Other

- *(gpu)* liste le test d'identité du qmv bf16 en lignes
- *(metal)* qmv bf16 en lignes pour la vérification spéculative et compteurs DFlash dans serve

## [0.3.23](https://github.com/azerozero/saragossa/compare/v0.3.22...v0.3.23) - 2026-11-20

### Fixed

- *(serve)* l'enfant du relais ferme les sockets héritées non déclarées

### Other

- Merge pull request #302 from azerozero/fix/relais-sans-refus
- *(perf)* ligne vide avant la liste du plancher de bruit
- Merge pull request #287 from azerozero/perf/prefill-quant
- *(prefill)* portée du bf16×4 Qwen MoE, semis MTP compris, commentaires à jour
- *(prefill)* bf16×4 par défaut sur les petites projections des Qwen MoE
- *(prefill)* variante bf16×4 des petites projections u4 en opt-in
- Merge remote-tracking branch 'origin/main' into perf/prefill-quant
- *(prefill)* GEMM bf16×3 des petites projections Qwen en opt-in

## [0.3.22](https://github.com/azerozero/saragossa/compare/v0.3.21...v0.3.22) - 2026-11-20

### Fixed

- *(mtp)* clés du tronc ignorées quand la tête est embarquée dans un shard
- *(mtp)* tête MTP embarquée dans les shards et avertissement si inactive

### Other

- Merge pull request #300 from azerozero/test/mlp-fusionne-offsets
- Merge pull request #299 from azerozero/fix/mtp-tete-embarquee
- *(metal)* offsets alignés acceptés par le MLP fusionné depuis f2331f6

## [0.3.21](https://github.com/azerozero/saragossa/compare/v0.3.20...v0.3.21) - 2026-11-20

### Fixed

- *(prefill)* API des tenseurs coopératifs du MLP bf16 fusionné

### Other

- Merge remote-tracking branch 'origin/main' into perf/prefill-bf16-fusions
- *(perf)* KL des fusions MLP et b/a du prefill BF16
- *(prefill)* poids bf16 synthétiques tronqués après l'échelle
- *(prefill)* MLP bf16 fusionné et GEMM b/a de la linear attention, opt-in

## [0.3.20](https://github.com/azerozero/saragossa/compare/v0.3.19...v0.3.20) - 2026-11-20

### Other

- Merge pull request #286 from azerozero/perf/suffix-decoding
- *(decoder)* bilan opt-in des drafts lookup acceptés
- *(decoder)* source de brouillon SuffixDecoding opt-in pour les lookups

## [0.3.19](https://github.com/azerozero/saragossa/compare/v0.3.18...v0.3.19) - 2026-11-20

### Fixed

- *(prefill)* borne le tampon de timestamps GPU à 4096 échantillons

### Other

- Merge pull request #282 from azerozero/perf/prefill-bf16-27b
- tests X1 exemptés de la CI GPU, GEMM NA absent du runner
- *(gpu)* tests X1 du prefill bf16 dans la liste synthétique
- Merge branch 'main' into perf/prefill-bf16-27b
- *(perf)* périmètre, conditions K et N et mémoire du GEMM bf16 à un produit
- *(prefill)* borne le GEMM bf16 à un produit au prefill Qwen dense bf16
- Merge remote-tracking branch 'origin/main' into perf/prefill-bf16-27b
- *(perf)* prefill BF16 à un produit, vitesse, KL et identité
- *(prefill)* GEMM bf16 à un produit par défaut, retrait de NFIRST
- *(prefill)* GEMM bf16 dense à un produit et grille N d'abord en opt-in

## [0.3.18](https://github.com/azerozero/saragossa/compare/v0.3.17...v0.3.18) - 2026-11-20

### Fixed

- *(serve)* pas de second 500 après réponse commencée, invalidation audio et embeddings
- *(serve)* le parent n'accepte plus après l'annonce du vidage
- *(serve)* panique moteur contenue, jeton hors argv, WebTransport durci

### Other

- Merge pull request #294 from azerozero/fix/na-compile-garde
- *(metal)* garde NA, noyaux générés par macro exclus de l'analyse
- Merge pull request #293 from azerozero/test/audit-suite
- Merge pull request #291 from azerozero/fix/audit-secu-docs
- liste GPU synthétique complétée, garde d'environnement unique, mélange nocturne
- *(serve)* relais séquentiel déterministe, sockets transmises par le test

## [0.3.17](https://github.com/azerozero/saragossa/compare/v0.3.16...v0.3.17) - 2026-11-19

### Fixed

- *(metal)* bibliothèque NA compilée une fois, échec signalé

## [0.3.16](https://github.com/azerozero/saragossa/compare/v0.3.15...v0.3.16) - 2026-11-19

### Other

- Merge pull request #281 from azerozero/perf/denoise-math-commune
- Merge pull request #283 from azerozero/perf/decode-quant-largeur
- Merge origin/main dans perf/denoise-math-commune
- *(denoise)* variante commune opt-in (SARAGOSSA_DENOISE_COMMON), libm reste le défaut
- *(decode)* sonde qmv u4 en DRAM, poids tournants et command buffers longs
- *(denoise)* variante mathématique commune par défaut sur CPU, libm en kill-switch
- *(decode)* sonde de bande passante des qmv u4 dense et MoE

## [0.3.15](https://github.com/azerozero/saragossa/compare/v0.3.14...v0.3.15) - 2026-11-18

### Added

- *(denoise)* variante faible latence DeepFilterNet3_ll, choisie par le modèle

### Fixed

- *(whisper)* is_silence prend AudioFeatures après fusion de main
- *(serve)* sérialiser les tests qui écrivent les atomiques de rechargement
- *(decode)* tampon de transposition qmv bf16 déclaré en uint4
- *(denoise)* fusion de main, multi-cœur exact avec Math et chemin GPU
- *(poids)* offsets des vues mmap liés partout, bf16 dense toujours copié
- *(denoise)* invariants, gardes de forme et cache des noyaux du chemin GPU DFN3
- *(stt)* déclarer les kill-switches dans runtime_flags et prouver l'identité résidente

### Other

- Merge pull request #279 from azerozero/feat/dfn3-faible-latence
- Merge pull request #280 from azerozero/perf/bf16-stage-defaut
- Merge pull request #252 from azerozero/perf/stt-tts-suite
- *(decode)* noyau qmv bf16 à blocs de 256 activé par défaut
- Merge remote-tracking branch 'origin/main' into perf/stt-tts-suite
- Merge pull request #275 from azerozero/perf/denoise-mc-exact
- Merge pull request #276 from azerozero/perf/decode-bf16-gemv
- *(decode)* identité au bit du qmv bf16 aussi sans fast-math
- Merge remote-tracking branch 'origin/main' into perf/denoise-mc-exact
- Merge pull request #273 from azerozero/perf/poids-partages
- *(decode)* qmv bf16 à lectures vectorisées, identique au bit, en opt-in
- *(denoise)* multi-cœur exact au bit près, GRU séquentiels
- Merge pull request #272 from azerozero/perf/denoise-gpu-exact
- *(poids)* poids quantifiés partagés entre processus par mmap opt-in
- *(denoise)* réseau DFN3 en lot sur GPU Metal pour les fichiers longs, identique au bit au CPU commun
- *(tts)* lire le QKV du talker sans copie quand les poids sont adossés au stockage
- *(tts)* prouver que les couches du talker fusionnées sont encodées
- intégrer main et préserver les optimisations audio
- fusionner les derniers correctifs audio de main
- *(tts)* comparer les couches du talker fusionnées au bit près
- intégrer les dernières évolutions de main
- fusionner main dans la suite STT et TTS
- *(stt)* garder les features sur le GPU, attention decode bf16 en opt-in
- *(tts)* encoder les couches du talker en 8 dispatches
- *(stt)* calculer le log-mel par trames en parallèle

## [0.3.14](https://github.com/azerozero/saragossa/compare/v0.3.13...v0.3.14) - 2026-11-18

### Added

- *(diarisation)* fusion QKV dans l'encodeur Metal résident et test streaming sur poids réels

### Fixed

- *(diarisation)* pas de q/k/v séparés sur le GPU en mode fusionné, écart journalisé

### Other

- Merge pull request #264 from azerozero/feat/diar-metal-qkv
- *(diarisation)* emprunt inutile signalé par clippy
- Merge remote-tracking branch 'origin/main' into feat/diar-metal-qkv
- *(diarisation)* branches séparée et fusionnée explicites, indépendantes de l'environnement

## [0.3.13](https://github.com/azerozero/saragossa/compare/v0.3.12...v0.3.13) - 2026-11-18

### Added

- *(serve)* relais des listeners h2 et QUIC, bascule séquentielle selon la mémoire
- *(serve)* liste de modèles rechargeable à chaud par SIGHUP
- *(serve)* relais sans coupure du binaire sur SIGUSR2
- *(bench)* harnais de latence des transports serve (SSE, WS, h2, WebTransport)
- *(diar)* diarisation incrémentale et q/k/v fusionnés opt-in

### Fixed

- *(serve)* relais plus strict sur les fd hérités et la socket QUIC
- *(serve)* accept non bloquant pendant le relais, délai du signal prêt
- *(bench)* appariement FIFO des trames WS cadencées à 30 ms, p95 interpolé, blocs ABBA stricts
- *(whisper)* ignore les fenêtres sans parole (no_speech_threshold 0,6 et logprob -1,0)

### Other

- Merge remote-tracking branch 'origin/main' into feat/relais-h2-quic
- *(serve)* rustfmt du relais
- *(serve)* alias du type des listeners hérités
- Merge pull request #259 from azerozero/bench/transports
- Merge pull request #258 from azerozero/fix/whisper-silence
- Merge pull request #257 from azerozero/feat/diar-streaming
- Merge pull request #268 from azerozero/release-plz-2026-11-17T23-00-37Z
- Merge remote-tracking branch 'origin/main' into feat/relais-sans-coupure
- *(whisper)* verdict de silence isolé et limites du logprob documentées

## [0.3.12](https://github.com/azerozero/saragossa/compare/v0.3.11...v0.3.12) - 2026-11-17

### Added

- *(config)* config check/print, --set, presets embarqués et rechargement SIGHUP des limites chaudes

### Fixed

- *(config)* isoler le test de rechargement, SIGHUP seulement avec profil, origine env valide, plafond de lot relu à l'admission

## [0.3.11](https://github.com/azerozero/saragossa/compare/v0.3.10...v0.3.11) - 2026-11-17

### Added

- *(denoise)* débruitage multi-cœur opt-in par segments avec fondu

### Other

- Merge pull request #262 from azerozero/release-plz-2026-11-17T20-34-59Z
- Merge pull request #260 from azerozero/feat/denoise-multicoeur
- *(denoise)* seuil limité au modèle synthétique, mémoire du mode N>1, flux inutile évité

## [0.3.10](https://github.com/azerozero/saragossa/compare/v0.3.9...v0.3.10) - 2026-11-17

### Added

- *(config)* profil TOML optionnel pour serve (--config)
- *(vision)* ordonne les patchs dans les groupes spatiaux
- *(vision)* normalise les pixels et répète les frames
- *(vision)* ajoute le bicubique antialiasé sur CPU
- *(vision)* reproduis les dimensions smart_resize de Qwen
- *(vision)* injecte les embeddings et transporte M-RoPE

### Fixed

- *(config)* mémoire serve, erreur TOML localisée, lock minimal
- *(vision)* bicubique aligné sur torchvision (poids int16), PIL informatif
- *(vision)* budget de pixels explicite avant allocation dans normalize et into_patches
- *(vision)* préserve la division entière de smart_resize
- *(snapshot)* recalcule le delta au préfixe sauvegardé
- *(mtp)* replie les cycles résidents sur les positions vision

### Other

- Merge pull request #256 from azerozero/feat/profils-toml
- *(gitleaks)* en-tête de bloc perdu à la fusion
- Merge remote-tracking branch 'origin/main' into feat/vision-preprocess
- *(gitleaks)* exceptions ciblées pour les SHA256 des fixtures tokenizer et les clés rust-cache
- *(gitleaks)* exception pour les SHA256 publics de la calibration vision
- *(vision)* dimensions entre backticks pour rustdoc
- *(vision)* oracle torchvision à côté de PIL pour le resize bicubique
- *(vision)* fusionne main et garde le prétraitement optionnel
- *(vision)* précise les limites du décodage image
- fusionne origin/main dans feat/vision-mrope
- *(vision)* explicite les invariants de lecture des flottants
- *(vision)* fige les logits de référence et le bruit BF16

## [0.3.9](https://github.com/azerozero/saragossa/compare/v0.3.8...v0.3.9) - 2026-11-17

### Added

- *(webtransport)* ajoute le débruitage duplex sur HTTP/3 WebTransport
- *(h2)* ajoute le débruitage duplex sur h2c
- *(ws)* expose le débruitage PCM en duplex

### Fixed

- *(prefill)* plafond flint vérifié avant l'ancre alignée et le prolongement
- *(build)* réserve les sondes mémoire TTS à la feature tts
- *(enhance)* durcit WS (échéance par trame, origine, pings), h2c loopback et statuts de chargement
- *(enhance)* refuse WebTransport hors loopback et stabilise le test de contre-pression h2
- *(test)* import Cell hors Metal et témoin dense bf16 informatif
- *(build)* réserve l'exemple tts_pcm_digest à la feature tts
- *(tts)* stabilise la relecture des poids et centralise les flags
- *(mtp)* prefill résident MTP sans couche de capture accepté
- *(prefill)* reprises du cache de préfixe identiques au bit au prefill à froid

### Other

- Merge pull request #241 from azerozero/fix/prefix-cache-alignement-propre
- Merge pull request #249 from azerozero/perf/tts-memoire
- Merge pull request #253 from azerozero/feat/transports-flux
- intègre main dans la réduction mémoire TTS
- *(tts)* consigne les mesures mémoire et les écarts de latence
- fusionne origin/main après les mesures TTS
- *(prefill)* hit exact sans queue et écart mesuré sans alignement
- Merge remote-tracking branch 'origin/main' into fix/prefix-cache-alignement-propre
- *(prefill)* marge du froid au token divergent dans l'oracle de préfixe
- *(prefill)* garder l'entrée du prompt complet pour les hits exacts
- *(tts)* ajoute les sondes mémoire et les empreintes PCM
- *(tts)* recharge les poids CPU du codec à la demande
- *(tts)* écarte les poids de l'encodeur du codec
- *(tts)* retire la table texte inutilisée du talker

## [0.3.8](https://github.com/azerozero/saragossa/compare/v0.3.7...v0.3.8) - 2026-11-17

### Added

- *(diar)* intègre main et rend la diarisation optionnelle
- *(diarisation)* diarisation hors ligne Nemotron-3-Diarization

### Fixed

- *(build)* réserve l'attention à clés masquées à stt et diar
- *(diar)* valide les dimensions et borne la lecture WAV

## [0.3.7](https://github.com/azerozero/saragossa/compare/v0.3.6...v0.3.7) - 2026-11-17

### Added

- *(denoise)* branche DFN3 sur la feature dédiée
- *(features)* rend les modalités optionnelles à la compilation

### Fixed

- *(test)* ne lancer que les tokenizers des modalités compilées
- *(features)* fusionne main et isole le débruitage en flux
- *(serve)* refuse les modèles des modalités non compilées
- *(features)* conserve l’ordre de lecture du flag MTP

### Other

- Merge pull request #245 from azerozero/feat/features-modulaires
- fusionne main avec les features modulaires
- *(features)* simplifie les gardes des imports et tests TTS
- *(features)* vérifie les réponses 501 sans modalité
- *(configuration)* décrit les features et les profils TOML
- *(features)* vérifie les combinaisons sur macOS

## [0.3.6](https://github.com/azerozero/saragossa/compare/v0.3.5...v0.3.6) - 2026-11-16

### Added

- *(denoise)* débruite en flux à mémoire bornée

### Fixed

- *(tts)* réinjecter EOS après le filtre top-k/top-p du talker

### Other

- Merge pull request #244 from azerozero/feat/denoise-flux
- Merge pull request #243 from azerozero/release-plz-2026-11-16T22-06-49Z
- Merge pull request #242 from azerozero/fix/tts-wer
- Merge remote-tracking branch 'origin/main' into feat/denoise-flux
- Merge remote-tracking branch 'origin/main' into fix/tts-wer

## [0.3.5](https://github.com/azerozero/saragossa/compare/v0.3.4...v0.3.5) - 2026-11-16

### Added

- *(denoise)* débruitage DeepFilterNet3 sur CPU, saragossa enhance et /v1/audio/enhance

### Fixed

- *(audio)* refuse les fréquences invalides sans panique
- *(denoise)* borne les allocations audio avant traitement

### Other

- Merge pull request #240 from azerozero/release-plz-2026-11-16T21-21-17Z
- *(docs)* une seule ligne vide entre les sections de voice-isolation.md
- Merge remote-tracking branch 'origin/main' into feat/voix-isolation
- *(docs)* retire les lignes vides finales de voice-isolation.md
- Merge remote-tracking branch 'origin/main' into feat/voix-isolation
- *(voix)* corrige la fenêtre de Vorbis
- *(audio)* couvre les limites et les hops incomplets
- *(serve)* fusionne main en conservant le débruitage DFN3
- *(voix)* spécifications DFN3, MossFormer2 et RE-USE, mesures et licences

## [0.3.4](https://github.com/azerozero/saragossa/compare/v0.3.3...v0.3.4) - 2026-11-16

### Added

- *(spec)* classe code et prose sur embeddings e5, en opt-in
- *(spec)* cherche les indices de code en sous-chaînes, toutes langues
- *(spec)* active le routage par classe de prompt par défaut
- *(spec)* route le greedy vers DFlash ou MTP selon la consigne

### Fixed

- *(spec)* écrit l'entrée sans captures si leur recollage échoue
- *(spec)* ne laisse la négation annuler que le radical code
- *(spec)* limite le routage par classe de prompt au greedy
- *(spec)* garde les captures DFlash quand MTP prolonge l'entrée
- *(spec)* ne route vers MTP que la prose sans aucun indice de code

### Other

- Merge pull request #237 from azerozero/fix/lint-docs
- *(release)* publication continue des PR release-plz, comme grob
- *(lint)* corrige markdownlint (pipes de tableau, espaces en code) et shellcheck (shebang bash, SC2155, SC2034)
- Merge pull request #232 from azerozero/release-plz-2026-11-16T18-41-46Z
- Merge remote-tracking branch 'origin/main' into perf/spec-routage
- *(spec)* compare l'historique MTP semé par le prefill DFlash
- *(spec)* corrige la doc de l'oracle d'alternance
- *(spec)* fusionne les deux prefills résidents avec semis MTP
- *(spec)* sème l'historique MTP sur GPU pendant le prefill DFlash
- Merge remote-tracking branch 'origin/main' into perf/spec-routage
- *(perf)* mesure le routage spéculatif par classe de prompt

## [0.3.3](https://github.com/azerozero/saragossa/compare/v0.3.2...v0.3.3) - 2026-11-16

### Added

- *(vision)* insertion des embeddings vision et M-RoPE (étape 3)
- *(stt)* ajouter compression_ratio aux segments verbose_json
- *(stt)* verbose_json, sortie SSE et entrée en flux pour Whisper
- *(tokenizer)* alerte si un Qwen3.5/3.6/3.8 ne porte pas la regex Qwen3.5

### Fixed

- *(serve)* refuser Transfer-Encoding et tout cadrage HTTP ambigu
- *(stt)* émettre le reste du texte final, U+FFFD compris, en fin de deltas
- *(stt)* borner la mémoire et le calcul des sessions en flux
- *(tokenizer)* alerte de regex dédupliquée sur le chemin canonique
- *(tokenizer)* alerte de regex écrite sans paniquer si stderr échoue
- *(quant)* tokenizer.json recopié à l'identique depuis la source
- *(decision)* d3 relu avec la regex Qwen2Tokenizer de son runtime transformers 5.17

### Other

- Merge pull request #235 from azerozero/perf/ecart-mlx
- Merge pull request #234 from azerozero/perf/tri-tirage
- Merge remote-tracking branch 'origin/main' into perf/ecart-mlx
- *(metal)* QMV micro à quatre lignes contre une ligne
- *(metal)* MLP micro fusionné sur tous les octets d'échelle
- Merge pull request #233 from azerozero/feat/vision-mrope-v2
- Merge remote-tracking branch 'origin/main' into feat/stt-streaming
- Merge remote-tracking branch 'origin/main' into perf/tri-tirage
- *(metal)* régularité du balayage top-k jugée après softcap
- *(metal)* cas k = vocabulaire, zéros signés gagnants et 37 groupes
- *(metal)* l'oracle du balayage top-k vérifie le chemin exécuté
- Merge pull request #229 from azerozero/fix/tokenizer-regex
- Merge pull request #208 from azerozero/release-plz-2026-11-15T14-47-08Z
- Merge pull request #225 from azerozero/dependabot/cargo/serde_json-1.0.154
- Merge pull request #223 from azerozero/dependabot/github_actions/github/codeql-action-4
- Merge pull request #222 from azerozero/dependabot/cargo/thiserror-2.0.21
- Merge pull request #220 from azerozero/dependabot/github_actions/actions/setup-python-7
- Merge pull request #217 from azerozero/dependabot/github_actions/actions/attest-build-provenance-4.2.2
- *(stt)* valider les routes STT avant le chargement du modèle
- *(deps)* bump serde_json from 1.0.150 to 1.0.154
- Merge remote-tracking branch 'origin/main' into fix/tokenizer-regex
- *(tokenizer)* alerte de regex unique, chargement poursuivi, stderr défaillant
- *(metal)* QMV micro à quatre lignes en opt-in
- *(stt)* envoyer le dernier morceau dès la fin de l'audio dans le client simulé
- *(metal)* GEMM u4 déroulé, MLP fusionné et QMV 4 lignes en micro-échelles
- *(bench)* longueurs, decode et essais de quant_micro réglables
- *(metal)* sampler top-k GPU par balayage simdgroup
- *(tts)* banc GPU du tirage des codebooks
- *(sampling)* sélection top-k CPU en un balayage au lieu de select_nth
- *(deps)* bump github/codeql-action from 3 to 4
- *(deps)* bump thiserror from 2.0.18 to 2.0.21
- *(deps)* bump actions/setup-python from 6 to 7
- *(deps)* bump actions/attest-build-provenance from 3.0.0 to 4.2.2

## [0.3.2](https://github.com/azerozero/saragossa/compare/v0.3.1...v0.3.2) - 2026-11-16

### Added

- *(vision)* valide les tenseurs et refuse les médias
- *(loader)* fusionner un adaptateur LoRA PEFT au chargement
- *(decision)* lecture de lettres H2O-Lightning, modes bidirectionnel et causal avec cache de préfixe exact
- *(decision)* règles de lecture de runtime.plan par défaut pour THX-02
- *(decision)* moyenne TTA sur les ordres d'options de THX-02
- *(decision)* charge les modèles de décision sans tête de vocabulaire
- *(decision)* prend en charge le format d3-code-readout-v1 de d3-flash
- *(decision)* ajoute l'inférence des modèles de décision THX-02 en texte
- *(quant)* charge les micro-échelles MLX sur CPU et Metal

### Fixed

- *(vision)* préserve le chargement texte des tours inconnues
- *(loader)* refuser une échelle LoRA infinie en f32
- *(loader)* refuser les dimensions LoRA nulles
- *(loader)* identité SSD de l'adaptateur LoRA sur les octets lus
- *(decision)* cache de préfixe borné au GEMM bf16 et au prolongement résident, cause mesurée
- *(decision)* contrat du tokenizer H2O épinglé par son SHA-256
- *(decision)* regex du tokenizer H2O lue par pointeur, sans indexation qui panique
- *(tts)* vérifier les poids de la trame fusionnée avant d'encoder
- *(tts)* garder le préfixe de l'état dans le repli streaming du codec
- *(tts)* décoder le batch au préfixe GPU par lots de 64 trames
- *(tts)* garder le préfixe codec CPU pour le glouton
- *(decision)* borne de 64 questions à la validation, sans indexation
- *(tts)* appliquer la politique QmmNaDequant TTS dans le prefill par embeddings
- *(clippy)* lints de rustc 1.95 venus de main
- *(decision)* refuser un format de décision inconnu
- *(quant)* encoder les empreintes de l'exemple micro sans LowerHex
- *(tts)* vérifier l'exactitude bf16 des tables et remonter l'erreur codec
- *(quant)* adosse les poids micro à la mémoire unifiée sans copie
- *(quant)* valide les entrées micro sans indexation paniquante
- *(quant)* traite un override de couche sans mode comme affine
- *(quant)* compare le vocabulaire KL avec des clés triées
- *(quant)* mesure le KL micro sur les IDs fixes du protocole
- *(quant)* masque les colonnes du GEMM micro hors multiple de 64
- *(serve)* encoder l'empreinte du tokenizer sans LowerHex
- *(deps)* adapter les empreintes SHA-256 à sha2 0.11
- *(tts)* adapter l'ajout des tokens spéciaux à tokenizers 0.23
- *(tts)* réserver l'instruction par défaut à VoiceDesign
- *(release)* rendre le paquet visible de release-plz
- *(serve)* utiliser Serena pour les voix inconnues
- *(tts)* rétablir le mode glouton sans configuration
- *(ci)* permet de taguer un commit de release précis au dispatch
- *(quant)* conserve le repli des projections de modes mixtes
- *(tts)* exposer la raison effective de fin de synthèse
- *(tts)* échantillonner selon la configuration du modèle
- *(tts)* insérer le locuteur CustomVoice demandé
- *(ci)* attacher main avant release-plz release et permettre la relance manuelle
- *(serve)* vérifie le préfixe émis en debug
- *(serve)* réserve le diagnostic aux flux dégradés

### Other

- Merge pull request #219 from azerozero/dependabot/github_actions/release-plz/action-0.5.148
- Merge pull request #218 from azerozero/dependabot/cargo/proptest-1.12.0
- Merge pull request #228 from azerozero/feat/vision
- Merge remote-tracking branch 'origin/main' into feat/lora-fusion
- fusionne origin/main dans feat/vision
- *(vision)* dédoublonne les fixtures et recalcule les patchs
- *(loader)* base LoRA « non quantifiée », pas seulement bf16
- *(loader)* chemin THX-02 du test LoRA par variable d'environnement
- Merge pull request #216 from azerozero/feat/decision-letters
- *(vision)* fige l’inventaire et les références Qwen
- *(decision)* corriger les listes et le code inline signalés par markdownlint
- *(tts)* profil codec sans indexation de HashMap
- *(tts)* préciser les constantes de tuile et leurs limites
- *(tts)* comparer tuiles et gather au bit sur poids synthétiques
- *(deps)* bump release-plz/action from 0.5.131 to 0.5.148
- *(deps)* bump proptest from 1.11.0 to 1.12.0
- *(decision)* cache de préfixe au bit sur plusieurs morceaux et sur suffixe borné
- *(decision)* tokens spéciaux et neutralisation partagés par les deux lecteurs
- Merge remote-tracking branch 'origin/main' into feat/decision-letters
- *(decision)* mesures d3-mini et H2O sous charge basse, cache de préfixe causal
- *(tts)* convolutions du codec GPU en tuiles, conv2 et résiduel fusionnés
- *(tts)* resserrer l'écart du préfixe codec GPU à un pas int16
- *(decision)* oracle d3-mini sur poids réels, tolérance bf16 figée
- Merge remote-tracking branch 'origin/main' into perf/tts-cp-fusion
- Merge pull request #214 from azerozero/feat/decision-tta
- Merge pull request #213 from azerozero/fix/tts-prefill-extend-identity
- Merge remote-tracking branch 'origin/main' into feat/decision-tta
- *(prefill)* oracle synthétique de l'empaquetage des décisions
- *(decision)* exiger l'égalité au bit des lectures empaquetées
- *(tts)* sélectionner les 64 premiers rangs du tirage sans tri complet
- *(tts)* décoder le préfixe du codec sur GPU
- *(tts)* encoder les couches du code predictor en 8 dispatches
- *(tts)* compter les dispatches par site source et par phase
- Merge pull request #212 from azerozero/feat/decision-d3
- Merge pull request #211 from azerozero/perf/tts-frame-batch
- Merge remote-tracking branch 'origin/main' into feat/decision-d3
- *(decision)* parcourt le GEMM bf16 des décisions par tuiles N d'abord
- Merge remote-tracking branch 'origin/main' into feat/quant-mxfp
- Merge remote-tracking branch 'origin/main' into perf/tts-frame-batch
- *(decision)* référence f32 d3-flash et tolérance bf16 figée
- *(tts)* décoder le codec streaming en recouvrement des trames
- *(tts)* enchaîner les 16 pas du code predictor dans un command buffer
- *(tts)* profiler chaque trame par phase avec horodatage GPU
- Merge remote-tracking branch 'origin/main' into feat/decision-models
- *(decision)* documenter la neutralisation élargie des tokens spéciaux
- *(quant)* décode l'échelle micro par bloc dans la tuile GEMM
- *(quant)* décode l'échelle micro une fois par mot dans le QMV
- Merge pull request #207 from azerozero/fix/tts-customvoice-sampling
- Merge remote-tracking branch 'origin/main' into fix/tts-customvoice-sampling
- Merge remote-tracking branch 'origin/main' into fix/deps-sha2-0.11-msrv
- Merge pull request #189 from azerozero/dependabot/cargo/tokenizers-0.23.2
- *(tts)* projeter les têtes échantillonnées dans le pas résident
- Merge pull request #184 from azerozero/dependabot/cargo/metal-0.33.0
- *(deps)* verrouiller sha2 0.11.0 pour conserver la MSRV 1.85
- Merge pull request #206 from azerozero/test/qualimetrie
- *(deps)* bump metal from 0.29.0 to 0.33.0
- Merge pull request #204 from azerozero/fix/release-plz-dispatch-sha
- Merge pull request #203 from azerozero/perf/serve-detok-incremental
- Merge remote-tracking branch 'origin/main' into test/qualimetrie
- *(ci)* partage le verrou GPU et tolère l'absence de CustomVoice
- Merge remote-tracking branch 'origin/main' into fix/tts-customvoice-sampling
- *(quant)* vérifie les produits E8M0 sous-normaux sur CPU
- *(quant)* décrit les micro-échelles MLX et leur validation
- *(quant)* ajoute les mesures corpus et vitesse locales
- *(mutants)* contrôler les tirages et la conversion PCM
- *(proptest)* versionner le journal des régressions
- *(embeddings)* ajouter un oracle synthétique CPU et HTTP
- Merge pull request #201 from azerozero/fix/release-plz-tag-branch
- *(tts)* couvrir les options de voix et de graine
- Merge remote-tracking branch 'origin/main' into perf/serve-detok-incremental
- *(whisper)* couvrir les fenêtres longues et le silence
- *(tests)* relier les contrats aux preuves de qualité
- *(fuzz)* exercer les parseurs publics en nightly CPU
- *(nightly)* exiger les modèles réels sous verrou GPU
- *(e2e)* mesurer la durée audio et les erreurs de mots
- *(serve)* reprend ByteLevel après une frontière stable
- *(tokenizer)* couvre les added tokens ordinaires
- *(deps)* bump sha2 from 0.10.9 to 0.11.1
- Merge remote-tracking branch 'origin/main' into perf/serve-detok-incremental
- *(deps)* bump tokenizers from 0.22.2 to 0.23.2
- *(serve)* détokenise les flux ByteLevel de façon incrémentale

## [0.3.1](https://github.com/azerozero/saragossa/compare/v0.3.0...v0.3.1) - 2026-11-15

### Added

- *(devtools)* logits de référence et divergence KL contre un run de référence
- *(bench)* perplexité MLX sur un checkpoint choisi
- *(bench)* fractions mémoire de llmd fixées à 0,5 × 0,5
- *(bench)* contrôle court de llmd (mémoire et prompt_tokens)
- *(bench)* llmd natif classé non supporté, erreur publiée
- *(bench)* conventions EOS fixées par l'essai de validation
- *(bench)* sonde de cache llmd par chargement, même document et nonce différent
- *(bench)* sondes EOS à arrêt naturel pour la validation
- *(bench)* rendu jinja vérifié contre le ChatML de saragossa
- *(bench)* harnais du bench concurrents Qwen3.8-27B
- *(serve)* empreinte des ids du prompt dans le journal de complétion
- *(serve)* refus explicite de min_p et des pénalités non neutres
- *(serve)* usage en fin de flux avec stream_options.include_usage
- *(serve)* chat_template_kwargs.enable_thinking respecté
- *(serve)* graine du sampler lue dans la requête chat
- *(decode_resident)* rejeu sans tête, point de reprise de session et bench Flash-Next à 32k
- *(decode_resident)* QSA creuse de Flash-Next, indexeur résident, top-512 radix exact et gather
- *(decode_resident)* profil GPU par brique du decode Flash-Next (devtools)
- *(decode_resident)* reset de session Flash-Next, débit mural répété dans le harnais
- *(decode_resident)* session résidente Flash-Next ≤ 2051 tokens et oracle N2 incrémental
- *(decode_resident)* attention QSA résidente de Flash-Next en régime dense
- *(decode_resident)* PLE résident de Flash-Next, lectures n-gram parallèles avant commit
- *(decode_resident)* hyper-connections résidentes de Flash-Next
- *(linear_attention)* variante Flash-Next résidente, normes l2 de q/k et gate σ
- *(qwen4_exp)* loader q4e8 et référence CPU f32 sans cache, oracle N1
- *(linear-attn)* variante CPU Flash-Next, norme l2 conforme et gate sigmoïde
- *(devtools)* convertisseur en flux qwen4_exp FP8 vers MLX affine mixte q4e8
- *(verify)* arbre DFlash à B = 3 par défaut en mode strict
- *(verify)* mode greedy strict opt-in avec qmv-T (SARAGOSSA_RUST_VERIFY_STRICT)
- *(metal)* kernels qmv-T u4 gs64 exacts en R lignes
- *(verify)* politique NAX par requête pour la vérification greedy
- *(serve)* tête MTP MoE en greedy sur serve, la CLI et le REPL
- *(mtp)* vérification MoE en lignes active avec une tête MTP MoE
- *(mtp)* tête MoE résidente et cycles D1/D2 fusionnés sur tronc MoE
- *(mtp)* chargement de la tête MTP MoE officielle du 35B-A3B
- *(dflash)* échantillonné par défaut quand un drafter est configuré
- *(dflash)* vérification échantillonnée exacte sous SARAGOSSA_RUST_DFLASH_SAMPLED
- *(cli)* option --stop répétable
- *(serve)* requêtes avec stop sur le chemin spéculatif
- *(mtp)* stop sequences dans le cœur MTP
- *(dflash)* stop sequences dans la boucle résidente (chaîne et arbre)
- *(decode)* boucle greedy lookup n-grammes sans tête MTP
- *(verify)* couches full-attn MoE en lignes pour la vérification résidente
- *(dflash)* arbre de vérification jusqu'à B=15 et oracle CPU f32
- *(dflash)* vérification en arbre opt-in (SARAGOSSA_RUST_DFLASH_TREE=B)
- *(dflash)* constructeur CPU de l'arbre de vérification DFlash 2
- *(serve)* lots multi-flux opt-in pour les chats streamés greedy
- *(serve)* concurrence d'E/S avec thread moteur unique
- *(decode)* base de K/V borné dans le cache CPU et l'arène
- *(metal)* pool et buffers K/V placement sparse, transferts explicites
- *(decoder)* ouvrir le light-batch à flint en time-slicing
- *(flint)* template de chat flint pour serve et le CLI
- *(flint)* prefill Metal résident avec sinks, NoPE et MoE sigmoïde
- *(flint)* decode Metal résident avec sinks, NoPE et MoE sigmoïde
- *(flint)* charger RecursiveSI/flint et l'exécuter sur l'oracle CPU
- *(kv)* câblage du format dans le setup résident, flag SARAGOSSA_RUST_KV_QUANT (défaut bf16)
- *(kv)* attention 2-passes sur KV quantifié (d128/d256)
- *(kv)* écritures quantifiées (seed, append, lignes) sur GPU
- *(kv)* codec affine Q8/Q4 et oracle CPU
- *(mtp)* routage opt-in des requêtes échantillonnées vers le decode MTP lossless
- *(decoder)* drafter DFlash opt-in dans le chemin greedy MTP, boucle résidente streamée
- *(pairwise)* mode dflash dans le harnais V5, drafter chargé par le CLI pairwise
- *(dflash)* bascule par acceptance observée vers la boucle MTP dans le harnais
- *(dflash)* boucle DFlash avec drafter résident et vérification en lignes
- *(dflash)* quantification affine du draft et bloc DFlash 2 résident
- *(devtools)* drafter DFlash 2 sur CPU et cycle D2 fusionné segmenté sous compteurs GPU
- *(pairwise)* harnais V5 (contrôle ligne de matrice) dans l'arbre et plafond de draft par cas
- outils de quantification mixte, fixtures voix et préfixe SARAGOSSA_

### Fixed

- *(tts)* réserve QmmNaDequant à une activation explicite
- *(metal)* ajoute le repli u5 et retire une barrière doublée
- *(loader)* accepte la quantification globale u5
- *(decode_resident)* accepter le qmv u5 dans le decode résident
- *(deps)* quinn-udp 0.5.16 pour RUSTSEC-2026-0538
- *(build)* fins de ligne LF pour les textes inclus par include_str!
- *(devtools)* impose le chemin résident et corrige le top-1 KL
- *(serve)* normalise le PCM 24 bits sur sa pleine échelle
- *(serve)* rejette les WAV hostiles avant allocation
- *(bench)* paths.sh conforme à shellcheck
- *(bench)* récupère la dernière ligne JSONL interrompue
- *(bench)* exige une divergence prouvée pour le repli du gel
- *(bench)* conserve la requête lors des erreurs HTTP
- *(serve)* rend l’empreinte du prompt explicitement optionnelle
- *(serve)* distingue les identifiants des flux simultanés
- *(serve)* refuse le thinking actif avant le chargement
- *(serve)* refuse les pénalités non gérées avec HTTP 400
- *(serve)* masque les valeurs des erreurs de validation
- *(serve)* respecte les options usage du flux SSE
- *(serve)* accepte les graines signées
- *(bench)* réponse vide jamais admissible
- *(bench)* marque ≈ calculée par groupe de tableau
- *(bench)* famille de réglage entièrement divergente de l'AR, défaut gardé
- *(bench)* événements de plusieurs tokens comptés exactement
- *(bench)* prose wikitext détokenisée pour un découpage identique chez llmd
- *(bench)* version de llama-server lue sur stderr dans env
- *(bench)* préfixe <think> injecté par llama-server retiré avant comptage
- *(bench)* reprise de nuit et environnement maîtrisé
- *(bench)* identité greedy obligatoire et preuves d'analyse resserrées
- *(bench)* références AR du réglage, gel resserré, arrêt et reprise
- *(bench)* rusage distingue processus disparu et lecture refusée
- *(bench)* admissibilité, comptage par événement et reprise idempotente
- *(bench)* contrôles d'entrée et de qualité bloquants
- *(bench)* erreur de flux terminale et fragments bruts conservés
- *(serve)* un seul id de chunk par flux SSE OpenAI
- *(serve)* enable_thinking=true refusé tant que son rendu n'est pas qualifié
- *(bench)* santé llmd par /v1/models ou /health
- *(dflash)* décision de bascule lue une fois pour le prefill et le semis MTP
- *(dflash)* all_rows ignoré hors Metal sans avertissement
- *(dflash)* captures du prefill résident bornées à la fenêtre du drafter
- *(rope)* partager les fréquences f32 et préciser les rotations
- *(decode_resident)* RoPE Flash-Next précise, oracle ré-ancré et seuil QSA calibré
- *(decode_resident)* filiation des points de reprise, sélection QSA native justifiée, routage CPU propre conservé
- *(decode_resident)* reprise avec clés brutes de l'indexeur, mutation QSA et RoPE longue sondées
- *(decode_resident)* injection hc en tranches couvrantes, uint4 aligné, norme à 1024 threads vérifiée
- *(decode_resident)* budget de session avec instantanés DeltaNet, reset empoisonné, mutation N2 tardive et porte de débit bloquante
- *(decode_resident)* garde mémoire Flash-Next au pic de chaque étape, globaux et états compris
- *(decode_resident)* session Flash-Next restaurée sur échec d'encodage, empoisonnée sur erreur GPU
- *(decode_resident)* config Flash-Next validée à l'ouverture, borne dense tirée du budget QSA
- *(decode_resident)* KV Flash-Next épinglé en f32, hors overrides bf16 et quantifié
- *(qwen_loader)* table n-gram limitée à un chemin relatif sous ngram/
- *(qwen4_exp)* bits bornés et métadonnées n-gram en arithmétique vérifiée
- *(qwen4_exp)* inventaire MTP complet sur métadonnées
- *(qwen4_exp)* préconditions de config séquentielles, group_size nul refusé, gate silu
- *(convert)* destination ambiguë, en-tête corrompu, ple_layer_ids strict
- *(convert)* pic mémoire échantillonné blocs encore vivants
- *(convert)* provenance au manifeste, couche PLE lue dans la config
- *(quant)* étendue affine en f64, codes par division, restitution finie
- *(convert)* validation des plages source, de l'échelle n-gram et des configs dégénérées
- *(convert)* refus d'une conversion sur place, liens symboliques compris
- *(convert)* reprise n-gram contrôlée sans --verify
- *(verify)* rejeux après rejet dans la portée greedy de leur vérification
- *(verify)* portée greedy de requête limitée aux vérifications spéculatives
- *(verify)* qmv-T seulement là où le mono prend le qmv rapide
- *(verify)* arbre DFlash à B = 3 en greedy strict seulement
- *(verify)* GEMM NA exclus des deux sélecteurs en mode strict
- *(verify)* politique greedy de la requête, quelle que soit la tête
- *(verify)* qmv-T à 16 lignes en variante device, skinny exclu en strict
- *(metal)* contraction coupée dans la réduction rms des prologues u3 et u8
- *(metal)* contrat canonique du qmv u4 gs64 entre mono et lignes
- *(kernels)* contrat contract(off) sur le scan inverse, fixtures sans espaces finaux
- *(verify)* norme QKV en miroir du mono dense, prédicat unique
- *(verify)* attention en arbre bit-identique au pas AR du chemin
- *(verify)* vérification en lignes bit-identique au mono jusqu'à 4 lignes
- *(prefill)* la garde du morceau court ne fait que sonder la clé arrondie
- *(prefill)* scratch du dernier morceau dimensionné sur la capacité du morceau
- *(cli)* la chauffe MTP ne laisse plus de préfixe en cache
- *(assets)* override MTP incomplet refusé, load_local isolé en test
- *(mtp)* head_dim déduit de q_proj sans surcharge de la config
- *(mtp)* tête MoE hors contrat résident, repli sans arène MTP
- *(mtp)* formes de la tête validées contre le tronc avant le GPU
- *(assets)* override MTP dans runtime_flags, refus typé et tests isolés
- *(mtp)* verify fusionné D1/D2 exact sur le tronc MoE du 35B
- *(devtools)* ablation de vérification confinée au bench
- *(serve)* arrêt sur stop texte aussi sans streaming
- *(mtp)* sortie tronquée au dernier token émis sur annulation
- *(serve)* lot ouvert aux requêtes de session avec stop servies en MTP
- *(serve)* arrêt du decode sur stop texte en streaming
- *(decode)* lookup greedy inactif sous plafond de débit
- *(verify)* bras LinearMoe sous PREFILL_MOE_ROWS et queue coop retirée
- *(dflash)* corrections de revue de la vérification en arbre
- *(lightbatch)* garder la signature publique du callback streaming
- *(serve)* validation avant chargement et fin par flux dans les lots
- *(serve)* compare-exchange au lieu de fetch_update déprécié
- *(serve)* servir les clients en demi-fermeture
- *(serve)* bearer avant le corps, budget d'octets et clients partis
- *(lightbatch)* exiger v_proj pour le pas multi-flux et oracle strict
- *(linear)* verrouiller le mode journal hors du scan fusionné
- *(verify)* gardes et compilation paresseuse du préfixe NAX en lignes
- *(matmul)* biais du skinny x2 sur l'activation scindée
- *(prefill)* appliquer les relectures du x3 k/v Qwen
- *(decode)* réserver l'append fusionné à flint
- *(decode)* appliquer les relectures des dispatches flint
- *(prefill)* appliquer les relectures du MoE coop flint
- *(prefill)* appliquer les relectures du K/V fenêtré
- *(decode)* réserver l'adoption fenêtrée aux couches à sinks
- *(prefill)* appliquer les relectures du K/V bf16 flint
- *(kv)* appliquer les relectures de la vivacité des vues
- *(metal)* garder les heaps d'un buffer perdu même si le pool est empoisonné
- *(metal)* appliquer les relectures du K/V sparse
- *(decoder)* appliquer les relectures de l'éviction et du light-batch
- *(decoder)* évincer les copies remplacées du cache de préfixe
- *(flint)* appliquer les relectures du prefill Steel
- *(flint)* appliquer les relectures du prefill Metal
- *(flint)* appliquer les relectures du decode Metal résident
- *(flint)* appliquer les relectures du chargeur et de l'oracle CPU
- *(ci)* retirer les usages dépréciés par le toolchain stable
- *(kv)* retours de relecture Codex et Fable
- *(prefill)* vues K/V partagées comptées par allocation, contrat unsafe explicite
- *(mtp)* semis GPU validé et concat K|V bâti avant libération CPU
- *(mtp)* refus au premier token sans semis, doc du flag recalée
- *(decoder)* KV bf16 décidé par la génération, garde d'extension alignée sur l'exécution
- *(serve)* segments KV partagés comptés une fois, fenêtre CPU sans parcours
- *(prefill)* dernier morceau vérifié, seq vide refusé, un morceau sous timestamps GPU
- *(serve)* snapshots de préfixe linéaires et bornés en octets
- *(prefill)* le MLP fusionné revérifie son prédicat avant tout dispatch
- *(prefill)* MLP fusionné réservé aux poids sans offset
- *(serve)* routage des requêtes échantillonnées aligné sur l'opt-in MTP, politique testée dans les deux états
- *(prefix-cache)* seuil de recalcul complet réservé aux runtimes Metal
- *(prefix-cache)* extension résidente conditionnée à la porte d'attention et au support résident, trace sans Metal
- *(dflash)* hit du cache de préfixe refusé sans couverture de la fenêtre, oracle d'extension indépendant
- *(decoder)* stop token accepté comme draft DFlash exclu de la sortie
- *(dflash)* bascule refusée sans sidecar MTP et profondeur MTP strictement positive
- *(dflash)* module résident sous Metal, barrières des kernels et validation du quantificateur
- *(metal)* capacité KV vérifiée avant l'append en lignes et blocs split-K par ligne aux frontières
- *(mtp)* lookup réservé au greedy et ligne primaire écrite dans l'historique committed
- *(deps)* corriger les avis quinn-proto, rustls et chacha20
- *(mtp)* préserver le chemin fusionné pendant le profilage
- *(deps)* corriger la vulnérabilité TLS de rustls
- *(qwen)* prendre en charge Qwen3.8 et détecter la tête MTP

### Other

- Merge pull request #187 from azerozero/dependabot/cargo/md5-0.8.1
- Merge pull request #192 from azerozero/perf/gemma4-prefill-resident
- Merge pull request #186 from azerozero/dependabot/github_actions/github/codeql-action/init-4.39.3
- *(deps)* aligner codeql-action/analyze sur init en v4.39.3
- *(deps)* bump github/codeql-action/init from 3.39.3 to 4.39.3
- Merge pull request #176 from azerozero/ci/parite-grob
- *(perf)* corrige les erreurs markdownlint existantes
- utilise l’action officielle actionlint
- *(sécurité)* épingle checkout et CodeQL sur leurs commits
- *(prek)* préserve les fixtures et les données de référence
- précise la copie du préambule et la compilation au runtime
- *(rust)* retire la compilation redondante avant les tests
- déclenche CodeQL et les tests GPU dans la merge queue
- *(codeql)* filtre les PR sans code avec un statut de remplacement
- Merge remote-tracking branch 'origin/main' into ci/parite-grob
- Merge pull request #173 from azerozero/fix/kernels-bugs-latents
- *(metal)* qualifie la surcharge du kernel de déquantification
- Merge remote-tracking branch 'origin/main' into fix/kernels-bugs-latents
- Merge pull request #171 from azerozero/fix/bf16-qwen35-charabia
- Merge pull request #170 from azerozero/perf/qmv-u5
- *(metal)* précise le cache scratch et les oracles TTS u6
- *(metal)* couvre toutes les formes u6 avec l’oracle CPU
- *(decode_resident)* aligne l’oracle sur le support u5
- *(mtp)* précise les limites du chargement inline mixte
- *(metal)* vectorise le qmv u5/u6 et ouvre le prefill NAX
- *(metal)* ajoute le qmv affine MLX 5 bits gs64
- *(goldens)* précise l'exception Gemma 4 et la raison du critère de top-1
- *(goldens)* oracle KL Gemma 4 en régime chat au lieu du texte brut
- *(e2e)* json_object vérifie la forme JSON, plus le contenu
- Merge pull request #166 from azerozero/feat/quant-gemma4
- Merge branch 'main' into feat/quant-gemma4
- Merge remote-tracking branch 'origin/main' into feat/endpoints-responses
- Merge pull request #164 from azerozero/perf/mtp-historique-bf16
- Merge pull request #163 from azerozero/test/goldens-reference
- Merge pull request #161 from azerozero/fix/serve-erreurs
- Merge pull request #158 from azerozero/fix/gemma4-rope-proportionnelle
- *(gemma4)* budget de 80 tokens pour la réponse française cohérente
- Merge origin/main dans fix/gemma4-rope-proportionnelle
- Merge pull request #157 from azerozero/perf/dflash-captures
- Merge pull request #154 from azerozero/perf/chargement-sans-copie
- Merge pull request #160 from azerozero/test/e2e-conformite
- Merge pull request #159 from azerozero/fix/templates-outils
- Merge pull request #155 from azerozero/perf/mtp-historique-unique
- Merge remote-tracking branch 'origin/main' into refactor/model-plan
- *(portable)* limite deux goldens audio à aarch64 et le blob creux à Unix
- Merge remote-tracking branch 'origin/main' into build/portable-linux-windows
- Merge branch 'main' into build/portable-linux-windows
- Merge branch 'main' into test/goldens-tokenizer
- Merge pull request #149 from azerozero/feat/instantane-etat
- Merge pull request #150 from azerozero/feat/kl-reference
- *(devtools)* échappe les crochets de la doc de --kl-record
- Merge pull request #136 from azerozero/fix/mtp-memoire-256k
- Merge pull request #132 from azerozero/fix/poids-gdn-offset-z
- Merge branch 'main' into fix/poids-gdn-offset-z
- Merge pull request #143 from azerozero/fix/tts-voix-par-defaut
- Merge remote-tracking branch 'origin/main' into fix/tts-voix-par-defaut
- Merge pull request #135 from azerozero/fix/audio-limites-silencieuses
- Merge pull request #144 from azerozero/perf/memoire-ar-256k
- Merge pull request #141 from azerozero/perf/poids-denses-bf16
- Merge pull request #140 from azerozero/perf/tts-qmm-u6
- Merge pull request #139 from azerozero/perf/tts-prefill-memoise
- Merge pull request #138 from azerozero/fix/profil-kernels-par-modele
- Merge pull request #137 from azerozero/fix/dflash-chargeur-sans-copie
- Merge pull request #134 from azerozero/refactor/shared-rows-unsafe
- Merge pull request #131 from azerozero/fix/wav-entree-hostile
- Merge branch 'main' into feat/bench-ab
- Merge pull request #127 from azerozero/feat/gpulock
- *(quant)* convert_ls et opt_quant dans le README
- *(bench)* pause de 20 s entre configurations
- *(bench)* données de la nuit u4ls du 9 novembre
- Merge pull request #125 from azerozero/feat/bench-prompts-disjoints
- *(gpu)* tests GPU synthétiques sur les PR, tests lourds la nuit
- *(metal)* exceptions rows_exact conditionnées au pipeline skinny
- *(gpu)* job de mesure des tests GPU à poids synthétiques
- Merge pull request #118 from azerozero/feat/qwen4exp-mtp
- *(bench)* données brutes du bench concurrents Qwen3.8-27B
- *(dflash)* plancher d'acceptance de l'arbre par défaut et commentaires précisés
- *(dflash)* acceptance du résident comparée à la boucle CPU en chaîne
- *(serve)* précise le rendu natif sans thinking
- *(bench)* reprise d'une nuit interrompue
- *(bench)* local.env, convention EOS et gel dans le README du harnais
- *(bench)* chemins locaux hors du code, paths.sh seule source
- *(serve)* TCP_NODELAY sur les connexions acceptées
- *(serve)* champs de sampling, enable_thinking, include_usage et empreinte
- *(dflash)* prefill sans relecture des états finaux hors bascule MTP
- *(dflash)* gardes du drafter à base fixe simplifiées, tests de capacité et de refus précis
- *(dflash)* KV du drafter à base fixe, dimensionné sur la fenêtre
- *(sampling)* oracle top-k élargi à tous les k, ex aequo totaux et -inf
- *(sampling)* top-k par sélection partielle au lieu du tri du vocabulaire
- *(rope)* tests GPU sous la convention du dépôt, tolérance de 2 ulp sur cos/sin CPU
- *(rope)* résoudre les fréquences au setup résident
- *(perf)* résultats du N2 résident à 4096 tokens
- *(decode_resident)* capture native retirée, couverture creuse exigée, justification du routage sans NaN
- *(decode_resident)* code mort de l'imposition QSA retiré, doc de l'étape 5 corrigée
- *(perf)* QSA creuse Flash-Next, oracles ré-ancrés, RoPE précise et débit à 32k
- *(decode_resident)* portes N2 selon l'arbitrage, oracle ré-ancré strict et dérive publiée
- *(decode_resident)* seuil QSA calibré plafonné à 1e-4
- *(qwen4_exp)* couches QSA ré-ancrées sur les entrées torch du golden long
- *(decode_resident)* session Flash-Next à capacité maximale sous budget
- *(decode_resident)* N2 QSA creuse, sélection imposée aux quasi-égalités, CPU contre torch sur 4096 tokens
- *(perf)* débit résident Flash-Next, porte verte à 46,3 tok/s et profil par brique
- *(decode_resident)* routage du top-k large et repli du Down hc testés, invariants documentés
- *(decode_resident)* Down des hyper-connections Flash-Next en split-K, temps d'encodage mesuré
- *(metal_backend)* top-k softmax large pour plus de 256 experts, identique bit à bit
- *(decode_resident)* hyper-connections Flash-Next, norme groupée large et injection en tranches
- *(decode_resident)* bench Flash-Next refuse 0 répétition, reset partiel testé hors saturation
- *(perf)* porte native Flash-Next justifiée, mutation tardive, budget de session et reset
- *(perf)* N2 résident Flash-Next, contrôle natif, seuil 1e-6 et contrat de session
- *(decode_resident)* passe native N2 contrôlée jusqu'à la première bascule, NaN refusés, seuil d'imposition le plus strict
- *(decode_resident)* allow ciblés dans le module Flash-Next, ε l2 documenté dans le kernel
- *(metal_backend)* top-k imposé des oracles compilé seulement pour les tests
- *(perf)* porte N2 résidente à routage imposé, verte sur 2051 tokens
- *(decode_resident)* mutation N2 sur la plus grande marge, point du pire écart imposé
- *(decode_resident)* N2 résident à routage imposé aux quasi-égalités, témoin natif et mutation
- *(perf)* N2 résident Flash-Next, porte rouge au-delà de 88 tokens par bascule de routage
- *(decode_resident)* mesure du débit AR résident Flash-Next après 1k tokens
- *(metal_backend)* MoE 512/top-10 et matmuls résidents aux formes de Flash-Next
- *(tools)* manifeste exigé hors meta.json, chemin n-gram borné, publication restaurable
- *(qwen4_exp)* provenance n-gram et refus avant écriture des goldens N2
- *(qwen4_exp)* provenance N2 sur ngram/meta.json et la table désignée
- *(tools)* compatibilité de provenance vérifiée avant écriture, table n-gram désignée
- *(perf)* provenance N2 par golden, recalculée sur les fichiers ouverts
- *(qwen4_exp)* provenance N2 vérifiée sur les fichiers réellement ouverts
- *(tools)* provenance propre à chaque golden N2, mélange refusé
- *(linear-attn)* forward Flash-Next contre une référence naïve, gates σ et SiLU
- *(perf)* référence CPU Flash-Next, quasi-égalités justifiées par l'oracle
- *(tools)* oracles Flash-Next reproductibles dans tools/flashnext
- *(qwen4_exp)* quasi-égalités justifiées par l'oracle, passe chaînée native
- *(qwen4_exp)* visibilité pub(crate) des modules math et ngram
- *(perf)* référence CPU Flash-Next, oracles N1 et N2, coûts mesurés
- *(qwen4_exp)* oracle N2 sur les poids q4e8 v2, quasi-égalités explicites
- *(perf)* conversion q4e8-v2, erreurs exactes, reprise et changement des loaders FP8
- *(convert)* experts multi-blocs, lm_head u6 relu en valeurs, sources invalides
- *(loader)* forme des échelles FP8 par le cœur commun
- *(perf)* conversion Flash-Next q4e8, format, coût et qualité contre FP8
- *(verify)* couverture obligatoire de chaque mode O2 sélectionné
- *(verify)* mtp-skip par la boucle MTP directe, couverture des suffixes de reprise
- *(verify)* preuve de reprise du préfixe dans l'oracle de réutilisation
- *(verify)* modes O2 mtp-skip et dflash-cpu à couverture obligatoire
- *(verify)* oracle chaîné du qmv-T selon SARAGOSSA_RUST_QMV_FAST
- *(verify)* O2 exige le repli Skip multiligne et la largeur d'arbre du défaut
- *(verify)* lm_head en threadgroup S = 16, périmètre du mode strict
- *(verify)* oracles du mode strict de 2 à 32 lignes, chaîne et entrée publique
- *(verify)* mode strict SARAGOSSA_RUST_VERIFY_STRICT
- *(verify)* oracles O1 du mode strict qmv-T
- *(metal)* oracle du contrat qmv sur align64 et masquage partiel
- *(perf)* contrat canonique du qmv, bilan O3, ABBA et Gemma
- *(metal)* oracle du contrat qmv sur kernel intact, queue et miroirs rms u3/u8
- *(verify)* renommer la graine des données de test (faux positif CodeQL)
- *(verify)* oracle de couche dense mono contre lignes, QKV concaténé
- *(verify)* O2 n'accepte que les lots réellement passés en lignes
- *(readme)* limites réelles de l'identité greedy
- *(verify)* nettoyage de l'attention en arbre
- *(verify)* O2/O3 exigent une couverture réelle par mode
- *(verify)* O1 borné, passe stricte sans skinny, QKV fusionné
- *(perf)* identité greedy après correction de l'arbre
- *(verify)* cas prose32k-1 hors dépôt, empreinte vérifiée
- *(perf)* identité greedy, corrections et écarts restants
- *(verify)* oracles d'identité greedy en lignes contre le mono
- *(readme)* identité greedy spéculatif bornée aux quasi-égalités
- *(perf)* DFlash 2 sur le 35B-A3B, débit, écarts avec l'AR et sampler top-p
- *(prefill)* ABBA de la garde par sondage sur 4097, 5078 et 5079 tokens
- *(prefill)* garde du morceau court, sondage sans allocation arrondie
- *(prefill)* scratch du dernier morceau, mesures ABBA sur 5078 et 5079 tokens
- *(prefill)* le morceau court reprend la scratch du morceau plein
- *(mtp)* tête MoE single-copy couverte en bf16, via with_mtp_sidecar et en erreur
- *(mtp)* contrat de quantize_affine_into et erreurs des experts explicités
- *(mtp)* empreinte de la tête du 35B-A3B attribuée et réduite
- *(mtp)* tête MoE du 35B-A3B en une seule copie Metal
- *(mtp)* emplacement de la tête, contexte mémoire et graine serve
- tête MTP du 35B-A3B exposée en greedy, débit, RAM et matrice serve
- *(mtp)* oracles MoE stricts sur leurs prompts, seuils avec marge
- *(perf)* tête MTP officielle sur le 35B-A3B, portes G0 à G5
- *(mtp)* débit échantillonné de la tête MoE contre l'AR
- *(perf)* lookup greedy de bout en bout avec la queue MoE en lignes
- *(moe)* portée des réglages de la queue de vérification en lignes
- *(perf)* queue MoE de vérification calquée sur le mono-ligne et dédup écartée
- *(devtools)* ablation par poste du pas de vérification résidente
- *(linear-attn)* projection complète en un matmul dans la vérification en lignes
- *(moe)* oracles bit à bit et oracle de vérification en lignes renforcé
- *(moe)* queue de vérification en lignes calquée sur le mono-ligne
- *(dflash)* identité exigée et arrêts de production en échantillonné
- *(perf)* DFlash échantillonné, oracles et pairwise à T = 0,7
- *(perf)* arrêt sur stop texte streamé ou non, sortie à l'annulation
- *(serve)* stop texte dans le lot sans séquence d'ids
- *(perf)* section des stop sequences au niveau des campagnes
- *(perf)* mesures serve, A/B sans stop et compteurs MTP
- *(spec)* frontières de tour, annulation et flux échantillonné des stop sequences
- *(perf)* stop sequences sur le chemin spéculatif
- *(mtp)* oracle AR des stop sequences sans drafter
- *(dflash)* oracle AR des stop sequences, chaîne, streaming et arbre
- Merge pull request #100 from azerozero/perf/moe-verify-rows
- *(decode)* scénarios du lookup greedy prouvés par compteurs
- *(perf)* queue coop de la vérification MoE retirée
- *(decode)* oracle bloquant du lookup greedy contre l'AR
- *(verify)* oracle MoE en lignes, bench ρ(R) et mesures de spéculation
- *(dflash)* arbre de vérification B=8 par défaut quand le drafter est chargé
- *(perf)* mesure pairwise de la vérification en arbre DFlash 2
- *(perf)* DFlash contre MTP selon le contexte, mesures pairwise
- *(serve)* client parti détecté à l'écriture
- *(lightbatch)* pas multi-flux dense à N flux pour les modèles hybrides
- *(linear)* extraire le cœur par ligne du pas linear-attn résident
- *(perf)* mesures pairwise des captures SSM en journal
- *(linear)* captures SSM en journal de pseudo-valeurs, rollback par replay
- *(dflash)* préciser la portée de la trace d'arbre et du bench par lignes
- *(dflash)* trace diagnostique de vérification en arbre et bench du coût par lignes
- *(verify)* préfixe NAX en lignes borné à 16k de contexte
- *(verify)* préfixe NAX en lignes par défaut
- *(verify)* mesures du préfixe NAX en lignes à 32k et 64k
- *(verify)* préfixe NAX opt-in pour l'attention full en lignes
- *(dflash)* vérification résidente bf16 contre l'AR et docs des arènes f32
- *(dflash)* arène de vérification résidente en bf16 au-delà du seuil greedy
- *(dflash)* fenêtre plus petite que le bloc, sorties finies et flag centralisé
- *(dflash)* attention en lignes fenêtrée contre le naïf et le CPU
- *(dflash)* attention fenêtrée du drafter en lignes
- *(matmul)* refuser les sorties non finies du skinny
- *(matmul)* mesures du skinny x2 après correctif du biais
- *(matmul)* skinny u4 x2 par défaut pour la vérification large
- *(matmul)* variante skinny u4 x2 à q brut et épilogue par groupe
- *(prefill)* mesures du GEMM NAX bf16×3 des k/v du Qwen 27B
- *(prefill)* k/v du Qwen 27B sur le GEMM NAX bf16×3
- *(flint)* comparer les hidden du MoE tuilé bit à bit
- *(prefill)* MoE routé coop tuilé 32×64 pour flint
- *(flint)* refuser les logits non finis dans la dérive du greedy x3
- *(prefill)* GEMM NAX bf16×3 par défaut pour flint
- *(prefill)* GEMM NAX u4 tuilé bf16×3 pour les petites formes de flint
- *(prefill)* GEMM NAX u4 opt-in pour les petites projections de flint
- *(decode)* rotation de K et append K/V fusionnés
- *(decode)* QKV concaténé pour flint
- *(flint)* seuils propres à la dérive du NAX fenêtré
- *(prefill)* NAX d128 fenêtré avec sinks pour les couches glissantes flint
- Merge pull request #80 from azerozero/perf/flint-nax-d128
- *(prefill)* durcir les oracles du MoE coop flint
- *(flint)* oracle mlx séparé entre MoE qmv f32 et coop bf16
- *(prefill)* MoE routé groupé (coop) pour le routeur sigmoïde de flint
- *(flint)* K/V borné bit-identique au K/V plein sur flint-int4 réel
- *(prefill)* K/V des couches glissantes flint borné à la fenêtre
- *(flint)* le stockage bf16 n'est plus un opt-in dans les commentaires
- *(flint)* K/V de prefill bf16 adopté par défaut
- *(flint)* adoption du K/V bf16 réelle, contre le semis par copie
- *(decode)* l'arène adopte le K/V bf16 de flint
- *(prefill)* réserve K/V proportionnelle au prompt pour prolonger en place
- *(prefill)* stocker en bf16 le K/V de prefill de flint
- *(metal)* lire un K/V bf16 dans l'attention prefill avec sinks
- *(kv)* vérifier aussi l'allocation de V dans le prolongement en place
- *(metal)* prolonger le K/V en place d'un tour à l'autre par vivacité des vues
- *(run)* reprendre le préfixe entre les tours du REPL
- *(metal)* identifier les allocations K/V par l'objet MTLBuffer
- *(flint)* prefill Steel avec sinks, par morceaux et préfixe prolongé
- Merge pull request #68 from azerozero/feat/flint-serve-tools
- Merge pull request #67 from azerozero/perf/flint-windowed-decode
- *(flint)* forcer le 2-passes fenêtré et durcir ses gardes
- *(flint)* decode 2-passes fenêtré avec sink sur une vue décalée du KV
- *(prefill)* recopier le préfixe sur le GPU et rendre le K/V prolongé adoptable
- *(decode)* l'arène adopte le K/V bf16 du prefill au lieu de le recopier
- *(prefill)* stocker en bf16 le K/V du prefill par défaut
- *(prefill)* stocker en bf16 le K/V du prefill par morceaux (opt-in)
- Merge pull request #58 from azerozero/perf/prefill-embed-chunks
- *(prefill)* prolonger la queue d'un prompt servi dans la réserve du KV
- *(prefill)* plonger chaque morceau sur le GPU depuis les jetons
- *(kv)* pistes mesurées pour l'attention decode quantifiée
- Merge pull request #55 from azerozero/perf/kv-quant
- *(kv)* échelle et biais hors des produits, un chargement par lane
- *(kv)* microbench decode 27B (q24/kv4) pour chaque format de KV
- *(kv)* mesures qui gardent bf16 par défaut
- *(kv)* dérive réelle Q8/K8V4/Q4 contre bf16 (ignoré)
- *(kv)* format KV porté par l'état, résolution unique et GpuElement::U8
- *(prefill)* grille NAX d256 tête la plus rapide par défaut
- *(prefill)* oracles NAX et miroir bf16 sans faux positifs
- *(prefill)* variantes NAX d256 bk64 et bq128 en opt-in
- *(prefill)* miroir bf16 incrémental du K/V pour le NAX par morceaux
- *(prefill)* grille NAX d256 tête la plus rapide en opt-in
- *(mtp)* hidden final du prefill relu en place, norme finale au semis GPU
- *(mtp)* retours d'audit du semis GPU
- *(prefill)* K/V du prefill complet relu en place, arène semée sur GPU
- *(mtp)* projections K|V du semis en un GEMM tuilé
- *(decode)* conversion bf16 du semis KV résident parallélisée
- *(mtp)* semis de l'historique sur GPU, un command buffer par bloc
- *(mtp)* premier token émis avant le semis de l'historique
- *(mtp)* semis de l'historique sans projection Q
- *(prefill)* dernière ligne seule pour l'AR, couche finale réduite au K/V
- *(prefill)* embeddings et RMSNorm CPU parallélisés par ligne
- *(decoder)* mesures du KV bf16 greedy refaites, oracles pairwise hors seuil
- *(decode)* KV résident bf16 en greedy au-delà de 16k positions
- *(decoder)* cache de préfixe MTP prolongé par le prefill résident, plafond en octets
- *(decoder)* KV CPU par segments partagés, préfixe jamais recopié
- *(serve)* hit court gardé quand l'extension passe par le prefill résident
- *(prefill)* extension de préfixe par le prefill résident, gated delta NAX
- *(prefill)* prefill résident par morceaux, scratch borné à 4096 tokens
- *(prefill)* k avant RoPE en scratch partagé, hit serve ignoré s'il est trop court
- *(prefill)* attention causale d256 sur les accélérateurs neuronaux
- Merge pull request #42 from azerozero/fix/serve-footprint-log
- Merge pull request #41 from azerozero/ci/prek-aligne-grob
- *(prek)* clippy et rustdoc en -D warnings, main protégé, cargo sweep
- *(codeql)* analyse sur toute pull request, sans filtre de chemins
- *(prefill)* scratch dense gate/up/hidden absents sur le chemin MLP fusionné
- Merge pull request #37 from azerozero/perf/prefill-fused-mlp
- *(prefill)* MLP dense fusionné gate+up+SwiGLU avec sortie bf16
- *(mtp)* logits des lignes cibles dans le verify résident et ligne bonus à T>0
- Merge pull request #33 from azerozero/perf/prefix-cache-arc
- *(mtp)* spéculatif échantillonné sur le chemin résident
- Merge pull request #31 from azerozero/perf/sampling-residual-linear
- *(sampling)* distribution résiduelle en O(V) pour le spéculatif lossless
- *(gemm)* stores vectorisés de la tuile u4 et fragments A non masqués quand M est aligné
- *(gemm)* chargeur de tuile u4 gs64 vectorisé et tuile threadgroup paddée
- Merge pull request #26 from azerozero/perf/prefill-steel-27b
- *(prefill)* attention causale Steel d256 étendue au 27B (24/4 têtes)
- *(serve)* routage MTP testé avec le flag explicite, indépendant de l'environnement
- *(perf)* extension résidente du cache de préfixe et bascule intégrée, préparées pour le bench de nuit
- *(deps)* rustls-webpki 0.103.16 et webpki-roots 1.0.10 (RUSTSEC-2026-0381, RUSTSEC-2026-0382)
- *(prefix-cache)* extension d'un préfixe par le chemin résident en lignes, historique MTP semé par le prefill DFlash
- *(perf)* cache de préfixe DFlash mesuré en serve multi-tour, réserve sur le contexte long
- *(dflash)* prefill DFlash via le cache de préfixe, captures bornées à la fenêtre du drafter
- *(perf)* phase 7, drafter DFlash résident contre le défaut MTP en pairwise (396 paires)
- *(prefill)* captures des couches cibles dans le prefill résident pour le drafter DFlash
- *(perf)* divergence T2 ramenée à un near-tie entre prefill résident et per-op
- *(perf)* bascule par acceptance mesurée contre décision par requête, verdict A
- *(perf)* microbench du drafter DFlash résident et routage skinny bf16
- *(dflash)* kernel skinny bf16 pour les matmuls du drafter résident
- *(perf)* sous-profil du drafter DFlash résident, coût dominé par le GPU du bloc
- *(dflash)* token suivant lu sur la dernière ligne du bloc de vérification et profil par segment
- *(metal)* chemin full-attn en lignes fusionné pour la vérification MTP
- *(metal)* delta linear-attention par séquence avec captures dans la vérification en lignes
- *(mtp)* lookup n-grammes actif par défaut sur prompt long après mesure pairwise
- *(metal)* GEMM skinny sur unités matricielles et linear-attention batchée pour la vérification large
- *(metal)* kernel qmm8 u4 gs64 à huit lignes, opt-in après mesure négative
- *(mtp)* lookup n-grammes expérimental et seuil mesuré du GEMM NA
- *(mtp)* semer l'historique par blocs et borner le draft par la génération
- *(mtp)* vérifier les drafts en lignes sur les couches full-attention
- *(mtp)* prefix cache et semis batché pour l'historique committed
- *(mtp)* choisir l'historique fresh ou committed selon la longueur du prompt
- *(mtp)* activer adaptive_fresh en profondeur 2 par défaut
- *(perf)* publier le criblage pairwise des optimisations MTP
- *(metal)* clarifier les variantes des fixtures numériques
- corriger la sélection des PR pour la fusion automatique
- Merge pull request #9 from azerozero/publication/tools-quant

### Fixed

- Chargeur : les quants MLX `u6` (ex. `mlx-community/*-oQ6`) étaient refusés à la validation de config alors que les kernels existent ; les modes `mxfp4`/`nvfp4` et un `output_gate_type` autre que `swish`/`silu` (Qwen3.8) sont désormais refusés avec un message clair au lieu d'une erreur de poids manquant ou d'une sortie corrompue.
- Decode résident : la liste blanche `RESIDENT_SUPPORTED_BITS` ignorait u2/u6 alors que leurs kernels résidents (`FastQmvU2`/`FastQmvU6`) existent ; un 27B oQ6 tombait en per-op.
- Kernels Metal `affine_gather_*` (experts MoE, embeddings) : le décodage bitstream n'était appliqué qu'à u3 ; u6 passait en lanes alignées (5 x 6 bits != 32) et aurait produit des experts corrompus. La condition devient `32 % bits != 0`, identique pour u3.
- MTP : un sidecar quantifié (`mlx-community/*-MTP-4bit`, `fc.scales`/`fc.biases` sans préfixe) échouait en `dtype non supporté pour mtp.fc: U32` ; la normalisation des clés du chargeur s'aligne sur celle de la détection (`assets.rs`). Le gain dépend du prompt et reste à mesurer sur des runs comparables.
- CLI et `run` : détection automatique de la tête MTP locale pour les modèles denses sous Metal en greedy non contraint. `SARAGOSSA_RUST_MTP_DECODE=0` désactive MTP ; les alias `RETI_*` restent acceptés, avec priorité aux variables `SARAGOSSA_*`. La profondeur reste à 1 par défaut, sans activation automatique du contrôleur adaptatif.

## [0.3.0](https://github.com/azerozero/saragossa/releases/tag/v0.3.0) - 2026-07-31

### Added

- Stockage single-copy des poids quantifiés en mémoire unifiée (défaut ON) : une seule résidence dès le chargement (35B-A3B : 40,5 → 19,4 Go ; 27B : ~22 Go), pic de load ≈ steady ([#7](https://github.com/azerozero/saragossa/pull/7))
- Gathers MoE rapides u3 : un 35B mixed 3/4-bit décode à ~150 tok/s (+60 %, dépasse le 4-bit avec 18 % de mémoire en moins) ([#7](https://github.com/azerozero/saragossa/pull/7))
- `saragossa serve` prêt-à-l'emploi : `--model` par chemin nu ou id Hugging Face (auto-téléchargement), picker interactif sans argument, decode spéculatif MTP automatique (modèle dense + tête + requête greedy) ([#7](https://github.com/azerozero/saragossa/pull/7))

### Fixed

- Guard de bit-width du decode résident : les quants u6/u2 retombent sur le per-op correct au lieu de produire du texte corrompu ([#7](https://github.com/azerozero/saragossa/pull/7))
- Purge du prefix-cache après la chauffe de libération des poids (restaure le byte-id MoE sur prompt long) ([#7](https://github.com/azerozero/saragossa/pull/7))

## [0.2.0](https://github.com/azerozero/saragossa/releases/tag/v0.2.0) - 2026-07-30

### Added

- kernels affine u2/u3 + decode MTP spéculatif streaming (v0.2.0) ([#6](https://github.com/azerozero/saragossa/pull/6))
- gemma4 loader, saragossa run/list CLI, per-session cache isolation, guided-JSON polish
- *(serve)* structured output json_object (v1)

### Fixed

- *(tests)* chemins HF portables via $HOME (pas de chemin personnel en dur)

### Other

- *(release-plz)* mode git pur (pas de registre crates.io)
- *(release-plz)* bump l'action v0.5.50 → v0.5.131 (fix auth git push)
- *(brew)* bump la Formula vers v0.2.0
- licence Apache-2.0 seule + table de quantification
- *(brew)* activer le build stable (tag v0.1.0) ([#5](https://github.com/azerozero/saragossa/pull/5))
- *(readme)* réécrire en anglais et documenter le projet en autonome ([#4](https://github.com/azerozero/saragossa/pull/4))
- hygiène bon-marché (cache, binaires prébuiltés, harden-runner, MSRV, doc) ([#3](https://github.com/azerozero/saragossa/pull/3))
- *(brew)* ajouter une formule Homebrew (tap installable en --HEAD)
- sync du moteur saragossa depuis reti @aa3bec9
- sync du moteur saragossa depuis reti @2d7d437
- ajouter le contexte de contribution saragossa
- renforcer les contrôles et la publication crates.io
- sync du moteur saragossa depuis reti @523e0da
- Shared predictive memory guard + external-encoder transcription seam
- Make decode pacing hot-configurable
- True SSE streaming, /health endpoint, bench-serve subcommand
- Add /v1/embeddings, fix non-Metal build, standalone-first README
- Add OpenAI-compatible audio endpoints to serve
- Serve hardening, loader guard, and engine updates
- Add dual MIT/Apache-2.0 license files + gitleaks allowlist
- run push-time audit via cargo-audit CLI, not the check-run action
- relative tolerance off-M5 for kernel-variant asserts
- commit Cargo.lock (audit needs it) + widen portable ULP tolerance to 16
- Portable bitwise asserts + 35B OptiQ-4bit benchmark row
- bench rig, per-run GPU temperatures, verified HF links
- honest same-protocol A/B benchmarks + model links
- rust checks, daily audit, auto-merge on green (mirrors the parent repo)
- Sync engine to 2026-07-06 state
- Initial release: pure-Rust Metal LLM inference engine for Apple Silicon
