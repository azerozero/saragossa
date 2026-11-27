# Profils de configuration du moteur

Statut : proposition, étape 2 non implémentée. Aucun lecteur TOML, endpoint
admin, observateur de fichiers ni argument `--profile` n'est ajouté ici.
Les features Cargo de [compilation](../features.md) sélectionnent le code ;
un profil ne peut jamais réactiver une modalité absente du binaire.

## Référence et périmètre

Grob sépare ses fonctions par module, les préréglages distribués dans
`presets/` (avec un index et des descriptions) et les compositions locales dans
`profiles/`. Saragossa conserverait ses modules moteur actuels et reprendrait
seulement ce format de configuration, sans routage de fournisseurs ni secrets.

Le moteur centralise déjà la lecture des variables dans `src/runtime_flags.rs`.
`runtime_preset.rs` choisit des défauts par famille de modèle ; `KernelProfile`
fige les choix de noyaux pour un exécuteur. Les quelque 150 flags
`SARAGOSSA_RUST_*` évoqués pour ce chantier ne forment pas une liste stable :
il faut inventorier les lectures réelles, y compris dans `metal_backend`,
`decoder` et les modalités, plutôt que figer ce nombre dans l'API.
Les familles `SARAGOSSA_SERVE_*`, `SARAGOSSA_MEMORY_*`, `SARAGOSSA_TTS_*` et
`SARAGOSSA_DECISION_*` méritent le même registre, sans changer leur nom.

## Fichiers et résolution

Proposition de commande : `saragossa serve --profile perf`, ou
`--profile ./profiles/local.toml`. Un nom simple désigne un preset embarqué et
versionné ; un chemin explicite désigne un fichier local. Aucun téléchargement
ni recherche implicite dans le dossier courant. `presets/index.toml` donnerait
pour chaque nom sa description, ses features requises et sa version de schéma.
`profiles/` contiendrait des exemples reproductibles, sans chemins personnels.

Ordre, par clé : environnement explicite > fichier > défaut du moteur.
Le fichier peut étendre un unique preset ; ses valeurs remplacent celles du
preset. Le défaut reste celui de la famille du modèle, puis le défaut général.
Pour préserver la compatibilité, l'alias historique `RETI_*` reste une variable
d'environnement, après le nom canonique `SARAGOSSA_*` et avant le fichier.
Refuser une variable explicite invalide serait une rupture du comportement
actuel, qui retombe sur le défaut. Cette décision reste à prendre lors de
l'étape 2 ; le refus n'est pas un comportement acquis.
Les options de requête (température, top-p, budget de tokens) conservent leur
contrat actuel ; `--profile` choisit un fichier, sans créer une seconde priorité
concurrente pour les mêmes réglages.

```toml
schema_version = 1
extends = "perf"

[meta]
description = "Cache de préfixe borné pour une machine à mémoire limitée"

[engine]
backend = "metal"

[flags]
SARAGOSSA_RUST_PREFIX_CACHE_BYTES = 1073741824
SARAGOSSA_RUST_KV_QUANT = "bf16"
SARAGOSSA_MEMORY_HEADROOM_BYTES = 2147483648
```

Un registre typé décrirait chaque nom canonique, alias, type, unité, plage,
défaut par modèle, feature requise, site de lecture et classe de mutabilité.
Un booléen TOML reste un booléen ; une taille reste un entier en octets. Rejeter
les clés inconnues, valeurs hors plage, cycles d'héritage, versions inconnues
et options d'une feature absente. Les profils ne contiennent jamais de code.
Un diagnostic de configuration effective afficherait valeur, origine et
nécessité de redémarrer. Les chemins sensibles seraient masqués.

La résolution doit précéder `initialize_kernel_environment`, tout premier
accès à un `OnceLock`, le chargement des poids et la création des threads.
Ne pas implémenter les profils par `set_var` : les lectures consulteraient une
configuration typée figée, avec l'environnement capturé au démarrage.
L'absence de profil doit reproduire exactement les défauts actuels.

## Préréglages proposés

| Preset | Intention | Réglages envisagés |
|---|---|---|
| `perf` | Chemins qualifiés pour le modèle et le matériel | Défauts actuels de `KernelProfile`, decode résident, cache préfixe ; aucune optimisation expérimentale activée en bloc |
| `prudence` | Comparaison Metal avec moins de fusions | Désactivation explicite des variantes expérimentales ; préserver les garde-fous mémoire et vérifier les oracles |
| `reference-cpu` | Oracle numérique CPU | Backend CPU, aucun exécuteur Metal ; quantification et modèle identiques à la comparaison |
| `low-memory` | Borner les caches et modèles résidents | Budgets préfixe réduits, pool serveur d'un modèle, marge mémoire conservée ; KV quantifié seulement après qualification |
| `diagnostic` | Mesure reproductible | Traces de timing explicites et bornées, instrumentation devtools si compilée ; aucun contenu de prompt journalisé par défaut |

Chaque preset serait testé sur des modèles synthétiques et les oracles
existants, avec la configuration effective enregistrée dans les résultats.
Le mode CPU est une référence de comparaison, pas une promesse de parité
bit à bit avec tous les noyaux Metal. Aucun profil ne charge de poids en ligne.

## Rechargement : possibilités et limites

| Réglage | État actuel observé | Proposition |
|---|---|---|
| Limite de débit du decode | `set_decode_max_tokens_per_s` écrit un `AtomicU64` | Candidat immédiat, effet au prochain intervalle de decode |
| Température, top-k/top-p, seed, max tokens par défaut | Options de génération copiées pour une requête | Nouvelle version applicable aux requêtes suivantes, jamais modifier une génération en cours |
| Profondeur MTP (`MTP_MAX_DRAFT`) | Lecture directe dans la politique CLI/serve | Capturer au début de chaque requête après validation ; ne pas varier au milieu d'un decode |
| Traces et métriques sans contenu | Certains flags sont en `OnceLock` | Candidats après remplacement explicite par un état dynamique ; pas rechargeables aujourd'hui |
| Limites de connexions, streams et octets en attente | `serve_*` en `OnceLock`, structures créées au démarrage | Candidats après refonte de l'admission ; une baisse ne coupe pas les requêtes acceptées |
| Capacité des caches et pool LRU | Valeurs mémorisées, objets durables | Candidats entre requêtes après éviction contrôlée ; ne jamais libérer des buffers encore empruntés |
| Garde mémoire et marge de sécurité | Valeurs en `OnceLock` et réservations actives | Rester figées initialement ; évolution possible avec comptabilité des réservations |

Restent figés pour la durée du processus : les kill-switches de noyaux lus en
`OnceLock` (`DECODE_RESIDENT_FULL`, `DECODE_PIPELINE`, `QMM_SKINNY_KERNEL`,
`LINEAR_*`, variantes MoE), `KernelProfile` et les pipelines associés, le
backend, le format KV (`KV_QUANT`, `KV_BF16`), les layouts d'arène, tailles de
blocs de cache et caractéristiques des snapshots persistants. Les poids,
quantifications, tokenizer, sidecars MTP/DFlash et embedder du classifieur
sont choisis au chargement : leur changement impose au minimum le drainage,
l'invalidation des caches et un nouveau chargement du modèle. L'étape initiale
les classerait donc « redémarrage requis ». Un profil ne réinitialise jamais
un `OnceLock` déjà lu et ne réinterprète jamais un cache sous un autre layout.

Un futur `POST /admin/config/reload` authentifié, ou un rechargement explicite
du fichier, construirait une configuration candidate complète. Valider toutes
les clés puis comparer les valeurs effectives, après priorité de l'environnement.
Si une valeur figée change : rejeter atomiquement avec la liste des clés et
« redémarrage requis », sans application partielle. Sinon publier une version
immuable entre requêtes sur le thread moteur ; les requêtes en cours gardent
leur version. Un fichier invalide conserve la dernière version valide.
La surveillance du fichier serait optionnelle, avec temporisation et lecture
après remplacement atomique. Le journal ne contiendrait que version, clés,
origines et résultat, sans corps de requête.

## Critères avant implémentation

Inventorier tous les lecteurs et leurs caches réels ; couvrir la priorité
canonique/alias/fichier/défaut, les erreurs de schéma, l'héritage et les features
absentes. Comparer sans profil les mêmes sorties et tests qu'aujourd'hui.
Tester un rechargement valide, invalide et concurrent, un changement figé
refusé et une baisse de budget avec des requêtes actives. Aucune de ces
fonctionnalités de configuration n'est livrée par l'étape 1.

## Implémenté : `config`, `--set`, presets, SIGHUP

- `saragossa config check FILE|PRESET [--set k=v]...` : valide ; erreur de syntaxe ou de schéma avec ligne et colonne, code de sortie 2.
- `saragossa config print [FILE|PRESET] [--set k=v]...` : une ligne `cle = valeur  # origine` par clé (`--set`, `env NOM`, `profil FICHIER`, `preset NOM`, `défaut`). Les options dédiées de `serve` (`--max-tokens-cap`...) et les secrets ne sont pas concernés.
- `serve --set section.cle=valeur` (répétable) : priorité de la CLI, donc au-dessus de l'environnement. Valeur TOML, ou texte brut. Une option dédiée explicite (`--read-timeout-secs`) l'emporte sur `--set` pour la même clé.
- Presets : `--profile low-mem|throughput` ou `extends = "low-mem"` (un seul niveau). Ordre : `--set` > env > fichier > preset > défaut. `low-mem` : 16 connexions, file 8, 2 flux, pool 1, 64 Mio en vol, cache de préfixe 2 Gio, 1 flux enhance, marge mémoire 4 Gio. `throughput` : 128 connexions, file 64, pool 4, 1 Gio en vol, cache de préfixe 32 Gio, 4 flux enhance. Valeurs non mesurées. Un fichier local nommé comme un preset doit être passé avec un chemin (`./low-mem`).
- SIGHUP (Unix) : relit le profil et n'applique que `serve.read_timeout_secs` (nouvelles connexions) `serve.max_tokens_cap` (requêtes suivantes), `serve.max_tok_s` et `serve.gpu_share` (prochain token ou command buffer ; sommeil interrompu en 20 ms si on repasse à 100). Si une autre clé change, tout est refusé avec la liste (« redémarrage requis »), rien n'est appliqué ; une erreur de lecture garde l'ancien état. Sans profil, ou avec un preset seul, rien à recharger. Le plafond de tokens utilisé pour l'éligibilité aux lots reste celui du lancement.
- Travailler sur le Mac pendant l'inférence : `--gpu-share 10-100` (`run`, `serve`, `serve.gpu_share`) mesure le temps GPU de chaque command buffer (`GPUEndTime - GPUStartTime`) et dort `busy * (100 / share - 1)` ensuite, en decode (AR, MTP, DFlash, un tour spéculatif = ses command buffers) comme entre les morceaux de prefill. Il borne aussi le pool rayon à `share` % des cœurs, réglé une fois au lancement. À 100 ou sans l'option, le chemin est inchangé (un chargement atomique) ; la sortie greedy ne change pas, seul le temps change. Le pipeline de decode, qui chevauche les command buffers, est désactivé sous bridage. `--max-tok-s R` (`serve.max_tok_s`, ex-`SARAGOSSA_RUST_MAX_TOK_S`) plafonne en plus le decode AR en tok/s ; MTP et DFlash ne le respectent pas (seul `--gpu-share` les bride). Limites : une génération dont le pipeline de decode est déjà engagé ne voit pas un bridage activé ensuite ; le decode en lots (lightbatch) ignore `--max-tok-s`. Précédence `max_tok_s` : CLI > `SARAGOSSA_RUST_MAX_TOK_S` > profil, comme les autres clés. Hors périmètre : le disque (lu seulement au chargement) et la RAM, déjà bornée par `[memory] cap_bytes` / `headroom_bytes` (garde mémoire) et `serve.prefix_cache_bytes`, sans plafond de contexte dédié.
- Rechargement de la part et du plafond : modifier `serve.gpu_share` ou `serve.max_tok_s` dans le profil puis `kill -HUP <pid>` les applique à chaud : la part au command buffer suivant, génération en cours comprise, le plafond au token AR suivant (mêmes exceptions que plus haut : pipeline déjà engagé, MTP, DFlash et lots ignorent `max_tok_s`) ; retirer la clé rend la valeur par défaut (100, illimité). Comme pour `serve.idle_unload_secs`, une option CLI (`--gpu-share`, `--max-tok-s`) ou la variable `SARAGOSSA_RUST_MAX_TOK_S` (alias `RETI_`) épingle la valeur : le profil rechargé ne la change pas. Pas d'endpoint HTTP d'administration : le signal, réservé au propriétaire du processus, suffit et n'ouvre aucune surface réseau.
- Céder le GPU quand on se sert du Mac (macOS, opt-in) : `--auto-yield` (`serve.auto_yield = true`, `SARAGOSSA_SERVE_AUTO_YIELD=1`, précédence CLI > env > profil ; défaut éteint, chemin inchangé). Un thread lit chaque seconde le temps depuis la dernière entrée clavier ou souris de la session (`CGEventSourceSecondsSinceLastEventType` sur la source matérielle HID, même compteur que `HIDIdleTime`, sans permission ni dépendance ; les événements synthétiques ne comptent pas). Entrée depuis moins de 5 s : la part tombe à `serve.auto_yield_share` (25, 10 à 100, jamais au-dessus de la part configurée) ; après `serve.auto_yield_idle_secs` (60, au moins 5) sans entrée, la part configurée revient. Entre les deux, l'état ne change pas (hystérésis : pas d'oscillation pendant une pause de quelques secondes). Seul le rapport cyclique change, le pool rayon reste celui du lancement ; les tokens sont identiques (test `gpu_share_keeps_greedy_output_synthetic`, bascule toutes les 3 ms pendant la génération). Un SIGHUP pendant le cédage enregistre la nouvelle part, appliquée au retour d'inactivité. Ces trois clés exigent un redémarrage. Limites : une manette de jeu ne passe pas par ce compteur (un jeu joué uniquement à la manette n'est pas vu après 60 s), l'appli plein écran au premier plan n'est pas détectée, et taper son prompt dans un client local déclenche aussi le cédage. Sans session graphique (démon système), le compteur est absurde et la part configurée reste.
- Lancement en service : préfixer par `taskpolicy -b` (`taskpolicy -b saragossa serve --auto-yield ...`, ou `ProcessType = Background` dans le plist launchd) place le processus en classe d'arrière-plan : ordonnancement CPU et E/S derrière les applis interactives. Cela ne bride pas le GPU, d'où `--gpu-share` et `--auto-yield`.
- Protocole de mesure (à faire hors bancs, machine au repos) : un jeu ou une appli GPU avec compteur d'images (par exemple un benchmark de jeu en boucle, fps relevés par l'overlay Metal `MTL_HUD_ENABLED=1`) au premier plan, et `serve` qui génère en continu (requêtes chat de 512 tokens en boucle, tok/s lus dans les timings de réponse). Quatre passes de 3 min en ordre ABBA : A = `serve` sans option, B = `serve --auto-yield` avec la souris en mouvement (cédage actif) ; plus deux témoins, le jeu seul et `serve` seul. Relever fps médian et 1 % bas, tok/s médian, et le délai de bascule (journal `auto-yield:` de `serve`). Critère : fps sous B proches du jeu seul, tok/s sous B près de 25 % de `serve` seul, retour à pleine vitesse 60 s après la dernière entrée.
- Libérer la RAM quand personne n'utilise `serve` : `--idle-unload-secs N` (`serve.idle_unload_secs`, `SARAGOSSA_RUST_IDLE_UNLOAD_SECS`, 0 ou absent = désactivé, rechargeable à chaud par SIGHUP). Après N secondes sans requête, file vide et moteur libre (jamais pendant une génération ni avec une requête en file), le moteur décharge tous les modèles par le chemin d'invalidation existant : poids, arènes GPU, KV, caches de préfixe en RAM, STT/embeddings. La requête suivante recharge sur le thread moteur et attend, sans erreur ; deux requêtes simultanées se serrent dans la file. Les modèles `--preload` sont déchargés comme les autres (le préchargement ne vaut qu'au démarrage). Le cache de préfixe SSD reste sur disque. Le journal écrit `unloaded idle_secs=… in_ms=… freed_bytes=…` et `loaded model=… in_ms=…`. Précédence : CLI > `--set` > `SARAGOSSA_RUST_IDLE_UNLOAD_SECS` > profil.
- SIGUSR2 (Unix), relais sans coupure : `serve` relance le binaire du même chemin (ou `--upgrade-binary PATH`) avec les mêmes arguments. L'enfant hérite de la socket d'écoute TCP ou Unix (`SARAGOSSA_LISTEN_FDS`, seul fd sans FD_CLOEXEC), vérifie son adresse, l'adopte et le signale sur une paire de sockets (`SARAGOSSA_RELAY_READY_FD`). L'ancien processus cesse alors d'accepter, finit ses requêtes en cours et sort, au plus tard après `serve.drain_timeout_secs` (30, comme `read_timeout_secs`) : une réponse en flux plus longue que ce délai est coupée, à régler en conséquence. Un enfant qui ne se dit pas prêt en 10 min est arrêté. Si l'enfant sort avant d'être prêt, l'ancien continue et journalise l'erreur.
  - Listeners enhance : h2c et WebTransport suivent dans la même variable (`7,h2=9,wt=10`), adresse vérifiée par l'enfant. HTTP/2 : l'ancien cesse d'accepter, envoie GOAWAY et attend la fin des flux ouverts. QUIC : les deux processus partagent la socket UDP ; l'enfant ne la lit qu'après son annonce « prêt », et dès cette annonce l'ancien ne la lit plus et ferme ses sessions (CONNECTION_CLOSE, H3_NO_ERROR) pour que les clients se reconnectent chez l'enfant. Les sessions WebTransport ouvertes ne migrent pas, et l'enfant génère un nouveau certificat (relire `/v1/audio/enhance/certificate`).
  - `serve.relay_mode` : `parallel` garde les deux processus résidents pendant le vidage (modèles préchargés deux fois) ; `sequential` lance l'enfant sans précharger, son moteur attend la sortie de l'ancien (requêtes en file, au-delà de `serve.queue` réponse 503 comme d'habitude) puis précharge ; `auto` (défaut) ne passe en séquentiel que si la garde mémoire de `serve` refuse une seconde empreinte égale à celle de l'ancien (garde désactivée : parallèle).
