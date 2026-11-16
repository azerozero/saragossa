# Profils de configuration du moteur

Statut : proposition, étape 2 non implémentée. Aucun lecteur TOML, endpoint
admin, observateur de fichiers ni argument `--profile` n'est ajouté ici.
Les features Cargo de [compilation](../features.md) sélectionnent le code ;
un profil ne peut jamais réactiver une modalité absente du binaire.

## Référence et périmètre

Grob sépare les fonctions dans `src/features/`, les préréglages distribués dans
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
Une variable explicite invalide est une erreur, jamais un repli silencieux.
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
