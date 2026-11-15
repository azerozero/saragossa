# Détection des stops et décodage incrémental du serveur

## Contrat et qualification

Le serveur utilise `tokenizers` 0.22.2 `DecodeStream` seulement si le décodeur
est **ByteLevel pur**. Les normalizers et pre-tokenizers ne participent pas au
décodage. Les séquences de décodeurs, Metaspace, le cleanup et Gemma ne sont pas
qualifiés. Gemma 4 conserve ses cinq marqueurs de contrôle, dont les ids sont
résolus une fois à la création du flux.

ByteLevel transforme chaque surface de token en octets, les concatène et
applique `from_utf8_lossy`. Dès qu'un delta ne se termine pas par U+FFFD, sa
frontière finale ne contient plus d'octet UTF-8 incomplet : les octets suivants
ne peuvent plus modifier son texte. `DecodeStream` conserve le contexte de ce
raccord. Les tokens spéciaux ignorés et les ids absents ne produisent aucun
octet et sont éliminés avant la fenêtre. `skip_special_tokens=false` conserve
les spéciaux ; les tests vérifient les deux modes.

Les ids depuis le dernier delta stable forment une seconde fenêtre, utilisée
pour le snapshot de la queue. Elle permet de détecter un stop **au même token**
que le décodage complet, même si le token contient `STOP` suivi d'un octet UTF-8
incomplet. U+FFFD littéral est retenu aussi prudemment qu'un fragment UTF-8.
Le snapshot final, déjà décodé intégralement par `finish_completion`, libère la
queue et vérifie une fois le préfixe émis.

En release, le flux conserve le texte décodé dans un buffer et une longueur
émise, sans recopie ni comparaison du préfixe à chaque token qualifié. Un stop nouveau
commence dans les nouveaux octets ou dans les `L-1` derniers octets du snapshot
précédent, où `L` est la longueur maximale des stops. La recherche conserve
`L+3` octets, arrondis vers la frontière UTF-8 précédente : la marge couvre le
remplacement du dernier U+FFFD. Le bloc think vide est toujours filtré avant
les stops. La rétention d'un préfixe de stop examine uniquement des suffixes.

Une configuration non qualifiée, une erreur de `DecodeStream` ou une queue de
plus de 64 tokens utilise le décodage complet. Ce dernier compare le préfixe
émis. Un flux ByteLevel dégradé reprend avec un `DecodeStream` neuf dès que le
snapshot complet ne finit plus par U+FFFD : cette frontière ne contient plus
d'octet UTF-8 incomplet. Le pas de reprise recherche encore les stops dans
tout le snapshot. Le repli reste quadratique tant que cette frontière manque,
ainsi que pour les décodeurs non qualifiés. Le compteur `fallback_decodes` est
journalisé à la destruction uniquement si un flux qualifié s'est dégradé, y
compris pour `TextStop` et les flux de lot. Une suite de spéciaux ignorés ne
fait pas grossir la fenêtre. Aucun gain de performance Gemma n'est revendiqué.

## Oracles CPU

Les tests de `serve::streaming` couvrent :

- UTF-8 sur plusieurs tokens, octets invalides, U+FFFD littéral, espaces de
  tête, ids inconnus, spéciaux conservés/ignorés et longues suites de spéciaux ;
- concaténation des deltas et queue égale au décodage complet à chaque coupe ;
- `finish()` à toutes les coupes, y compris dans le bloc think vide ;
- stops au début, chevauchant plusieurs tokens, multioctets, imbriqués et
  présents avant un UTF-8 incomplet dans un même token ;
- deux propriétés de 256 cas sur les octets aléatoires et les stops ;
- les 42 séquences `ids`/`ids_special` des goldens locaux de chacune des trois
  familles Qwen, Gemma 3 et Gemma 4, à toutes les coupes, plus 128 séquences
  aléatoires par famille ; SHA-256 du tokenizer contrôlé avant utilisation.

Le test `bytelevel_batch_preserves_content_stop_position_and_usage` vérifie
également deux flux entrelacés sans GPU : contenu, fin `stop`, nombre de tokens
consommés et absence de consommation supplémentaire après le stop. Les
oracles de streaming comparent aussi chaque décision à celle de `TextStop`.

Les fixtures locales absentes sont signalées `SKIP`. Définir explicitement
`SARAGOSSA_TOKENIZER_QWEN3`, `SARAGOSSA_TOKENIZER_GEMMA3` et
`SARAGOSSA_TOKENIZER_GEMMA4` force leur validation et fait échouer le test si le
fichier manque ou si son SHA diffère. Les fichiers de poids ne sont pas lus.

```sh
cargo test --no-default-features --features devtools --bin saragossa serve::
cargo test --no-default-features --lib tokenizer::
cargo test --release --no-default-features --features devtools \
  --bin saragossa detokenizer_cpu_bench -- --ignored --nocapture
```

## Mesure du 14 novembre 2026

Parent : `057a7bf`. Tokenizer Qwen local correspondant au golden `qwen3.jsonl`,
SHA-256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`.
Build release CPU sans feature Metal. Le harnais contient le chemin parent
(décodage complet, contains/find, suffixe et copie du texte émis) et le chemin
serve candidat, dans le même binaire. Aucun GPU, réseau ou poids de modèle.

Corpus multilingue fixe répété, deux stops absents. Une paire d'échauffement
jetée puis neuf paires, ordre avant/après alterné. Chronométrage monotone du
travail CPU ; médiane en ms/token. À 1k/8k/32k, mesure des 128 pas suivant N
tokens, préparation exclue. À 200 tokens, réponse entière sans préparation.
Le candidat vérifie le snapshot contre l'oracle et zéro repli hors chronomètre.

| Tokens déjà produits | Parent ms/token | Incrémental ms/token | Min–max incrémental |
|---:|---:|---:|---:|
| 200 (réponse entière) | 0,005843 | 0,000674 | 0,000669–0,000685 |
| 1 024 | 0,058551 | 0,000690 | 0,000664–0,000707 |
| 8 192 | 0,432236 | 0,000663 | 0,000629–0,000699 |
| 32 768 | 1,792081 | 0,000686 | 0,000646–0,000789 |

La pente liée à la longueur de réponse disparaît sur ce corpus ; le coût à
32k reste très inférieur au seuil de 0,5 ms/token. Ces valeurs ne sont pas
comparables directement aux 0,21/1,67/6,5 ms de l'audit initial, dont le corpus
et le harnais diffèrent. Aucun gain de débit serveur ou inter-token GPU n'est
déduit de cette mesure CPU ; le protocole service 8k/32k reste à exécuter sous
verrou GPU.
