# Bench concurrents, campagnes du 19 et 20 novembre 2026, M5 Max 128 Gio

Données brutes produites par le harnais [`bench/competitors/campaign.py`](../../campaign.py)
(campagne autonome en une commande). Aucune analyse n'est publiée ici.

| dossier | date | contenu |
|---|---|---|
| `19b/` | 19 nov. | six moteurs (saragossa, llama.cpp, mlx-lm, llmd, ollama, mistral.rs) ; 27B 4 bits, 35B-A3B 4 bits, 27B BF16 ; binaire saragossa antérieur aux correctifs ; 90 entrées, 234 mesures sur 270 (6 échecs de chargement, 3 entrées sautées) |
| `20-sar/` | 20 nov. | saragossa seul, `main` d894441 ; 18 entrées, 54 mesures |
| `20-dfl/` | 20 nov. | saragossa 27B BF16 en AR, MTP et DFlash, `main` 0af2102, après la PR #305 ; 9 entrées, 27 mesures |

Par dossier :

- `journal.jsonl.gz` : journal complet (chargements, requêtes, sondes, capteurs) ;
- `freeze.json` : réglages gelés, versions, sha256 des binaires et des poids ;
- `summary.json`, `DONE`, `campaign.out` (19b seulement) : bilan et sortie de la campagne ;
- `logs.tar.gz` : sortie des serveurs, un fichier par entrée et répétition ;
- `prompts/*.json` : empreintes des prompts (voir plus bas).

Les chemins personnels sont remplacés par `~`, le dossier temporaire de session par
`<scratch>`, et l'identifiant de la batterie est retiré de `power.source_raw`.

## Protocole

M5 Max 128 Gio, sur secteur, 3 répétitions par cellule, décodage greedy (température 0),
256 tokens générés, prompts de 512, 4096 et 16384 tokens, contexte 20480, porte de charge
`loadavg < 2`, refroidissement entre entrées (45 s, 5 min au plus entre modèles).

## Prompts

Le texte des prompts n'est pas publié. `prompts/<modèle>.json` garde, par longueur cible,
`prompt_tokens` et le `sha256` du contenu. `campaign.py` les reconstruit par
`build_prompts` : concaténation triée des `*.rs` du dossier `--prompt-source`
(défaut `src/` du dépôt, `freeze.json` en donne le chemin et `repo_commit`), tokens
spéciaux retirés, plus grand préfixe tenant dans la cible une fois le template de chat
du modèle appliqué (lancé avec le venv mlx-lm, `transformers`). Comparer les sha256 à
ceux publiés ; le prompt dépend du commit des sources.

## Réserves

- n = 3 par cellule, sans test statistique.
- Quantifications différentes selon les moteurs (MLX 4 bits contre GGUF Q4_K_M, etc.) :
  les débits ne comparent pas des poids identiques.
- Les concurrents ne sont mesurés que le 19 ; les campagnes du 20 ne concernent que saragossa.
- TTFT à 512 tokens en BF16 bimodal.
- mistral.rs en mode spéculatif se dégrade en contexte long (pression mémoire).
- llmd n'a ni MTP ni DFlash.
- `20-dfl/freeze.json` indique une alimentation sur batterie : le gel date d'un premier
  lancement arrêté par la porte d'alimentation avant toute mesure. La reprise a mesuré sur
  secteur, comme le montrent les 9 entrées `on_ac: true` du journal.
