# Drafter DFlash : découverte automatique

`saragossa serve` et `saragossa run` chargent le drafter DFlash 2 sans variable
d'environnement. Ordre de recherche, le premier qui convient gagne :

1. `SARAGOSSA_RUST_DFLASH_DRAFT_DIR=<dossier>` : toujours prioritaire, sans
   contrôle de compatibilité ; une valeur vide ou `0` désactive le drafter
   (aucune recherche ensuite) ;
2. le sous-dossier `dflash/` du dossier du modèle (`<modèle>/dflash/config.json`
   et `model.safetensors`) ;
3. le cache Hugging Face (`HF_HUB_CACHE`, `HF_HOME` ou `~/.cache/huggingface/hub`) :
   un snapshot dont le `README.md` déclare `base_model` (`Qwen/Qwen3.8-27B`)
   et dont ce nom préfixe celui du modèle (`Qwen3.8-27B-4bit`).

Un drafter découvert (2 et 3) doit valider `hidden_size`, `num_target_layers`
et, s'il le déclare, `vocab_size` du modèle ; sinon il est ignoré et le log le
dit. Le drafter n'est joignable que par le chemin MTP : il faut une tête MTP
active (`mtp.safetensors`, ou `SARAGOSSA_RUST_MTP_SIDECAR` pour un tronc MoE).

Au démarrage, une ligne explique le décodage choisi : `MTP : tête embarquée`,
`DFlash : drafter <chemin> (sous-dossier du modèle)`, ou `aucun drafter DFlash :
décodage MTP/AR, gain possible avec …`. `saragossa doctor` (section CONSEILS)
signale un modèle non quantifié et un drafter manquant, avec la commande.
