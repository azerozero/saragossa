# Têtes MTP dans les shards principaux

Le chargeur MTP actuel ne peut pas charger de manière fiable une tête répartie
dans les shards principaux avec des quantifications différentes par module.
La détection automatique reste limitée aux sidecars. Une détection positive
supplémentaire dans `assets.rs` ne rendrait pas ce format chargeable.

## Ce qui est déjà pris en charge

`normalize_mtp_sidecar_key` dans `src/decoder/mtp.rs` transforme
`language_model.mtp.*` en `mtp.*`. Le préfixe n'est donc pas le blocage.
`dense_from_tensors` sait charger `fc`, les deux `pre_fc_norm_*`, les projections
d'attention et de MLP de `layers.0`, ainsi que les normes dont `norm`.
Il accepte un mélange de matrices denses et de matrices affines u4.

## Ce qui manque pour le format inline mixte

1. **Une source multi-shards.** `MtpWeightsInfo.path` désigne un fichier unique ;
   `MtpHead::from_sidecar_with` ouvre un seul `LazySidecar`, dont les vues utilisent
   un seul header et un seul descripteur. Il faut une résolution clé → shard →
   offset, limitée aux tenseurs MTP, avec détection des doublons et validation des
   payloads. Pointer vers le shard contenant `fc` perd les tenseurs des autres
   shards. Les lectures directes vers Metal doivent conserver le bon fichier pour
   chaque poids, scale et bias.
2. **La quantification par module.** `load_mtp_linear` impose `Some(4)` dans
   `infer_mtp_affine_layout`, puis `bits=4` dans les deux constructeurs CPU/Metal.
   Il ne reçoit pas le bloc `quantization` de `config.json`. Les helpers
   `quant_params_for` et `quant_override_usize` de `src/qwen_loader.rs`
   gèrent ces paramètres pour le tronc, mais le chemin MTP ne les appelle pas.
   Il faut transmettre les valeurs globales et les overrides de chaque clé
   réelle `language_model.mtp.<module>`, puis vérifier bits, mode, groupe et forme
   logique avant allocation. Déduire seulement le groupe avec `bits=4` donne une
   mauvaise largeur pour les matrices u5/u6/u8.
3. **Des oracles de chargement.** Un fixture réparti sur au moins deux shards,
   avec normes denses, projections de plusieurs quantifications, clé dupliquée,
   tenseur absent et métadonnées incohérentes doit couvrir le chargeur CPU et les
   vues Metal. L'identité des conventions de normes doit aussi être vérifiée
   contre la référence du checkpoint ; le chemin dense utilise actuellement
   une heuristique sur la moyenne.

Le format décrit pour `Jundot/Qwen3.8-27B-oQ4e-mtp` relève de ces limites.
Cette conclusion vient du chemin de chargement local ; elle ne constitue pas
une validation des fichiers de ce checkpoint. Renommer un shard en
`mtp.safetensors` ne résout ni la répartition ni les bits par module.
