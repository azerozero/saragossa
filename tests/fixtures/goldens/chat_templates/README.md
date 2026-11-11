# Références de templates

Ces quatre couples `.bin`/`.json` sont copiés sans modification depuis la
branche `test/goldens-reference`, non fusionnée au moment de la correction.
Ils proviennent de `transformers.apply_chat_template` avec transformers
5.22.2. Les `.bin` contiennent du JSON UTF-8 ; les en-têtes décrivent la
provenance, les empreintes des configs/tokenizers et le SHA256 du payload.
Aucun poids de modèle n'est nécessaire au test de rendu direct.

- `golden::chat_templates::templates_match_all_28_transformers_references`
  compare 26 rendus et deux erreurs de rôles Gemma 3, sans ignorer les écarts.
- `serve::state::template_tests::serve_renders_all_eight_reference_cases_without_weights`
  exerce le rendu utilisé par `prompt_encoding` après conversion wire.
- `serve::state::reference_goldens::external_serve_templates_match_transformers`
  vérifie aussi les IDs de tokens en appelant `prompt_encoding`, avec les
  tokenizers/configs locaux et un décodeur sentinelle synthétique non exécuté.
  Ce test reste ignoré par défaut et échoue si un chemin manque.

Pour l'oracle de tokenisation, définir `SARAGOSSA_QWEN38_DIR`,
`SARAGOSSA_QWEN36_MOE_DIR`, `SARAGOSSA_GEMMA3_DIR`, `SARAGOSSA_GEMMA4_DIR`, puis :

```sh
CARGO_NET_OFFLINE=true SARAGOSSA_REQUIRE_REAL_MODELS=1 cargo test --bin saragossa \
  serve::state::reference_goldens::external_serve_templates_match_transformers \
  -- --ignored --exact
```

Les identifiants d'appels et les événements HTTP sont vérifiés séparément
par les tests de `serve::tool_output`, `serve::http` et `serve::anthropic`.
Aucune de ces vérifications ne démontre la qualité des décisions d'un modèle
lors d'une génération réelle.
