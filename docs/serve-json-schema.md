# Schema-constrained serve output

`POST /v1/chat/completions` accepts the OpenAI response format:

```json
{
  "model": "local-model",
  "messages": [{"role": "user", "content": "Give a greeting for Paris."}],
  "temperature": 0.8,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "greeting",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "city": {"type": "string", "enum": ["Paris", "Brest"]},
          "greeting": {"type": "string", "enum": ["Bonjour"]}
        },
        "required": ["city", "greeting"],
        "additionalProperties": false
      }
    }
  }
}
```

The same constraint is available on `POST /v1/messages` using
`"output_config":{"format":{"type":"json_schema","schema":{...}}}`,
the [Anthropic structured output format](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
Other `output_config` fields are rejected. The token-counting endpoint retains
its existing rejection of output configuration. Strict tool calling is separate
and is not enabled by this response format.

The name must contain 1–64 ASCII letters, digits, underscores or hyphens.
`strict` is optional and accepts `true`, `false` or `null`; every accepted schema
is enforced, including when strict is false. No unsupported keyword is ignored.

## Accepted schema subset

| Keyword | Supported values |
|---|---|
| `type` | object, array, string, integer, number, boolean, null; nonempty unique arrays of types below the root |
| `properties` | Map of property names to schemas |
| `required` | Unique string names, each declared in `properties` |
| `enum` | Nonempty, unique scalar values; no object/array enum values |
| `items` | One schema for all array elements, including boolean schemas |
| `additionalProperties` | Boolean or schema; absence retains JSON Schema's permissive default |
| `title`, `description`, `default` | Annotations; do not supply missing content or defaults |

The root must explicitly declare `"type":"object"`. Nested empty schemas and
boolean schemas are accepted when the resulting grammar is satisfiable.
Schemas are limited to 64 KiB serialized, 32 levels and 1024 schema nodes, plus
llguidance's grammar/lexer limits. Invalid values, unsupported keywords,
unsatisfiable schemas and compilation warnings produce HTTP 400 with context.

All other keywords are unsupported: in particular `$schema`, `$id`, `$ref`,
`$defs`, `definitions`, `anyOf`, `oneOf`, `allOf`, `not`, conditionals,
`minimum`/`maximum`/`multipleOf`, string length, `pattern`, `format`, array length,
`prefixItems`, `contains`, `uniqueItems`, property counts and dependencies.
`x-guidance` is also rejected so requests cannot enable lenient compilation.

Generation may use a restricted spelling/order of valid JSON: llguidance orders
declared properties and emits non-ASCII string characters as UTF-8 rather than
arbitrary `\uXXXX` spellings. The constraint guarantees the accepted schema, not
that every equivalent serialization is reachable. Numeric magnitude has the
same limitation as `json_object`: the syntax grammar does not impose an f64 bound.

## Runtime behavior

The schema is compiled and cached per request before model loading. The model's
llguidance trie/factory is initialized lazily on its first schema request. Each
generation has its own parser. HF ByteLevel and ByteFallback tokenizer surfaces
preserve partial UTF-8 bytes, added special tokens and SentencePiece spaces;
special tokens are excluded from content, and all configured EOS/EOT IDs are
accepted only after a complete object. The synthetic WordLevel fixtures use the
existing decoder surfaces.

Schema masks are intersected with the existing JSON syntax guard, then applied
before the existing CPU greedy/top-k/top-p sampler. Fast-forward and backtracking
are disabled. Guided requests retain the existing solo/CPU routing, including
the exclusions from batching and speculative decoding. Text and `json_object`
requests never initialize the schema factory or compile a schema.

An object completed at exactly `max_tokens` ends with `finish_reason: "stop"`.
An incomplete object returns `incomplete_json`; in SSE, already-emitted deltas
remain a prefix and the stream ends with an error instead of normal completion.
Custom `stop` sequences are rejected because they could cut a required value.

## Library choice

[llguidance](https://github.com/guidance-ai/llguidance) supplies the schema compiler
and token-trie mask engine. Its MIT license is compatible with the engine's
Apache-2.0 license and allowed by `deny.toml`. Version 1.9.1 declares Rust 1.87;
the selected **llguidance 1.2.0 and toktrie 1.2.0 are pinned exactly** and have
been compiled with Rust 1.85. Raising either version requires repeating that
check. Earlier 1.0.0 also compiled in the compatibility probe.

Default features are disabled: no Lark parser, referencing resolver, rayon
integration, ahash feature or optional JSON Schema validator. Besides llguidance
and toktrie, the lockfile gains anyhow, derivre 0.3.8, bytemuck_derive, hashbrown
0.15.5, heck, strum, strum_macros and syn 3. The remaining dependencies are shared
with the engine. llguidance enables serde_json's `preserve_order` feature for
the entire crate, changing `Value` objects from sorted `BTreeMap` keys to
insertion-ordered `IndexMap` keys. `chat_template::canonical_json` explicitly
restores the historical recursive key order wherever these values become prompt
text: Anthropic tool declarations, arguments and JSON content fallbacks,
Responses conversion and Chat Completions JSON content fallbacks. Gemma 4 uses
the same sorted-entry helper for its native tool syntax. Serialized tool-call
arguments returned to clients also retain their historical order when replayed
as history.
Arrays, string escaping and number formatting are unchanged, preserving prompt
bytes and the token sequences used by the prefix cache.

Raw Chat Completions tools and arguments still preserve the client's source
order for Qwen/flint through the separate Python-compatible JSON renderer;
they must not be canonicalized. The schema size check only measures serialized
length, and compiled grammars are cached per request without a serialized JSON
cache key, so schema compilation keeps llguidance's property-order semantics.

`cargo deny` checks the actual locked graph without adding license or advisory
exceptions; duplicate-version warnings are permitted by the existing policy.

No second HF tokenizers crate is needed: llguidance reads the serialized
tokenizer 0.22 representation to construct its byte trie. The engine's tokenizer
continues to handle prompt encoding and output decoding.

A schema-aware extension of the homegrown JSON automaton would avoid these
dependencies, but would also need required-property tracking, enum prefix
recognition, nested schema state and token-boundary handling. The compatible
llguidance version avoids maintaining that second semantic compiler. The API
deliberately exposes a tested subset rather than the library's full evolving
JSON Schema dialect.

## Verification

Unit tests cover compilation errors/limits, required keys, enums, nested arrays,
additional properties, raw Unicode, terminal masks, multiple EOT IDs,
multi-byte tokens and request isolation. Property tests sample masks and validate
the resulting required keys, enum values and boolean types independently.
Serve tests exercise greedy/sampled generation, SSE and exact/exhausted budgets;
a local Gemma 3 tokenizer test also exercises split UTF-8 bytes with synthetic
weights. The SDK e2e matrix checks required content despite a conflicting `{}`
prompt on both APIs, at temperature 0 and 0.8.

Byte-level regressions compare recursively reordered tool declarations and
arguments with the sorted output from before `preserve_order`. Run
`cargo test golden::chat_templates` and `cargo test serve::` without regenerating
the chat-template goldens.
