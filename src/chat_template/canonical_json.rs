//! JSON compact historique des prompts, indépendant de `preserve_order`.
//!
//! Les objets sont triés récursivement comme l'ancien `BTreeMap` de
//! `serde_json`. Les tableaux, nombres et échappements restent inchangés.
//! Ce format ne remplace pas le JSON Python qui conserve l'ordre source.

use serde_json::{Map, Value};

/// Sérialise une valeur avec les mêmes octets que `serde_json` sans
/// `preserve_order`, pour les prompts et les clés de cache qui en dépendent.
pub fn to_string(value: &Value) -> String {
    let mut out = String::new();
    write(value, &mut out);
    out
}

/// Ordre commun au JSON compact et au format d'outils Gemma 4.
pub(super) fn sorted_entries(object: &Map<String, Value>) -> Vec<(&String, &Value)> {
    let mut entries: Vec<_> = object.iter().collect();
    entries.sort_unstable_by_key(|(key, _)| *key);
    entries
}

fn write(value: &Value, out: &mut String) {
    match value {
        Value::Object(object) => {
            out.push('{');
            for (index, (key, value)) in sorted_entries(object).into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(key.clone()).to_string());
                out.push(':');
                write(value, out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (index, value) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write(value, out);
            }
            out.push(']');
        }
        _ => out.push_str(&value.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde::Serialize;
    use std::collections::BTreeMap;

    // Oracle indépendant : la représentation de Value sur main avant llguidance.
    #[derive(Serialize)]
    #[serde(untagged)]
    enum MainValue {
        Null,
        Bool(bool),
        Number(serde_json::Number),
        String(String),
        Array(Vec<MainValue>),
        Object(BTreeMap<String, MainValue>),
    }

    impl From<&Value> for MainValue {
        fn from(value: &Value) -> Self {
            match value {
                Value::Null => Self::Null,
                Value::Bool(v) => Self::Bool(*v),
                Value::Number(v) => Self::Number(v.clone()),
                Value::String(v) => Self::String(v.clone()),
                Value::Array(v) => Self::Array(v.iter().map(Self::from).collect()),
                Value::Object(v) => {
                    Self::Object(v.iter().map(|(k, v)| (k.clone(), Self::from(v))).collect())
                }
            }
        }
    }

    fn value_strategy() -> impl Strategy<Value = Value> {
        let scalar = prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map(Value::Bool),
            any::<i64>().prop_map(|v| Value::Number(v.into())),
            any::<u64>().prop_map(|v| Value::Number(v.into())),
            any::<f64>().prop_map(|v| serde_json::json!(v)),
            ".*".prop_map(Value::String),
        ];
        scalar.prop_recursive(4, 64, 8, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..8).prop_map(Value::Array),
                prop::collection::vec((".*", inner), 0..8)
                    .prop_map(|entries| Value::Object(entries.into_iter().collect())),
            ]
        })
    }

    proptest! {
        #[test]
        fn matches_main_btree_bytes(value in value_strategy()) {
            let original = value.to_string();
            let main = MainValue::from(&value);
            let expected = serde_json::to_string(&main).expect("invariant: JSON valide");
            let actual = to_string(&value);
            prop_assert_eq!(actual.as_bytes(), expected.as_bytes());
            prop_assert_eq!(value.to_string(), original);
        }
    }

    #[test]
    fn sorts_objects_inside_arrays_without_changing_scalars_or_array_order() {
        let value: Value =
            serde_json::from_str(r#"{"z":[{"é":-0.0,"a":"\n\"\\"},{"z":null,"a":true}],"a":1e20}"#)
                .expect("invariant: JSON valide");
        assert_eq!(
            to_string(&value),
            r#"{"a":1e+20,"z":[{"a":"\n\"\\","é":-0.0},{"a":true,"z":null}]}"#,
        );
    }
}
