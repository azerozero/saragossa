//! `tojson(ensure_ascii=False)` du Jinja de transformers, appliqué au texte
//! JSON brut de la requête.
//!
//! `serde_json::Value` trie les clés des objets ; le template, lui, garde
//! l'ordre du client. Le texte brut est donc reformaté jeton par jeton plutôt
//! que réanalysé.

use std::fmt;

use serde::de::{Deserialize, Deserializer, MapAccess, Visitor};
use serde_json::value::RawValue;

/// Objet JSON dont les membres gardent l'ordre du texte source.
pub(super) struct OrderedObject(pub(super) Vec<(String, Box<RawValue>)>);

impl OrderedObject {
    /// Lit un objet JSON ; `None` pour tout autre texte.
    pub(super) fn parse(json: &str) -> Option<Self> {
        serde_json::from_str(json).ok()
    }

    /// Valeur brute du premier membre `key`.
    pub(super) fn get(&self, key: &str) -> Option<&RawValue> {
        self.0
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.as_ref())
    }
}

impl<'de> Deserialize<'de> for OrderedObject {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct OrderedVisitor;
        impl<'de> Visitor<'de> for OrderedVisitor {
            type Value = OrderedObject;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("un objet JSON")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<OrderedObject, A::Error> {
                let mut members = Vec::new();
                while let Some(member) = map.next_entry::<String, Box<RawValue>>()? {
                    members.push(member);
                }
                Ok(OrderedObject(members))
            }
        }
        deserializer.deserialize_map(OrderedVisitor)
    }
}

/// Chaîne d'une valeur JSON brute, `None` si ce n'en est pas une.
pub(super) fn json_string(raw: &RawValue) -> Option<String> {
    serde_json::from_str(raw.get()).ok()
}

/// Reformate un texte JSON valide comme `json.dumps(ensure_ascii=False)` :
/// blancs retirés, `", "` et `": "` hors des chaînes, chaînes réécrites avec
/// les échappements minimaux et l'unicode en clair. L'ordre des clés et les
/// nombres restent ceux du texte source.
pub(super) fn python_json(json: &str) -> String {
    let mut out = String::with_capacity(json.len() + json.len() / 8);
    let mut chars = json.char_indices();
    while let Some((start, ch)) = chars.next() {
        match ch {
            '"' => {
                let mut end = json.len();
                let mut escaped = false;
                for (index, c) in chars.by_ref() {
                    if escaped {
                        escaped = false;
                    } else if c == '\\' {
                        escaped = true;
                    } else if c == '"' {
                        end = index + 1;
                        break;
                    }
                }
                let token = &json[start..end];
                match serde_json::from_str::<String>(token) {
                    Ok(text) => out.push_str(&serde_json::Value::String(text).to_string()),
                    Err(_) => out.push_str(token),
                }
            }
            ',' => out.push_str(", "),
            ':' => out.push_str(": "),
            c if c.is_whitespace() => {}
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reformats_like_python_json_dumps() {
        let raw = r#" { "z" : [1 ,2.5,true,null] , "a:b" : "x, y: \"q\" \\ \/ é \n" ,
                       "é" : { } , "☺" : "☺" } "#;

        assert_eq!(
            python_json(raw),
            r#"{"z": [1, 2.5, true, null], "a:b": "x, y: \"q\" \\ / é \n", "é": {}, "☺": "☺"}"#
        );
    }

    #[test]
    fn ordered_object_keeps_source_order() {
        let object = OrderedObject::parse(r#"{"b": 1, "a": {"y": 2, "x": 3}}"#)
            .expect("invariant: objet JSON");

        let keys: Vec<&str> = object.0.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(keys, ["b", "a"]);
        assert_eq!(
            object.get("a").map(|raw| python_json(raw.get())),
            Some(r#"{"y": 2, "x": 3}"#.to_string())
        );
        assert!(OrderedObject::parse("[1]").is_none());
    }
}
