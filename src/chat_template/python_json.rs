//! `tojson(ensure_ascii=False)` du Jinja de transformers : `json.dumps` de
//! Python sur la valeur que `json.loads` a lue dans la requête.
//!
//! NOTE: `serde_json::Value` range ses objets dans un `BTreeMap` et trie donc
//! les clés, alors que Python garde l'ordre du client. Activer la feature
//! `preserve_order` corrigerait l'ordre, mais pour tout le crate : les octets
//! des prompts Qwen du shim Anthropic changeraient aussi. D'où ce petit modèle
//! JSON local, ordonné, lu depuis le texte brut de la requête.
//!
//! Fidélité à Python : clés dans l'ordre source, une clé répétée garde la
//! position de sa première occurrence et la valeur de la dernière, chaînes
//! réécrites avec les échappements minimaux et l'unicode en clair, flottants
//! au format `repr` (`1e2` → `100.0`, `1.50` → `1.5`, `1e16` → `1e+16`).
//! Limites, propres à serde_json sans `arbitrary_precision` : `-0` est lu en
//! flottant (`-0.0`, Python écrit `0`) et un entier hors de `i64`/`u64` aussi,
//! alors que Python garde l'un et l'autre en entier.

use std::collections::HashMap;
use std::fmt;

use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::Number;

/// Valeur JSON dont les objets gardent l'ordre de leurs clés.
#[derive(Debug, PartialEq)]
pub(super) enum JsonValue {
    /// `null`.
    Null,
    /// Booléen.
    Bool(bool),
    /// Nombre.
    Number(Number),
    /// Chaîne.
    String(String),
    /// Tableau.
    Array(Vec<JsonValue>),
    /// Objet ordonné.
    Object(JsonObject),
}

/// Objet JSON ordonné, indexé par clé.
#[derive(Debug, Default, PartialEq)]
pub(super) struct JsonObject {
    members: Vec<(String, JsonValue)>,
    index: HashMap<String, usize>,
}

impl JsonObject {
    /// Lit un objet JSON ; `None` pour tout autre texte.
    pub(super) fn parse(json: &str) -> Option<Self> {
        match serde_json::from_str(json).ok()? {
            JsonValue::Object(object) => Some(object),
            _ => None,
        }
    }

    /// Valeur du membre `key`.
    pub(super) fn get(&self, key: &str) -> Option<&JsonValue> {
        self.index.get(key).map(|&slot| &self.members[slot].1)
    }

    /// Membres dans l'ordre source.
    pub(super) fn iter(&self) -> impl Iterator<Item = (&str, &JsonValue)> {
        self.members
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }

    /// Comme un `dict` Python : la dernière valeur gagne, à la première place.
    fn insert(&mut self, key: String, value: JsonValue) {
        match self.index.get(&key) {
            Some(&slot) => self.members[slot].1 = value,
            None => {
                self.index.insert(key.clone(), self.members.len());
                self.members.push((key, value));
            }
        }
    }
}

impl JsonValue {
    /// Texte d'une chaîne JSON.
    pub(super) fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }

    /// `json.dumps(value, ensure_ascii=False)`.
    pub(super) fn to_python_json(&self) -> String {
        let mut out = String::new();
        self.write_python_json(&mut out);
        out
    }

    fn write_python_json(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Self::Number(number) => match number.as_f64().filter(|_| number.is_f64()) {
                Some(value) => out.push_str(&python_float_repr(value)),
                None => out.push_str(&number.to_string()),
            },
            Self::String(text) => {
                out.push_str(&serde_json::Value::String(text.clone()).to_string())
            }
            Self::Array(items) => {
                out.push('[');
                for (position, item) in items.iter().enumerate() {
                    if position > 0 {
                        out.push_str(", ");
                    }
                    item.write_python_json(out);
                }
                out.push(']');
            }
            Self::Object(object) => {
                out.push('{');
                for (position, (key, value)) in object.iter().enumerate() {
                    if position > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&serde_json::Value::String(key.to_string()).to_string());
                    out.push_str(": ");
                    value.write_python_json(out);
                }
                out.push('}');
            }
        }
    }
}

/// `repr(float)` de Python : plus courte écriture exacte, notation
/// scientifique quand l'exposant décimal sort de `[-4, 16)`.
fn python_float_repr(value: f64) -> String {
    if !value.is_finite() {
        // `json.dumps` écrit les non-finis ainsi ; serde_json n'en lit pas.
        return if value.is_nan() {
            "NaN".to_string()
        } else if value > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        };
    }
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .unwrap_or((scientific.as_str(), "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    if !(-4..16).contains(&exponent) {
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        return format!("{sign}{mantissa}e{exponent_sign}{:02}", exponent.abs());
    }
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let (integer, fraction) = if exponent >= 0 {
        let split = usize::try_from(exponent).unwrap_or(0) + 1;
        if digits.len() > split {
            (digits[..split].to_string(), digits[split..].to_string())
        } else {
            (format!("{digits:0<split$}"), String::new())
        }
    } else {
        let zeros = usize::try_from(-exponent - 1).unwrap_or(0);
        ("0".to_string(), format!("{}{digits}", "0".repeat(zeros)))
    };
    let fraction = if fraction.is_empty() {
        "0".to_string()
    } else {
        fraction
    };
    format!("{sign}{integer}.{fraction}")
}

impl<'de> Deserialize<'de> for JsonValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(JsonVisitor)
    }
}

struct JsonVisitor;

impl<'de> Visitor<'de> for JsonVisitor {
    type Value = JsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("une valeur JSON")
    }

    fn visit_unit<E>(self) -> Result<JsonValue, E> {
        Ok(JsonValue::Null)
    }

    fn visit_bool<E>(self, value: bool) -> Result<JsonValue, E> {
        Ok(JsonValue::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<JsonValue, E> {
        Ok(JsonValue::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<JsonValue, E> {
        Ok(JsonValue::Number(value.into()))
    }

    fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<JsonValue, E> {
        Number::from_f64(value)
            .map(JsonValue::Number)
            .ok_or_else(|| E::custom("nombre JSON non fini"))
    }

    fn visit_str<E>(self, value: &str) -> Result<JsonValue, E> {
        Ok(JsonValue::String(value.to_string()))
    }

    fn visit_string<E>(self, value: String) -> Result<JsonValue, E> {
        Ok(JsonValue::String(value))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<JsonValue, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element()? {
            items.push(item);
        }
        Ok(JsonValue::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<JsonValue, A::Error> {
        let mut object = JsonObject::default();
        while let Some((key, value)) = map.next_entry::<String, JsonValue>()? {
            object.insert(key, value);
        }
        Ok(JsonValue::Object(object))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dumps(json: &str) -> String {
        serde_json::from_str::<JsonValue>(json)
            .expect("invariant: JSON valide")
            .to_python_json()
    }

    #[test]
    fn dumps_like_python_json_dumps() {
        let raw = r#" { "z" : [1 ,2.5,true,null] , "a:b" : "x, y: \"q\" \\ \/ é \n" ,
                       "é" : { } , "☺" : "☺" } "#;

        assert_eq!(
            dumps(raw),
            r#"{"z": [1, 2.5, true, null], "a:b": "x, y: \"q\" \\ / é \n", "é": {}, "☺": "☺"}"#
        );
    }

    #[test]
    fn numbers_follow_python_repr() {
        let raw = "[1e2, 1.50, -0, 10, 1e16, 1.5e-5, 0.0001, 123.456, -0.0, 1.0, 12345678901234567, 2.5e-300]";

        assert_eq!(
            dumps(raw),
            "[100.0, 1.5, -0.0, 10, 1e+16, 1.5e-05, 0.0001, 123.456, -0.0, 1.0, 12345678901234567, 2.5e-300]"
        );
    }

    #[test]
    fn object_keeps_source_order_and_last_duplicate_wins() {
        let object = JsonObject::parse(r#"{"b": 1, "a": {"y": 2, "x": 3}, "b": 4}"#)
            .expect("invariant: objet JSON");

        let keys: Vec<&str> = object.iter().map(|(key, _)| key).collect();
        assert_eq!(keys, ["b", "a"]);
        assert_eq!(
            object.get("b").map(JsonValue::to_python_json).as_deref(),
            Some("4")
        );
        assert_eq!(
            object.get("a").map(JsonValue::to_python_json).as_deref(),
            Some(r#"{"y": 2, "x": 3}"#)
        );
        assert!(JsonObject::parse("[1]").is_none());
    }
}
