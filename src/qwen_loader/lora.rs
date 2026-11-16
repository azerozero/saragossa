//! Adaptateurs LoRA PEFT fusionnés au chargement.
//!
//! Format lu : `adapter_config.json` et `adapter_model.safetensors`, tels que
//! les écrit PEFT. Chaque module ciblé reçoit `W' = W + s·B·A`, avec
//! `s = lora_alpha / r`, ou `lora_alpha / √r` sous `use_rslora`. La fusion se
//! fait en f32 sur une base non quantifiée (BF16, F16 ou F32). Un poids
//! quantifié ciblé est refusé : requantifier après fusion changerait l'erreur
//! de quantification, et cette dérive n'a pas de KL publiée.
//!
//! Toute variante que ce chargeur ne sait pas reproduire exactement est
//! refusée : DoRA, `fan_in_fan_out`, biais LoRA, `modules_to_save`, motifs de
//! rang ou d'alpha par module, tenseurs autres que `lora_A`/`lora_B`.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::error::{InferError, Result};
use crate::tensor::Tensor;

const CONFIG_FILE: &str = "adapter_config.json";
const WEIGHTS_FILE: &str = "adapter_model.safetensors";
const KEY_PREFIX: &str = "base_model.model.";

fn invalid(message: impl Into<String>) -> InferError {
    InferError::Config(format!("LoRA : {}", message.into()))
}

/// Adaptateur LoRA PEFT prêt à fusionner dans un modèle de base.
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Modules par nom de poids de base (`<module>.weight`, nommage du checkpoint).
    modules: HashMap<String, LoraModule>,
    paths: [PathBuf; 2],
    sha256: String,
}

/// Facteurs low-rank d'un module : `A` `[r, in]`, `B` `[out, r]`.
#[derive(Debug, Clone)]
pub(crate) struct LoraModule {
    a: Tensor,
    b: Tensor,
    scale: f32,
}

impl LoraAdapter {
    /// Lit un adaptateur PEFT depuis son répertoire. Aucun code n'est exécuté.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les fichiers manquent ou si l'adaptateur sort du
    /// périmètre fusionnable exactement (voir la documentation du module).
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config_path = dir.join(CONFIG_FILE);
        let weights_path = dir.join(WEIGHTS_FILE);
        let config_bytes = read(&config_path)?;
        let config: Value = serde_json::from_slice(&config_bytes)
            .map_err(|e| invalid(format!("{CONFIG_FILE} illisible : {e}")))?;
        let (rank, scale) = validate_config(&config)?;
        let weight_bytes = read(&weights_path)?;
        let modules = parse_modules(&weight_bytes, rank, scale)?;
        let mut hasher = Sha256::new();
        hasher.update(&config_bytes);
        hasher.update(&weight_bytes);
        Ok(Self {
            modules,
            paths: [config_path, weights_path],
            sha256: crate::hex(hasher.finalize()),
        })
    }

    /// Empreinte SHA-256 de la configuration et des poids de l'adaptateur.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Nombre de modules ciblés.
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Fichiers lus, pour l'identité des instantanés d'état.
    pub(crate) fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    pub(crate) fn module(&self, base_weight: &str) -> Option<&LoraModule> {
        self.modules.get(base_weight)
    }

    pub(crate) fn module_names(&self) -> impl Iterator<Item = &str> {
        self.modules.keys().map(String::as_str)
    }
}

impl LoraModule {
    #[cfg(test)]
    pub(crate) fn rank(&self) -> usize {
        self.a.shape()[0]
    }

    #[cfg(test)]
    pub(crate) fn a(&self) -> &[f32] {
        self.a.data()
    }

    #[cfg(test)]
    pub(crate) fn b(&self) -> &[f32] {
        self.b.data()
    }

    #[cfg(test)]
    pub(crate) fn scale(&self) -> f32 {
        self.scale
    }

    /// Ajoute `s·B·A` à `weight` (`[out, in]`, ligne-majeur), en f32.
    pub(crate) fn merge_into(&self, name: &str, shape: &[usize], weight: &mut [f32]) -> Result<()> {
        let [rank, inputs] = *self.a.shape() else {
            return Err(invalid(format!("{name} : lora_A n'est pas 2D")));
        };
        let [outputs, rank_b] = *self.b.shape() else {
            return Err(invalid(format!("{name} : lora_B n'est pas 2D")));
        };
        if shape != [outputs, inputs] || rank != rank_b || weight.len() != outputs * inputs {
            return Err(invalid(format!(
                "{name} : base {shape:?} incompatible avec A [{rank}, {inputs}] et B [{outputs}, {rank_b}]"
            )));
        }
        let (a, b, scale) = (self.a.data(), self.b.data(), self.scale);
        weight
            .par_chunks_mut(inputs)
            .zip(b.par_chunks(rank))
            .for_each(|(row, b_row)| {
                for (k, &b_ok) in b_row.iter().enumerate() {
                    let coefficient = scale * b_ok;
                    let a_row = &a[k * inputs..(k + 1) * inputs];
                    for (w, &a_ki) in row.iter_mut().zip(a_row) {
                        *w += coefficient * a_ki;
                    }
                }
            });
        Ok(())
    }
}

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|source| InferError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Valide la configuration PEFT et rend `(r, échelle)`.
fn validate_config(config: &Value) -> Result<(usize, f32)> {
    if config["peft_type"] != "LORA" {
        return Err(invalid(format!(
            "peft_type {} non pris en charge",
            config["peft_type"]
        )));
    }
    let rank = config["r"]
        .as_u64()
        .and_then(|r| usize::try_from(r).ok())
        .filter(|&r| r > 0)
        .ok_or_else(|| invalid("r entier positif requis"))?;
    let alpha = config["lora_alpha"]
        .as_f64()
        .filter(|alpha| alpha.is_finite())
        .ok_or_else(|| invalid("lora_alpha fini requis"))?;
    let truthy = |key: &str| config.get(key).is_some_and(|v| v.as_bool() == Some(true));
    let present = |key: &str| {
        config.get(key).is_some_and(|v| match v {
            Value::Null => false,
            Value::Object(map) => !map.is_empty(),
            Value::Array(items) => !items.is_empty(),
            _ => true,
        })
    };
    for key in [
        "use_dora",
        "fan_in_fan_out",
        "lora_bias",
        "use_qalora",
        "use_bdlora",
    ] {
        if truthy(key) {
            return Err(invalid(format!("{key} non pris en charge")));
        }
    }
    for key in [
        "modules_to_save",
        "rank_pattern",
        "alpha_pattern",
        "target_parameters",
        "layer_replication",
        "trainable_token_indices",
        "alora_invocation_tokens",
    ] {
        if present(key) {
            return Err(invalid(format!("{key} non pris en charge")));
        }
    }
    if !matches!(config.get("bias"), None | Some(Value::Null)) && config["bias"] != "none" {
        return Err(invalid(format!(
            "bias {} non pris en charge",
            config["bias"]
        )));
    }
    // NOTE: PEFT calcule l'échelle en flottant Python (f64), puis la
    // multiplie aux poids f32 ; l'arrondi en f32 ici suit le même ordre.
    let denominator = if truthy("use_rslora") {
        (rank as f64).sqrt()
    } else {
        rank as f64
    };
    Ok((rank, (alpha / denominator) as f32))
}

/// Associe chaque paire `lora_A`/`lora_B` au poids de base qu'elle corrige.
fn parse_modules(bytes: &[u8], rank: usize, scale: f32) -> Result<HashMap<String, LoraModule>> {
    let tensors = safetensors::SafeTensors::deserialize(bytes)
        .map_err(|e| invalid(format!("{WEIGHTS_FILE} illisible : {e}")))?;
    let mut pairs: BTreeMap<String, [Option<Tensor>; 2]> = BTreeMap::new();
    for (key, view) in tensors.tensors() {
        let stripped = key
            .strip_prefix(KEY_PREFIX)
            .ok_or_else(|| invalid(format!("clé {key} hors de {KEY_PREFIX}")))?;
        let (module, slot) = if let Some(module) = stripped.strip_suffix(".lora_A.weight") {
            (module, 0)
        } else if let Some(module) = stripped.strip_suffix(".lora_B.weight") {
            (module, 1)
        } else {
            return Err(invalid(format!("tenseur {key} non pris en charge")));
        };
        let tensor = crate::safetensor::tensor_from_safetensor_parts(
            &key,
            view.dtype(),
            view.shape(),
            view.data(),
        )?;
        if tensor.data().iter().any(|x| !x.is_finite()) {
            return Err(invalid(format!("{key} contient une valeur non finie")));
        }
        pairs.entry(format!("{module}.weight")).or_default()[slot] = Some(tensor);
    }
    if pairs.is_empty() {
        return Err(invalid("aucun module"));
    }
    pairs
        .into_iter()
        .map(|(name, [a, b])| {
            let (Some(a), Some(b)) = (a, b) else {
                return Err(invalid(format!("{name} : paire lora_A/lora_B incomplète")));
            };
            let shapes_ok =
                matches!((a.shape(), b.shape()), ([ra, _], [_, rb]) if *ra == rank && *rb == rank);
            if !shapes_ok {
                return Err(invalid(format!(
                    "{name} : rang attendu {rank}, A {:?}, B {:?}",
                    a.shape(),
                    b.shape()
                )));
            }
            Ok((name, LoraModule { a, b, scale }))
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use safetensors::{serialize, Dtype, View};
    use std::borrow::Cow;

    pub(crate) struct F32View(pub(crate) Vec<usize>, pub(crate) Vec<u8>);

    impl View for F32View {
        fn dtype(&self) -> Dtype {
            Dtype::F32
        }
        fn shape(&self) -> &[usize] {
            &self.0
        }
        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.1)
        }
        fn data_len(&self) -> usize {
            self.1.len()
        }
    }

    pub(crate) fn f32_view(shape: Vec<usize>, values: &[f32]) -> F32View {
        F32View(shape, values.iter().flat_map(|v| v.to_le_bytes()).collect())
    }

    /// Module de test : nom du poids de base sans `.weight`, A, forme A, B, forme B.
    pub(crate) type TestModule<'a> = (&'a str, Vec<f32>, Vec<usize>, Vec<f32>, Vec<usize>);

    /// Écrit un adaptateur PEFT : `modules` = (nom du poids de base sans
    /// `.weight`, A, forme A, B, forme B).
    pub(crate) fn write_adapter(dir: &Path, config: &Value, modules: &[TestModule<'_>]) {
        let mut views = Vec::new();
        for (module, a, a_shape, b, b_shape) in modules {
            views.push((
                format!("{KEY_PREFIX}{module}.lora_A.weight"),
                f32_view(a_shape.clone(), a),
            ));
            views.push((
                format!("{KEY_PREFIX}{module}.lora_B.weight"),
                f32_view(b_shape.clone(), b),
            ));
        }
        let bytes = serialize(views, None).expect("invariant: safetensors sérialisable");
        std::fs::write(dir.join(WEIGHTS_FILE), bytes).expect("invariant: écriture adaptateur");
        std::fs::write(dir.join(CONFIG_FILE), config.to_string())
            .expect("invariant: écriture config");
    }

    pub(crate) fn lora_config(rank: usize, alpha: f64) -> Value {
        serde_json::json!({
            "peft_type": "LORA", "r": rank, "lora_alpha": alpha, "bias": "none",
            "use_dora": false, "use_rslora": false, "fan_in_fan_out": false,
            "modules_to_save": null, "rank_pattern": {}, "alpha_pattern": {},
            "target_modules": ["q_proj"]
        })
    }

    fn one_module_adapter(config: &Value) -> Result<LoraAdapter> {
        let dir = tempfile::tempdir().expect("invariant: répertoire temporaire");
        write_adapter(
            dir.path(),
            config,
            &[(
                "model.layers.0.self_attn.q_proj",
                vec![1.0, 2.0],
                vec![1, 2],
                vec![0.5, -1.0],
                vec![2, 1],
            )],
        );
        LoraAdapter::load(dir.path())
    }

    #[test]
    fn merge_matches_low_rank_product_computed_apart() {
        // W [3, 4], r = 2, s = 0,75 : W'·x doit égaler W·x + s·B·(A·x).
        let w: Vec<f32> = (0..12).map(|i| (i as f32 * 0.37).sin()).collect();
        let a: Vec<f32> = (0..8).map(|i| (i as f32 * 0.91).cos()).collect();
        let b: Vec<f32> = (0..6).map(|i| 0.1 * i as f32 - 0.2).collect();
        let module = LoraModule {
            a: Tensor::from_vec(vec![2, 4], a.clone()).expect("invariant: A"),
            b: Tensor::from_vec(vec![3, 2], b.clone()).expect("invariant: B"),
            scale: 0.75,
        };
        let mut merged = w.clone();
        module
            .merge_into("w", &[3, 4], &mut merged)
            .expect("invariant: formes compatibles");
        let x = [0.3_f64, -1.2, 0.8, 2.0];
        for o in 0..3 {
            let fused: f64 = (0..4).map(|i| f64::from(merged[o * 4 + i]) * x[i]).sum();
            let base: f64 = (0..4).map(|i| f64::from(w[o * 4 + i]) * x[i]).sum();
            let ax: Vec<f64> = (0..2)
                .map(|k| (0..4).map(|i| f64::from(a[k * 4 + i]) * x[i]).sum())
                .collect();
            let low_rank: f64 = (0..2).map(|k| f64::from(b[o * 2 + k]) * ax[k]).sum();
            let expected = base + 0.75 * low_rank;
            assert!(
                (fused - expected).abs() <= 1e-5,
                "ligne {o} : {fused} contre {expected}"
            );
        }
    }

    #[test]
    fn scale_follows_alpha_over_rank_or_rslora() {
        let adapter =
            one_module_adapter(&lora_config(1, 4.0)).expect("invariant: adaptateur valide");
        let module = adapter
            .module("model.layers.0.self_attn.q_proj.weight")
            .expect("invariant: module présent");
        assert_eq!(module.scale, 4.0);
        assert_eq!(adapter.module_count(), 1);
        assert_eq!(adapter.sha256().len(), 64);
        let mut rs = lora_config(4, 4.0);
        rs["use_rslora"] = Value::Bool(true);
        assert_eq!(
            validate_config(&rs).expect("invariant: rslora valide"),
            (4, 2.0)
        );
    }

    #[test]
    fn unsupported_variants_are_refused() {
        for (key, value) in [
            ("use_dora", Value::Bool(true)),
            ("fan_in_fan_out", Value::Bool(true)),
            ("bias", Value::from("all")),
            ("modules_to_save", serde_json::json!(["lm_head"])),
            ("rank_pattern", serde_json::json!({"q_proj": 8})),
            ("alpha_pattern", serde_json::json!({"q_proj": 8})),
            ("peft_type", Value::from("IA3")),
        ] {
            let mut config = lora_config(1, 1.0);
            config[key] = value;
            assert!(
                validate_config(&config).is_err(),
                "{key} aurait dû être refusé"
            );
        }
    }

    #[test]
    fn rank_mismatch_and_foreign_tensors_are_refused() {
        // r = 2 annoncé, tenseurs de rang 1.
        assert!(one_module_adapter(&lora_config(2, 1.0)).is_err());
        let dir = tempfile::tempdir().expect("invariant: répertoire temporaire");
        let views = vec![(
            format!("{KEY_PREFIX}model.layers.0.self_attn.q_proj.lora_magnitude_vector"),
            f32_view(vec![2], &[1.0, 1.0]),
        )];
        std::fs::write(
            dir.path().join(WEIGHTS_FILE),
            serialize(views, None).expect("invariant: sérialisable"),
        )
        .expect("invariant: écriture");
        std::fs::write(
            dir.path().join(CONFIG_FILE),
            lora_config(1, 1.0).to_string(),
        )
        .expect("invariant: écriture");
        assert!(LoraAdapter::load(dir.path()).is_err());
    }
}
