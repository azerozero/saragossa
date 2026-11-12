//! Capture passive des tenseurs du prefill Gemma 4, limitée au thread appelant.

use crate::{InferError, Result, Tensor};
use serde::Serialize;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

const POINTS: &[&str] = &[
    "input",
    "attention_input",
    "attention_raw",
    "attention_out",
    "post_attention_residual",
    "dense_input",
    "dense_raw",
    "dense_out",
    "router_input",
    "router_logits",
    "router_indices",
    "router_weights",
    "moe_input",
    "moe_raw",
    "moe_out",
    "ffn_sum",
    "ffn_out",
    "pre_scalar",
    "output",
];

thread_local! {
    static CAPTURE: RefCell<Option<State>> = const { RefCell::new(None) };
}

#[derive(Serialize)]
struct Entry {
    layer: Option<usize>,
    point: String,
    shape: [usize; 2],
    file: String,
    dtype: &'static str,
    source_dtype: &'static str,
    #[serde(skip)]
    values: Vec<f32>,
}

struct State {
    directory: PathBuf,
    prompt: Vec<usize>,
    layers: usize,
    backend: &'static str,
    current_layer: Option<usize>,
    entries: BTreeMap<String, Entry>,
}

fn io_error(path: &Path, source: std::io::Error) -> InferError {
    InferError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn key(layer: Option<usize>, point: &str) -> String {
    layer.map_or_else(|| point.to_owned(), |i| format!("layers.{i:03}.{point}"))
}

/// Une capture appartient au thread appelant, même lorsque Rayon calcule les matmuls.
pub(crate) struct Capture {
    // Rc rend le garde !Send : son Drop doit vider le même thread-local.
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl Capture {
    pub(crate) fn start(
        directory: &Path,
        prompt: &[usize],
        layers: usize,
        backend: &'static str,
    ) -> Result<Self> {
        CAPTURE.with_borrow_mut(|slot| {
            if slot.is_some() {
                return Err(InferError::Config("capture de couches déjà active".into()));
            }
            // create_dir refuse aussi un répertoire vide existant : aucun écrasement.
            std::fs::create_dir(directory).map_err(|e| io_error(directory, e))?;
            *slot = Some(State {
                directory: directory.to_path_buf(),
                prompt: prompt.to_vec(),
                layers,
                backend,
                current_layer: None,
                entries: BTreeMap::new(),
            });
            Ok(Self {
                _thread: std::marker::PhantomData,
            })
        })
    }

    pub(crate) fn finish(self) -> Result<()> {
        let state = CAPTURE
            .with_borrow_mut(Option::take)
            .ok_or_else(|| InferError::Config("capture absente".into()))?;
        let mut expected = vec![(None, "embeddings", state.prompt.len())];
        for layer in 0..state.layers {
            expected.extend(POINTS.iter().map(|p| (Some(layer), *p, state.prompt.len())));
        }
        expected.extend([(None, "final_norm", 1), (None, "logits", 1)]);
        if state.entries.len() != expected.len() {
            return Err(InferError::Config(
                "capture incomplète : chemin non instrumenté".into(),
            ));
        }
        let mut entries = Vec::with_capacity(expected.len());
        for (layer, point, rows) in expected {
            let name = key(layer, point);
            let entry = state
                .entries
                .get(&name)
                .filter(|e| e.shape[0] == rows)
                .ok_or_else(|| {
                    InferError::Config(format!("capture absente ou incomplète : {name}"))
                })?;
            let path = state.directory.join(&entry.file);
            let mut file = std::io::BufWriter::new(
                std::fs::File::create_new(&path).map_err(|e| io_error(&path, e))?,
            );
            for value in &entry.values {
                file.write_all(&value.to_le_bytes())
                    .map_err(|e| io_error(&path, e))?;
            }
            file.flush().map_err(|e| io_error(&path, e))?;
            entries.push(entry);
        }
        // index.json est publié en dernier : son absence signale une capture interrompue.
        let index = serde_json::json!({
            "format": "saragossa-layer-dump-v1", "backend": state.backend,
            "prompt_ids": state.prompt, "num_layers": state.layers,
            "logits": "last-token-after-softcap", "entries": entries,
        });
        let path = state.directory.join("index.json");
        let bytes = serde_json::to_vec_pretty(&index).map_err(|source| InferError::Json {
            path: path.clone(),
            source,
        })?;
        std::fs::write(&path, bytes).map_err(|e| io_error(&path, e))
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        CAPTURE.with_borrow_mut(|slot| *slot = None);
    }
}

pub(crate) struct LayerScope(Option<usize>);

pub(crate) fn enter_layer(layer: usize) -> LayerScope {
    LayerScope(CAPTURE.with_borrow_mut(|slot| {
        slot.as_mut()
            .and_then(|state| state.current_layer.replace(layer))
    }))
}

impl Drop for LayerScope {
    fn drop(&mut self) {
        CAPTURE.with_borrow_mut(|slot| {
            if let Some(state) = slot {
                state.current_layer = self.0;
            }
        });
    }
}

pub(crate) fn active() -> bool {
    CAPTURE.with_borrow(|slot| slot.is_some())
}

pub(crate) fn record(point: &str, tensor: &Tensor) -> Result<()> {
    CAPTURE.with_borrow_mut(|slot| {
        let Some(state) = slot else {
            return Ok(());
        };
        let (rows, cols) = tensor.as_matrix()?;
        let name = key(state.current_layer, point);
        let entry = state.entries.entry(name.clone()).or_insert_with(|| Entry {
            layer: state.current_layer,
            point: point.to_owned(),
            shape: [0, cols],
            file: format!("{name}.f32"),
            dtype: "<f4",
            source_dtype: "float32",
            values: Vec::new(),
        });
        if entry.shape[1] != cols || entry.shape[0] + rows > state.prompt.len() {
            return Err(InferError::Dimension(format!(
                "capture répétée ou forme incompatible : {name}"
            )));
        }
        entry.shape[0] += rows;
        entry.values.extend_from_slice(tensor.data());
        Ok(())
    })
}

pub(crate) fn record_routing(mut selections: Vec<Vec<(usize, f32)>>) -> Result<()> {
    let rows = selections.len();
    let cols = selections.first().map_or(0, Vec::len);
    let mut indices = Vec::with_capacity(rows * cols);
    let mut weights = Vec::with_capacity(rows * cols);
    for row in &mut selections {
        // MLX argpartition ne trie pas les experts ; comparer par ID, sans changer le calcul.
        row.sort_by_key(|(expert, _)| *expert);
        for (expert, weight) in row {
            indices.push(*expert as f32);
            weights.push(*weight);
        }
    }
    record(
        "router_indices",
        &Tensor::from_vec(vec![rows, cols], indices)?,
    )?;
    record(
        "router_weights",
        &Tensor::from_vec(vec![rows, cols], weights)?,
    )
}
