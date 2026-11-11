//! Instantané diagnostic post-prefill. Le decode reprend le format du cache de
//! préfixe ; seules les données persistantes traversent la frontière disque.
mod storage;
#[cfg(test)]
mod tests;
mod validation;
use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use storage::{invalid, io, Blob, Source, Writer};

/// Métadonnées figées avant la lecture des poids, uniquement en mode diagnostic.
#[derive(Clone, Debug, Default)]
pub(crate) struct SnapshotSources(Option<std::result::Result<Vec<Source>, String>>);

impl SnapshotSources {
    pub(crate) fn capture<'a>(paths: impl IntoIterator<Item = &'a Path>) -> Self {
        if crate::runtime_flags::state_snapshot_dir().is_none() {
            return Self::default();
        }
        Self(Some(
            (|| -> Result<Vec<Source>> {
                let mut sources = Vec::new();
                for path in paths {
                    sources.push(Source::read(path)?);
                    if let Some(parent) = path.parent() {
                        for name in ["model.safetensors.index.json", "config.json"] {
                            let auxiliary = parent.join(name);
                            if auxiliary.is_file() {
                                let source = Source::read(&auxiliary)?;
                                if !sources.contains(&source) {
                                    sources.push(source);
                                }
                            }
                        }
                    }
                }
                Ok(sources)
            })()
            .map_err(|e| e.to_string()),
        ))
    }

    fn verified(&self) -> Result<&[Source]> {
        let sources = self
            .0
            .as_ref()
            .ok_or_else(|| invalid("identité des poids absente au chargement"))?
            .as_ref()
            .map_err(invalid)?;
        if sources.is_empty() {
            return Err(invalid("poids sans chemin source"));
        }
        for source in sources {
            source.verify()?;
        }
        Ok(sources)
    }

    pub(crate) fn finish_loading(mut self) -> Self {
        if self.0.is_some() {
            if let Err(error) = self.verified() {
                self.0 = Some(Err(error.to_string()));
            }
        }
        self
    }

    pub(crate) fn append(&mut self, other: Self) {
        if let Some(Ok(sources)) = &mut self.0 {
            match other.finish_loading().0 {
                Some(Ok(extra)) => sources.extend(extra),
                Some(Err(error)) => self.0 = Some(Err(error)),
                None => self.0 = Some(Err("identité du sidecar absente".into())),
            }
        }
    }
}

#[cfg(test)]
thread_local! { static HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Key {
    version: u32,
    prompt_sha256: String,
    tokens: usize,
    sources: Vec<Source>,
    binary_sha256: String,
    mode: String,
    config: String,
    kv_dtype: String,
    runtime_sha256: String,
    little_endian: bool,
}

pub(super) struct Request {
    root: PathBuf,
    key: Key,
}

pub(super) struct State {
    pub(super) cache: CausalDecoderCache,
    pub(super) final_state: Tensor,
    pub(super) history: Option<LayerKvCache>,
    pub(super) captures: Option<PrefixCaptures>,
}

#[derive(Serialize, Deserialize)]
struct Rows {
    base: usize,
    len: usize,
    segments: Vec<Blob>,
}
impl Rows {
    fn write(rows: &kv_rows::KvRows, writer: &mut Writer) -> Result<Self> {
        Ok(Self {
            base: rows.base(),
            len: rows.len(),
            segments: rows
                .snapshot_segments()
                .into_iter()
                .map(|(dtype, bytes)| writer.blob(dtype, bytes))
                .collect::<Result<_>>()?,
        })
    }
    fn read(&self, root: &Path) -> Result<kv_rows::KvRows> {
        if self.base > self.len {
            return Err(invalid("base KV hors longueur"));
        }
        let mut rows = kv_rows::KvRows::default();
        rows.set_base(self.base);
        for blob in &self.segments {
            let bytes = blob.read(root)?;
            match blob.dtype.as_str() {
                "f32" => rows.push_owned(storage::f32_values(&bytes)?),
                "bf16" => rows.push_owned_bf16(storage::bf16_values(&bytes)?),
                _ => return Err(invalid("dtype KV inconnu")),
            }
            if rows.len() > self.len {
                return Err(invalid("segments KV trop longs"));
            }
        }
        if rows.len() != self.len {
            return Err(invalid("segments KV incomplets"));
        }
        Ok(rows)
    }
}

#[derive(Serialize, Deserialize)]
struct Dense {
    shape: Vec<usize>,
    data: Blob,
}
impl Dense {
    fn write(tensor: &Tensor, writer: &mut Writer) -> Result<Self> {
        Ok(Self {
            shape: tensor.shape().to_vec(),
            data: writer.f32(tensor.data())?,
        })
    }
    fn read(&self, root: &Path) -> Result<Tensor> {
        Tensor::from_vec(self.shape.clone(), self.data.f32(root)?)
    }
}
#[derive(Serialize, Deserialize)]
struct Layer {
    keys: Rows,
    values: Rows,
    kv_dim: Option<usize>,
    conv: Blob,
    conv_dim: Option<usize>,
    ssm: Blob,
    ssm_shape: Option<(usize, usize, usize)>,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    metal: Option<MetalLinear>,
}
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Serialize, Deserialize)]
struct MetalLinear {
    layout: crate::metal_backend::LinearSnapshotLayout,
    conv: Blob,
    ssm: Blob,
}
impl Layer {
    fn write(layer: &LayerKvCache, writer: &mut Writer, model: &CausalDecoder) -> Result<Self> {
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let _ = model;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let metal = if let Some(state) = layer.linear.metal_state() {
            let executor = model
                .forward_runtime()
                .metal_executor()
                .ok_or_else(|| invalid("executor absent"))?;
            let (layout, conv, ssm) = state.disk_parts(executor)?;
            let dtype = if layout.ssm_bf16 { "bf16" } else { "f32" };
            Some(MetalLinear {
                layout,
                conv: writer.f32(&conv)?,
                ssm: writer.blob(dtype, &ssm)?,
            })
        } else {
            None
        };
        Ok(Self {
            keys: Rows::write(&layer.keys, writer)?,
            values: Rows::write(&layer.values, writer)?,
            kv_dim: layer.kv_dim,
            conv: writer.f32(&layer.linear.conv)?,
            conv_dim: layer.linear.conv_dim,
            ssm: writer.f32(&layer.linear.ssm)?,
            ssm_shape: layer.linear.ssm_shape,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            metal,
        })
    }
    fn read(&self, root: &Path) -> Result<LayerKvCache> {
        let keys = self.keys.read(root)?;
        let values = self.values.read(root)?;
        if keys.len() != values.len()
            || keys.base() != values.base()
            || (keys.len() > 0
                && self
                    .kv_dim
                    .is_none_or(|dim| dim == 0 || keys.len() % dim != 0 || keys.base() % dim != 0))
        {
            return Err(invalid("dimensions K/V incohérentes"));
        }
        let mut layer = LayerKvCache {
            keys,
            values,
            kv_dim: self.kv_dim,
            ..Default::default()
        };
        layer.linear.conv = self.conv.f32(root)?;
        layer.linear.conv_dim = self.conv_dim;
        layer.linear.ssm = self.ssm.f32(root)?;
        layer.linear.ssm_shape = self.ssm_shape;
        if let Some((a, b, c)) = self.ssm_shape {
            if a.checked_mul(b).and_then(|n| n.checked_mul(c)) != Some(layer.linear.ssm.len()) {
                return Err(invalid("dimensions SSM CPU incohérentes"));
            }
        }
        Ok(layer)
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn read_metal(
        &self,
        root: &Path,
        model: &CausalDecoder,
    ) -> Result<Option<crate::metal_backend::LinearAttentionMetalState>> {
        self.metal
            .as_ref()
            .map(|state| {
                let expected = if state.layout.ssm_bf16 { "bf16" } else { "f32" };
                if state.ssm.dtype != expected {
                    return Err(invalid("dtype SSM incohérent"));
                }
                let metal = model
                    .forward_runtime()
                    .metal_executor()
                    .ok_or_else(|| invalid("executor absent"))?;
                metal.linear_state_from_disk(
                    &state.layout,
                    &state.conv.f32(root)?,
                    &state.ssm.read(root)?,
                )
            })
            .transpose()
    }
}
#[derive(Serialize, Deserialize)]
struct Captures {
    layer_ids: Vec<usize>,
    from: usize,
    hidden: Dense,
}
#[derive(Serialize, Deserialize)]
struct Manifest {
    key: Key,
    position: usize,
    layers: Vec<Layer>,
    final_state: Dense,
    history: Option<Layer>,
    captures: Option<Captures>,
}

impl CausalDecoder {
    pub(crate) fn with_snapshot_sources(mut self, sources: SnapshotSources) -> Self {
        self.snapshot_sources = sources.finish_loading();
        self
    }

    pub(super) fn snapshot_request(&self, prompt: &[usize], mode: &str) -> Result<Option<Request>> {
        let Some(root) = crate::runtime_flags::state_snapshot_dir() else {
            return Ok(None);
        };
        let sources = match self.snapshot_sources.verified() {
            Ok(sources) => sources,
            Err(error) => {
                eprintln!("state_snapshot refusé : {error}; prefill normal, sans écriture");
                return Ok(None);
            }
        };
        let mut hash = Sha256::new();
        for &token in prompt {
            hash.update((token as u64).to_le_bytes());
        }
        let exe = io(Path::new("current_exe"), std::env::current_exe())?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let kv_dtype = format!(
            "resident_bf16={} quant={:?} prefill_bf16={}",
            Self::resident_kv_bf16(false, prompt.len()),
            crate::runtime_flags::kv_quant_format(),
            crate::runtime_flags::prefill_kv_bf16_store_enabled()
        );
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let kv_dtype = "f32".to_string();
        let mut settings = crate::runtime_flags::state_snapshot_settings();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(metal) = self.forward_runtime().metal_executor() {
            settings.push(("kernel_profile", metal.snapshot_kernel_settings()));
        }
        // Permet aussi la compilation CPU, sans modification conditionnelle.
        settings.sort_by_key(|(name, _)| *name);
        let prompt_sha256 = format!("{:x}", hash.finalize());
        // Une bascule DFlash vers MTP préremplit un autre prompt : son écriture
        // ne doit pas remplacer l'instantané du prompt initial à accélérer.
        let entry = storage::digest(format!("{prompt_sha256}:{mode}").as_bytes());
        Ok(Some(Request {
            root: root.join(format!("entry-{entry}")),
            key: Key {
                version: 2,
                prompt_sha256,
                tokens: prompt.len(),
                sources: sources.to_vec(),
                binary_sha256: storage::hash_file(&exe)?,
                mode: mode.into(),
                config: format!(
                    "{:?};metal={}",
                    self.config,
                    cfg!(all(target_os = "macos", feature = "metal")) && self.snapshot_has_metal()
                ),
                kv_dtype,
                runtime_sha256: storage::digest(
                    &serde_json::to_vec(&settings).map_err(|e| invalid(e.to_string()))?,
                ),
                little_endian: cfg!(target_endian = "little"),
            },
        }))
    }

    fn snapshot_has_metal(&self) -> bool {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            self.forward_runtime().metal_executor().is_some()
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            false
        }
    }

    #[cfg(feature = "devtools")]
    pub(super) fn snapshot_request_dflash(
        &self,
        prompt: &[usize],
        mode: &str,
        drafter: Option<&crate::DFlashDraft>,
    ) -> Result<Option<Request>> {
        let mut request = self.snapshot_request(prompt, mode)?;
        if let (Some(request), Some(draft)) = (&mut request, drafter) {
            match draft.snapshot_sources.verified() {
                Ok(sources) => {
                    request.key.sources.extend_from_slice(sources);
                    request
                        .key
                        .config
                        .push_str(&format!(";draft={:?};quant={:?}", draft.info, draft.quant));
                }
                Err(error) => {
                    eprintln!("state_snapshot refusé : {error}; prefill normal, sans écriture");
                    return Ok(None);
                }
            }
        }
        Ok(request)
    }

    pub(super) fn snapshot_load(&self, request: &Option<Request>) -> Option<State> {
        let request = request.as_ref()?;
        match self.snapshot_read(request) {
            Ok(state) => {
                #[cfg(test)]
                HITS.set(HITS.get() + 1);
                eprintln!(
                    "state_snapshot hit mode={} tokens={} (prefill sauté)",
                    request.key.mode, request.key.tokens
                );
                Some(state)
            }
            Err(error) => {
                eprintln!("state_snapshot refusé : {error}; prefill normal");
                None
            }
        }
    }

    fn snapshot_read(&self, request: &Request) -> Result<State> {
        for source in &request.key.sources {
            source.verify()?;
        }
        let path = request.root.join("manifest.json");
        let file = io(&path, std::fs::File::open(&path))?;
        if io(&path, file.metadata())?.len() > 16 * 1024 * 1024 {
            return Err(invalid("manifeste trop grand"));
        }
        let manifest: Manifest =
            serde_json::from_reader(std::io::Read::take(file, 16 * 1024 * 1024 + 1))
                .map_err(|source| InferError::Json { path, source })?;
        if manifest.key.binary_sha256 != request.key.binary_sha256 {
            return Err(invalid("binaire différent (sha256), autre build refusé"));
        }
        if manifest.key != request.key {
            return Err(invalid(
                "clé différente (prompt, poids, mode, dtype ou réglages)",
            ));
        }
        if manifest.position != request.key.tokens || manifest.layers.len() != self.layers.len() {
            return Err(invalid("position ou nombre de couches incorrect"));
        }
        manifest.validate(self, request.key.tokens)?;
        let root = &request.root;
        let mut cache = self.empty_cache();
        cache.layers = manifest
            .layers
            .iter()
            .map(|layer| layer.read(root))
            .collect::<Result<_>>()?;
        cache.position = manifest.position;
        let final_state = manifest.final_state.read(root)?;
        if final_state.shape() != [1, self.final_norm.data().len()] {
            return Err(invalid("dernière ligne invalide"));
        }
        let history = manifest
            .history
            .as_ref()
            .map(|layer| layer.read(root))
            .transpose()?;
        if request.key.mode == "mtp-committed" && history.is_none() {
            return Err(invalid("historique MTP absent"));
        }
        if request.key.mode.starts_with("dflash") && manifest.captures.is_none() {
            return Err(invalid("captures DFlash absentes"));
        }
        let captures = manifest
            .captures
            .as_ref()
            .map(|c| -> Result<_> {
                let hidden = c.hidden.read(root)?;
                let (rows, cols) = hidden.as_matrix()?;
                if c.from.checked_add(rows) != Some(cache.position)
                    || c.layer_ids.len().checked_mul(self.final_norm.data().len()) != Some(cols)
                    || c.layer_ids.iter().any(|&i| i >= self.layers.len())
                {
                    return Err(invalid("captures DFlash incohérentes"));
                }
                Ok(PrefixCaptures::new(&c.layer_ids, c.from, hidden))
            })
            .transpose()?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let linear = manifest
                .layers
                .iter()
                .map(|layer| layer.read_metal(root, self))
                .collect::<Result<Vec<_>>>()?;
            if linear.iter().any(Option::is_some) {
                self.restore_prefix_cache_linear_metal(&mut cache, linear)?;
            }
        }
        for source in &request.key.sources {
            source.verify()?;
        }
        Ok(State {
            cache,
            final_state,
            history,
            captures,
        })
    }

    pub(super) fn snapshot_save(
        &self,
        request: &Option<Request>,
        cache: &CausalDecoderCache,
        final_state: &Tensor,
        history: Option<&LayerKvCache>,
        captures: Option<&PrefixCaptures>,
    ) -> Result<()> {
        let Some(request) = request else {
            return Ok(());
        };
        if let Err(error) = request.key.sources.iter().try_for_each(Source::verify) {
            eprintln!("state_snapshot refusé : {error}; sans écriture");
            return Ok(());
        }
        let mut writer = Writer::new(&request.root)?;
        let layers = cache
            .layers
            .iter()
            .map(|layer| Layer::write(layer, &mut writer, self))
            .collect::<Result<_>>()?;
        let history = history
            .map(|layer| Layer::write(layer, &mut writer, self))
            .transpose()?;
        let captures = captures
            .map(|c| -> Result<_> {
                Ok(Captures {
                    layer_ids: c.layer_ids.clone(),
                    from: c.from,
                    hidden: Dense::write(&c.hidden, &mut writer)?,
                })
            })
            .transpose()?;
        let key = request.key.clone();
        let manifest = Manifest {
            key,
            position: cache.position,
            layers,
            final_state: Dense::write(final_state, &mut writer)?,
            history,
            captures,
        };
        if let Err(error) = request.key.sources.iter().try_for_each(Source::verify) {
            eprintln!("state_snapshot refusé : {error}; sans publication");
            return Ok(());
        }
        writer.publish(&manifest)?;
        eprintln!(
            "state_snapshot écrit mode={} tokens={}",
            request.key.mode, cache.position
        );
        Ok(())
    }
}
