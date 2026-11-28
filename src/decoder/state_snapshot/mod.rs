//! Instantané diagnostic post-prefill. Le decode reprend le format du cache de
//! préfixe ; seules les données persistantes traversent la frontière disque.
pub(crate) mod persistent;
mod sources;
mod storage;
#[cfg(test)]
mod tests;
mod validation;
use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sources::Source;
use std::path::{Path, PathBuf};
use storage::{invalid, io, Blob, Writer};

const FORMAT_VERSION: u32 = 5;
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum SnapshotMode {
    Ar,
    MtpFresh,
    MtpCommitted,
    DflashReference,
    Dflash(String),
}

impl std::fmt::Display for SnapshotMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Ar => "ar",
            Self::MtpFresh => "mtp-fresh",
            Self::MtpCommitted => "mtp-committed",
            Self::DflashReference => "dflash-reference",
            Self::Dflash(drafter) => return write!(f, "dflash:{drafter}"),
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum BlobDtype {
    F32,
    Bf16,
}

impl BlobDtype {
    fn width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

/// Contenu et métadonnées figés avant la lecture des poids, uniquement en mode diagnostic.
#[derive(Clone, Debug, Default)]
pub(crate) struct SnapshotSources(Option<std::result::Result<Vec<Source>, String>>);

impl SnapshotSources {
    pub(crate) fn capture<'a>(paths: impl IntoIterator<Item = &'a Path>) -> Self {
        if crate::runtime_flags::state_snapshot_dir().is_none() && !persistent::enabled() {
            return Self::default();
        }
        Self(Some(
            (|| -> Result<Vec<Source>> {
                snapshot_calculations_complete()?;
                let mut sources = Vec::new();
                for path in paths {
                    if !sources.iter().any(|source: &Source| source.path() == path) {
                        sources.push(Source::read(path)?);
                    }
                    if let Some(parent) = path.parent() {
                        for name in [
                            "model.safetensors.index.json",
                            "config.json",
                            "tokenizer.json",
                            "tokenizer_config.json",
                            "special_tokens_map.json",
                        ] {
                            let auxiliary = parent.join(name);
                            if auxiliary.is_file()
                                && !sources.iter().any(|source| source.path() == auxiliary)
                            {
                                sources.push(Source::read(&auxiliary)?);
                            }
                        }
                    }
                }
                Ok(sources)
            })()
            .map_err(|e| e.to_string()),
        ))
    }

    /// Comme [`Self::capture`], suivi des sources déjà lues en mémoire
    /// (`read`), dont l'empreinte vient des octets effectivement consommés.
    pub(crate) fn capture_with_read<'a>(
        paths: impl IntoIterator<Item = &'a Path>,
        read: &[ReadSource],
    ) -> Self {
        let mut sources = Self::capture(paths);
        if let Some(Ok(list)) = &mut sources.0 {
            for source in read {
                match &source.0 {
                    Some(Ok(source)) => list.push(source.clone()),
                    Some(Err(error)) => {
                        sources.0 = Some(Err(error.clone()));
                        break;
                    }
                    None => {
                        sources.0 = Some(Err("identité d'une source lue absente".into()));
                        break;
                    }
                }
            }
        }
        sources
    }

    fn verified(&self) -> Result<&[Source]> {
        snapshot_calculations_complete()?;
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

/// Identité d'un fichier lu entièrement en mémoire, prise au moment de la
/// lecture. Vide quand aucun instantané n'est actif : rien n'est haché.
#[derive(Clone, Debug, Default)]
pub(crate) struct ReadSource(Option<std::result::Result<Source, String>>);

impl ReadSource {
    /// `before` est la métadonnée prise avant de lire `bytes` depuis `path`.
    pub(crate) fn new(path: &Path, before: &std::fs::Metadata, bytes: &[u8]) -> Self {
        if crate::runtime_flags::state_snapshot_dir().is_none() && !persistent::enabled() {
            return Self::default();
        }
        Self(Some(
            Source::from_read(path, before, bytes).map_err(|e| e.to_string()),
        ))
    }
}

/// Les ablations ne doivent ni relire un calcul complet ni publier un calcul tronqué.
fn snapshot_calculations_complete() -> Result<()> {
    if let Some(flag) = crate::runtime_flags::state_snapshot_ablation() {
        return Err(invalid(format!(
            "diagnostic {flag} observé actif, instantané désactivé jusqu'à la fin du processus"
        )));
    }
    Ok(())
}

#[cfg(test)]
thread_local! { static HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Key {
    version: u32,
    prompt_sha256: String,
    position_sha256: Option<String>,
    tokens: usize,
    sources: Vec<Source>,
    binary_sha256: String,
    mode: SnapshotMode,
    config: String,
    kv_dtype: String,
    runtime_sha256: String,
    little_endian: bool,
}

#[derive(Clone)]
pub(super) struct Request {
    root: PathBuf,
    key: Key,
}

pub(super) struct State {
    pub(super) cache: CausalDecoderCache,
    pub(super) final_state: Tensor,
    pub(super) history: Option<LayerKvCache>,
    #[cfg_attr(
        not(feature = "devtools"),
        allow(
            dead_code,
            reason = "Format commun des snapshots avec ou sans capture DFlash"
        )
    )]
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
            match blob.dtype {
                BlobDtype::F32 => rows.push_owned(storage::f32_values(&bytes)?),
                BlobDtype::Bf16 => rows.push_owned_bf16(storage::bf16_values(&bytes)?),
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
    #[cfg(gpu_metal)]
    metal: Option<MetalLinear>,
}
#[cfg(gpu_metal)]
#[derive(Serialize, Deserialize)]
struct MetalLinear {
    layout: crate::metal_backend::LinearSnapshotLayout,
    conv: Blob,
    ssm: Blob,
}
impl Layer {
    fn write(layer: &LayerKvCache, writer: &mut Writer, model: &CausalDecoder) -> Result<Self> {
        #[cfg(not(gpu_metal))]
        let _ = model;
        #[cfg(gpu_metal)]
        let metal = if let Some(state) = layer.linear.metal_state() {
            let executor = model
                .forward_runtime()
                .metal_executor()
                .ok_or_else(|| invalid("executor absent"))?;
            let (layout, conv, ssm) = state.disk_parts(executor)?;
            let dtype = if layout.ssm_bf16 {
                BlobDtype::Bf16
            } else {
                BlobDtype::F32
            };
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
            #[cfg(gpu_metal)]
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
    #[cfg(gpu_metal)]
    fn read_metal(
        &self,
        root: &Path,
        model: &CausalDecoder,
    ) -> Result<Option<crate::metal_backend::LinearAttentionMetalState>> {
        self.metal
            .as_ref()
            .map(|state| {
                let expected = if state.layout.ssm_bf16 {
                    BlobDtype::Bf16
                } else {
                    BlobDtype::F32
                };
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
    rope_delta: i64,
    vision_positions: Option<crate::vision::prompt::VisionPositions>,
    layers: Vec<Layer>,
    final_state: Dense,
    history: Option<Layer>,
    captures: Option<Captures>,
}

/// Majorant sans copie des blobs et du pic de préparation. Seules les tailles
/// KV sont lues ; chaque occurrence sérialisée compte, même aliasée.
fn staging_estimate(
    cache: &CausalDecoderCache,
    final_state: &Tensor,
    history: Option<&LayerKvCache>,
    captures: Option<&PrefixCaptures>,
) -> (u64, u64) {
    let mut blobs = (final_state.data().len() as u64).saturating_mul(4);
    #[cfg(gpu_metal)]
    let mut scratch = 0u64;
    #[cfg(not(gpu_metal))]
    let scratch = 0u64;
    for layer in cache.layers.iter().chain(history) {
        for rows in [&layer.keys, &layer.values] {
            blobs = blobs.saturating_add(rows.snapshot_bytes());
        }
        blobs = blobs.saturating_add(
            (layer.linear.conv.len() as u64)
                .saturating_add(layer.linear.ssm.len() as u64)
                .saturating_mul(4),
        );
        #[cfg(gpu_metal)]
        if let Some(state) = layer.linear.metal_state() {
            let bytes = state.estimated_bytes() as u64;
            blobs = blobs.saturating_add(bytes);
            // disk_parts : blit GPU, lecture CPU et conversion temporaire.
            scratch = scratch.max(bytes.saturating_mul(3));
        }
    }
    if let Some(captures) = captures {
        blobs = blobs.saturating_add((captures.hidden.data().len() as u64).saturating_mul(4));
    }
    let disk = blobs.saturating_add(MAX_MANIFEST_BYTES).saturating_add(64);
    // Manifeste, métadonnées et réécriture JSON du worker : marge conservatrice.
    let ram = blobs
        .saturating_add(scratch)
        .saturating_add(MAX_MANIFEST_BYTES * 8);
    (disk, ram)
}

impl CausalDecoder {
    pub(crate) fn with_snapshot_sources(mut self, sources: SnapshotSources) -> Self {
        self.snapshot_sources = sources.finish_loading();
        self
    }

    pub(super) fn snapshot_request(
        &self,
        prompt: &[usize],
        mode: SnapshotMode,
    ) -> Result<Option<Request>> {
        self.snapshot_request_positioned(prompt, mode, None)
    }

    pub(super) fn snapshot_request_positioned(
        &self,
        prompt: &[usize],
        mode: SnapshotMode,
        positions: Option<&crate::vision::prompt::VisionPositions>,
    ) -> Result<Option<Request>> {
        if let Some(positions) = positions {
            positions.delta_at(prompt.len())?;
        }
        if crate::runtime_flags::state_snapshot_warmup() {
            return Ok(None);
        }
        let Some(root) = persistent::store()
            .map(|store| store.root.as_path())
            .or_else(crate::runtime_flags::state_snapshot_dir)
        else {
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
        #[cfg(gpu_metal)]
        let kv_dtype = format!(
            "resident_bf16={} quant={:?} prefill_bf16={} mtp_history_bf16_v1={}",
            Self::resident_kv_bf16(false, prompt.len()),
            crate::runtime_flags::kv_quant_format(),
            crate::runtime_flags::prefill_kv_bf16_store_enabled(),
            super::mtp_history::bf16_capacity().is_some()
        );
        #[cfg(not(gpu_metal))]
        let kv_dtype = "f32".to_string();
        let environment = crate::runtime_flags::environment_sha256();
        let overrides = crate::runtime_flags::state_snapshot_overrides();
        #[cfg(gpu_metal)]
        let kernel_profile = self
            .forward_runtime()
            .metal_executor()
            .map(|metal| metal.snapshot_kernel_profile());
        #[cfg(not(gpu_metal))]
        let kernel_profile: Option<String> = None;
        let position_sha256 = position_digest(positions, prompt.len())?;
        if let Some(identity) = &position_sha256 {
            hash.update(identity.as_bytes());
        }
        let prompt_sha256 = crate::hex(hash.finalize());
        // Une bascule DFlash vers MTP préremplit un autre prompt : son écriture
        // ne doit pas remplacer l'instantané du prompt initial à accélérer.
        let entry = storage::digest(format!("{prompt_sha256}:{mode}").as_bytes());
        let mut request = Request {
            root: root.join(format!("entry-{entry}")),
            key: Key {
                version: FORMAT_VERSION,
                prompt_sha256,
                position_sha256,
                tokens: prompt.len(),
                sources: sources.to_vec(),
                binary_sha256: storage::binary_sha256()?.to_owned(),
                mode,
                config: format!(
                    "{:?};metal={}",
                    self.config,
                    cfg!(gpu_metal) && self.snapshot_has_metal()
                ),
                kv_dtype,
                runtime_sha256: storage::digest(
                    format!("{environment}:{kernel_profile:?}:{overrides}").as_bytes(),
                ),
                little_endian: cfg!(target_endian = "little"),
            },
        };
        if persistent::enabled() {
            let key =
                serde_json::to_vec(&request.key).map_err(|error| invalid(error.to_string()))?;
            let identity = storage::digest(&key);
            let name = storage::digest(format!("{identity}:{}", persistent::session()).as_bytes());
            request.root = root.join(format!("entry-{name}"));
        }
        Ok(Some(request))
    }

    fn snapshot_has_metal(&self) -> bool {
        #[cfg(gpu_metal)]
        {
            self.forward_runtime().metal_executor().is_some()
        }
        #[cfg(not(gpu_metal))]
        {
            false
        }
    }

    #[cfg(feature = "devtools")]
    pub(super) fn snapshot_request_dflash(
        &self,
        prompt: &[usize],
        mode: SnapshotMode,
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
        if let (Some(request), Some(store)) = (&mut request, persistent::store()) {
            let key =
                serde_json::to_vec(&request.key).map_err(|error| invalid(error.to_string()))?;
            let name = storage::digest(
                format!("{}:{}", storage::digest(&key), persistent::session()).as_bytes(),
            );
            request.root = store.root.join(format!("entry-{name}"));
        }
        Ok(request)
    }

    pub(super) fn snapshot_load(&self, request: &Option<Request>) -> Option<State> {
        if crate::runtime_flags::state_snapshot_warmup() {
            return None;
        }
        let request = request.as_ref()?;
        if let Some(store) =
            persistent::store().filter(|store| request.root.starts_with(&store.root))
        {
            let state = store.read(self, request);
            #[cfg(test)]
            if state.is_some() {
                HITS.set(HITS.get() + 1);
            }
            return state;
        }
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
            Err(InferError::Io { path, source })
                if path == request.root.join("manifest.json")
                    && source.kind() == std::io::ErrorKind::NotFound =>
            {
                eprintln!(
                    "state_snapshot absent mode={} tokens={}",
                    request.key.mode, request.key.tokens
                );
                None
            }
            Err(error) => {
                eprintln!("state_snapshot refusé : {error}; prefill normal");
                None
            }
        }
    }

    fn snapshot_read(&self, request: &Request) -> Result<State> {
        snapshot_calculations_complete()?;
        for source in &request.key.sources {
            source.verify()?;
        }
        let path = request.root.join("manifest.json");
        let file = io(&path, std::fs::File::open(&path))?;
        if io(&path, file.metadata())?.len() > MAX_MANIFEST_BYTES {
            return Err(invalid("manifeste trop grand"));
        }
        let manifest: Manifest =
            serde_json::from_reader(std::io::Read::take(file, MAX_MANIFEST_BYTES + 1))
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
        if position_digest(manifest.vision_positions.as_ref(), request.key.tokens)?
            != request.key.position_sha256
        {
            return Err(invalid("contexte positionnel différent de la clé"));
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
        cache.rope_delta = manifest.rope_delta;
        cache.vision_positions = manifest.vision_positions;
        cache.current_pos()?;
        let final_state = manifest.final_state.read(root)?;
        if final_state.shape() != [1, self.final_norm.data().len()] {
            return Err(invalid("dernière ligne invalide"));
        }
        let history = manifest
            .history
            .as_ref()
            .map(|layer| layer.read(root))
            .transpose()?;
        if request.key.mode == SnapshotMode::MtpCommitted && history.is_none() {
            return Err(invalid("historique MTP absent"));
        }
        if matches!(
            request.key.mode,
            SnapshotMode::DflashReference | SnapshotMode::Dflash(_)
        ) && manifest.captures.is_none()
        {
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
        #[cfg(gpu_metal)]
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
    ) {
        if crate::runtime_flags::state_snapshot_warmup() {
            return;
        }
        let Some(request) = request else {
            return;
        };
        if position_digest(cache.vision_positions.as_ref(), cache.position)
            .ok()
            .as_ref()
            != Some(&request.key.position_sha256)
        {
            eprintln!("state_snapshot refusé : contexte positionnel différent de la clé");
            return;
        }
        let store = persistent::store().filter(|store| request.root.starts_with(&store.root));
        let mut reservation = None;
        let permit = if let Some(store) = store {
            if store.skip_save(request) {
                return;
            }
            let Some(permit) = store.reserve() else {
                eprintln!("prefix_ssd skipped reason=writer_busy");
                return;
            };
            let (disk_bytes, ram_bytes) = staging_estimate(cache, final_state, history, captures);
            match store.admit_staging(request, disk_bytes, ram_bytes) {
                Ok(memory) => reservation = Some(memory),
                Err(error) => {
                    store.refuse(request, &error.to_string());
                    eprintln!("prefix_ssd skipped before_copy: {error}");
                    return;
                }
            }
            Some(permit)
        } else {
            None
        };
        let result = (|| -> Result<()> {
            snapshot_calculations_complete()?;
            request.key.sources.iter().try_for_each(Source::verify)?;
            let mut writer = if let Some(store) = store {
                Writer::buffered(&request.root, store.budget)
            } else {
                Writer::new(&request.root)?
            };
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
                rope_delta: cache
                    .vision_positions
                    .as_ref()
                    .map_or(Ok(cache.rope_delta), |p| p.delta_at(cache.position))?,
                vision_positions: cache
                    .vision_positions
                    .as_ref()
                    .map(|p| p.prefix(cache.position))
                    .transpose()?
                    .flatten(),
                layers,
                final_state: Dense::write(final_state, &mut writer)?,
                history,
                captures,
            };
            snapshot_calculations_complete()?;
            request.key.sources.iter().try_for_each(Source::verify)?;
            if let (Some(store), Some(permit)) = (store, permit) {
                return store.submit_reserved(
                    request,
                    writer.finish_buffered(&manifest)?,
                    permit,
                    reservation,
                );
            }
            writer.publish(&manifest)?;
            eprintln!(
                "state_snapshot écrit mode={} tokens={}",
                request.key.mode, cache.position
            );
            Ok(())
        })();
        if let Err(error) = result {
            if let Some(store) = store {
                store.refuse(request, &error.to_string());
            }
            eprintln!("state_snapshot refusé : {error}; sans écriture");
        }
    }
}

impl CausalDecoder {
    /// Recherche une frontière AR sur SSD sans déclencher de prefill.
    /// Les tokens doivent être le préfixe exact, sous la portée de session serve.
    ///
    /// # Errors
    /// Renvoie une erreur si l'identité du modèle ne peut pas être établie.
    pub fn load_persistent_prompt(
        &self,
        tokens: &[usize],
    ) -> Result<Option<CausalDecoderPromptState>> {
        if !persistent::enabled() {
            return Ok(None);
        }
        let request = self.snapshot_request(tokens, SnapshotMode::Ar)?;
        Ok(self
            .snapshot_load(&request)
            .map(|state| CausalDecoderPromptState::new(state.cache, state.final_state)))
    }

    /// Recherche la frontière persistante la plus longue au-delà de `minimum`.
    /// Seules les longueurs présentes dans l'index disque sont hachées.
    ///
    /// # Errors
    /// Renvoie une erreur si l'identité du modèle ne peut pas être établie.
    pub fn load_persistent_prefix(
        &self,
        tokens: &[usize],
        block: usize,
        minimum: usize,
    ) -> Result<Option<CausalDecoderPromptState>> {
        Ok(self
            .load_persistent_boundary(tokens, block, minimum, SnapshotMode::Ar)?
            .map(|state| CausalDecoderPromptState::new(state.cache, state.final_state)))
    }

    /// Frontière persistante la plus longue de `tokens` dans `mode`, multiple
    /// de `block` et au-delà de `minimum`. Seules les longueurs indexées sont hachées.
    pub(super) fn load_persistent_boundary(
        &self,
        tokens: &[usize],
        block: usize,
        minimum: usize,
        mode: SnapshotMode,
    ) -> Result<Option<State>> {
        let Some(store) = persistent::store() else {
            return Ok(None);
        };
        for length in store.boundaries(tokens.len(), block, minimum) {
            let request = self.snapshot_request(&tokens[..length], mode.clone())?;
            if let Some(state) = self.snapshot_load(&request) {
                return Ok(Some(state));
            }
        }
        Ok(None)
    }

    /// Programme la persistance d'une frontière AR nouvellement étendue.
    ///
    /// # Errors
    /// Renvoie une erreur si l'identité du modèle ne peut pas être établie.
    pub fn persist_prompt(&self, tokens: &[usize], state: &CausalDecoderPromptState) -> Result<()> {
        if !persistent::enabled() {
            return Ok(());
        }
        if tokens.len() != state.position() {
            return Err(invalid("frontière SSD incohérente"));
        }
        let request = self.snapshot_request(tokens, SnapshotMode::Ar)?;
        self.snapshot_save(&request, &state.cache, &state.final_state, None, None);
        Ok(())
    }
}

fn position_digest(
    positions: Option<&crate::vision::prompt::VisionPositions>,
    tokens: usize,
) -> Result<Option<String>> {
    positions
        .map(|p| p.prefix(tokens))
        .transpose()?
        .flatten()
        .map(|positions| {
            serde_json::to_vec(&positions)
                .map(|bytes| storage::digest(&bytes))
                .map_err(|e| invalid(e.to_string()))
        })
        .transpose()
}
