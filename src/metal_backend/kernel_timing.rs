//! Timestamps GPU opt-in du prefill resident.

use std::collections::HashMap;

use metal::{
    Buffer, CommandBufferRef, ComputeCommandEncoderRef, CounterSampleBuffer,
    CounterSampleBufferDescriptor, Device, MTLCounterSamplingPoint, MTLResourceOptions,
    MTLStorageMode, NSRange,
};

use crate::{InferError, Result};

use super::byte_len;

const COUNTER_ERROR_VALUE: u64 = u64::MAX;

#[derive(Clone, Copy, Debug)]
enum SamplingMode {
    DispatchBoundary,
    StageBoundary,
}

#[derive(Clone, Copy, Debug)]
struct SamplePair {
    label: &'static str,
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct TimingAccum {
    total_ns: u128,
    calls: u64,
}

/// Agrege les timestamps GPU d'un prefill resident.
pub(super) struct PrefillKernelTiming {
    sample_buffer: CounterSampleBuffer,
    resolved_buffer: Buffer,
    sample_count: usize,
    next_sample: usize,
    mode: SamplingMode,
    pairs: Vec<SamplePair>,
}

impl PrefillKernelTiming {
    /// Construit le compteur timestamp si le device l'expose.
    pub(super) fn try_new(device: &Device, layers: usize) -> Option<Self> {
        if !crate::runtime_flags::gpu_timestamps_enabled() {
            return None;
        }
        match Self::new(device, layers) {
            Ok(timing) => Some(timing),
            Err(error) => {
                eprintln!("gpu_timestamps unavailable: {error}");
                None
            }
        }
    }

    fn new(device: &Device, layers: usize) -> Result<Self> {
        let (counter_set, available) = select_timestamp_counter_set(device);
        let counter_set = counter_set.ok_or_else(|| {
            InferError::Metal(format!(
                "aucun MTLCounterSet timestamp disponible (sets: {})",
                if available.is_empty() {
                    "<aucun>".to_string()
                } else {
                    available.join(",")
                }
            ))
        })?;
        let mode = if device.supports_counter_sampling(MTLCounterSamplingPoint::AtDispatchBoundary)
        {
            SamplingMode::DispatchBoundary
        } else if device.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary) {
            eprintln!(
                "gpu_timestamps dispatch_boundary=unsupported; fallback=stage_boundary \
                 overhead=high attribution=not_representative"
            );
            SamplingMode::StageBoundary
        } else {
            return Err(InferError::Metal(
                "counter sampling indisponible aux frontieres dispatch/stage".to_string(),
            ));
        };
        let sample_count = layers
            .checked_mul(96)
            .and_then(|count| count.checked_add(128))
            .ok_or_else(|| InferError::Metal("sample_count GPU timing deborde".to_string()))?;
        let sample_count = sample_count.clamp(256, 8192);
        let descriptor = CounterSampleBufferDescriptor::new();
        descriptor.set_counter_set(&counter_set);
        descriptor.set_sample_count(u64::try_from(sample_count).map_err(|_| {
            InferError::Metal(format!(
                "sample_count GPU timing hors plage: {sample_count}"
            ))
        })?);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_label("reti prefill timestamp counters");
        let sample_buffer = device
            .new_counter_sample_buffer_with_descriptor(&descriptor)
            .map_err(|error| InferError::Metal(format!("MTLCounterSampleBuffer: {error}")))?;
        let resolved_buffer = device.new_buffer(
            byte_len::<u64>(sample_count)?,
            MTLResourceOptions::StorageModeShared,
        );
        eprintln!(
            "gpu_timestamps active: counter_set={} mode={mode:?} samples={sample_count}",
            counter_set.name()
        );
        Ok(Self {
            sample_buffer,
            resolved_buffer,
            sample_count,
            next_sample: 0,
            mode,
            pairs: Vec::new(),
        })
    }

    pub(super) fn uses_dispatch_boundary(&self) -> bool {
        matches!(self.mode, SamplingMode::DispatchBoundary)
    }

    fn reserve_pair(&mut self, label: &'static str) -> Result<SamplePair> {
        let end = self
            .next_sample
            .checked_add(1)
            .ok_or_else(|| InferError::Metal("index sample GPU timing deborde".to_string()))?;
        if end >= self.sample_count {
            return Err(InferError::Metal(format!(
                "MTLCounterSampleBuffer trop petit: besoin index {end}, capacite {}",
                self.sample_count
            )));
        }
        let pair = SamplePair {
            label,
            start: self.next_sample,
            end,
        };
        self.next_sample = end + 1;
        Ok(pair)
    }

    fn encoder_attachment(
        &self,
        pair: SamplePair,
    ) -> metal::ComputePassSampleBufferAttachmentDescriptor {
        let attachment = metal::ComputePassSampleBufferAttachmentDescriptor::new();
        attachment.set_sample_buffer(&self.sample_buffer);
        attachment.set_start_of_encoder_sample_index(pair.start as u64);
        attachment.set_end_of_encoder_sample_index(pair.end as u64);
        attachment
    }

    fn begin_dispatch_sample(&self, encoder: &ComputeCommandEncoderRef, pair: SamplePair) {
        encoder.sample_counters_in_buffer(&self.sample_buffer, pair.start as u64, true);
    }

    fn end_dispatch_sample(&self, encoder: &ComputeCommandEncoderRef, pair: SamplePair) {
        encoder.sample_counters_in_buffer(&self.sample_buffer, pair.end as u64, true);
    }

    fn push_pair(&mut self, pair: SamplePair) {
        self.pairs.push(pair);
    }

    /// Encode la resolution des timestamps vers un buffer CPU-lisible.
    pub(super) fn encode_resolve(&self, command_buffer: &CommandBufferRef) -> Result<()> {
        let blit = command_buffer.new_blit_command_encoder();
        blit.resolve_counters(
            &self.sample_buffer,
            NSRange::new(0, self.sample_count as u64),
            &self.resolved_buffer,
            0,
        );
        blit.end_encoding();
        Ok(())
    }

    /// Imprime la table par label apres `wait_until_completed`.
    #[allow(
        unsafe_code,
        reason = "lecture d'un MTLBuffer Shared de resultats de compteurs apres wait GPU"
    )]
    pub(super) fn dump_report(&self) -> Result<()> {
        let ptr = self.resolved_buffer.contents().cast::<u64>();
        if ptr.is_null() {
            return Err(InferError::Metal(
                "MTLBuffer resultats GPU timing sans pointeur CPU".to_string(),
            ));
        }
        // SAFETY: `resolved_buffer` est en StorageModeShared, dimensionne pour
        // `sample_count` valeurs u64, et le command buffer qui a appele
        // `resolveCounters` est termine avant cette lecture.
        let samples = unsafe { std::slice::from_raw_parts(ptr, self.sample_count) };
        let mut rows: HashMap<&'static str, TimingAccum> = HashMap::new();
        for pair in &self.pairs {
            let Some((&start, &end)) = samples.get(pair.start).zip(samples.get(pair.end)) else {
                continue;
            };
            if start == COUNTER_ERROR_VALUE || end == COUNTER_ERROR_VALUE || end <= start {
                continue;
            }
            let entry = rows.entry(pair.label).or_default();
            entry.total_ns += u128::from(end - start);
            entry.calls += 1;
        }
        let denominator = rows.values().map(|row| row.total_ns).sum::<u128>();
        if denominator == 0 {
            eprintln!("gpu_timestamps no_valid_samples=1");
            return Ok(());
        }
        let mut sorted = rows.into_iter().collect::<Vec<_>>();
        sorted.sort_by_key(|(_, row)| std::cmp::Reverse(row.total_ns));
        eprintln!(
            "gpu_timestamps summary mode={:?} samples={} used_pairs={} total_ms={:.3}",
            self.mode,
            self.next_sample,
            self.pairs.len(),
            denominator as f64 / 1.0e6
        );
        for (label, accum) in sorted {
            let total_ms = accum.total_ns as f64 / 1.0e6;
            let pct = accum.total_ns as f64 * 100.0 / denominator as f64;
            eprintln!(
                "gpu_timestamps label={label} calls={} total_ms={total_ms:.3} pct={pct:.2}",
                accum.calls
            );
        }
        Ok(())
    }
}

/// Execute `encode` entre deux echantillons GPU si le timer est actif.
pub(super) fn time_prefill_pass<F>(
    timing: Option<&mut PrefillKernelTiming>,
    command_buffer: &CommandBufferRef,
    fallback_encoder: Option<&ComputeCommandEncoderRef>,
    label: &'static str,
    encode: F,
) -> Result<()>
where
    F: FnOnce(&ComputeCommandEncoderRef) -> Result<()>,
{
    time_prefill_value(timing, command_buffer, fallback_encoder, label, encode)
}

/// Execute `encode` entre deux echantillons GPU et renvoie sa valeur.
pub(super) fn time_prefill_value<T, F>(
    timing: Option<&mut PrefillKernelTiming>,
    command_buffer: &CommandBufferRef,
    fallback_encoder: Option<&ComputeCommandEncoderRef>,
    label: &'static str,
    encode: F,
) -> Result<T>
where
    F: FnOnce(&ComputeCommandEncoderRef) -> Result<T>,
{
    let Some(timer) = timing else {
        let encoder = fallback_encoder
            .ok_or_else(|| InferError::Metal("encodeur prefill absent hors timing".to_string()))?;
        return encode(encoder);
    };

    let pair = timer.reserve_pair(label)?;
    match timer.mode {
        SamplingMode::DispatchBoundary => {
            let encoder = fallback_encoder.ok_or_else(|| {
                InferError::Metal("encodeur prefill absent pour timestamps dispatch".to_string())
            })?;
            timer.begin_dispatch_sample(encoder, pair);
            let result = encode(encoder);
            timer.end_dispatch_sample(encoder, pair);
            timer.push_pair(pair);
            result
        }
        SamplingMode::StageBoundary => {
            let attachment = timer.encoder_attachment(pair);
            let descriptor = metal::ComputePassDescriptor::new();
            descriptor
                .sample_buffer_attachments()
                .set_object_at(0, Some(&attachment));
            let encoder = command_buffer.compute_command_encoder_with_descriptor(descriptor);
            let result = encode(encoder);
            encoder.end_encoding();
            timer.push_pair(pair);
            result
        }
    }
}

fn select_timestamp_counter_set(device: &Device) -> (Option<metal::CounterSet>, Vec<String>) {
    let mut selected = None;
    let mut available = Vec::new();
    for counter_set in device.counter_sets() {
        let name = counter_set.name().to_string();
        if selected.is_none() && name.to_ascii_lowercase().contains("timestamp") {
            selected = Some(counter_set);
        }
        available.push(name);
    }
    (selected, available)
}
