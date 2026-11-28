//! Profil par trame de la génération TTS (`SARAGOSSA_TTS_INTERNAL_PROFILE=1`).
//!
//! Chaque phase cumule son temps mur CPU ; avec `SARAGOSSA_RUST_DECODE_PROFILE=1`
//! elle cumule aussi les command buffers synchronisés, l'attente CPU bloquée
//! sur le GPU, les dispatches encodés et le temps GPU horodaté par Metal.
//! Ces compteurs sont globaux : avec le codec recouvert
//! (`SARAGOSSA_TTS_CODEC_OVERLAP`), ses command buffers se mêlent aux phases
//! du thread de génération. Désactivé : un seul test de booléen par phase.

use crate::Result;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub(super) enum Phase {
    /// Tirage CPU de cb0 sur les logits talker (pénalité, top-k, top-p).
    Cb0Sample,
    /// Projection CPU `small_to_mtp` des entrées du code predictor.
    CpProjection,
    /// Pas résidents du code predictor (encodage, commit, attente, relecture).
    CpSteps,
    /// Tirage CPU des 15 codebooks.
    CpSample,
    /// Somme CPU des embeddings codec et texte de l'entrée talker suivante.
    InputEmbed,
    /// Pas résident du talker (28 couches et tête dans un command buffer).
    Talker,
    /// Callback de trame : codec streaming et émission PCM.
    Codec,
}

const PHASES: [(&str, Phase); 7] = [
    ("cb0_sample", Phase::Cb0Sample),
    ("cp_projection", Phase::CpProjection),
    ("cp_steps", Phase::CpSteps),
    ("cp_sample", Phase::CpSample),
    ("input_embed", Phase::InputEmbed),
    ("talker", Phase::Talker),
    ("codec", Phase::Codec),
];

#[derive(Clone, Copy, Default)]
struct Totals {
    wall: Duration,
    command_buffers: u64,
    dispatches: u64,
    wait_ns: u64,
    gpu_ns: u64,
}

pub(super) struct FrameProfile {
    enabled: bool,
    totals: [Totals; PHASES.len()],
    started: Option<Instant>,
    /// Dispatches par site source et par phase, sous
    /// `SARAGOSSA_RUST_DECODE_PROFILE_SITES=1`.
    sites: Option<Vec<SiteCounts>>,
}

#[cfg(gpu_metal)]
type SiteCounts = std::collections::HashMap<crate::metal_backend::DispatchProfileSite, u64>;
#[cfg(not(gpu_metal))]
type SiteCounts = ();

#[cfg(gpu_metal)]
fn sites_snapshot(enabled: bool) -> Option<SiteCounts> {
    enabled.then(crate::metal_backend::decode_profile_dispatch_sites_snapshot)
}

#[cfg(not(gpu_metal))]
fn sites_snapshot(_enabled: bool) -> Option<SiteCounts> {
    None
}

#[cfg(gpu_metal)]
fn gpu_counters() -> [u64; 4] {
    let (command_buffers, wait_ns, _, dispatches) = crate::metal_backend::decode_profile_snapshot();
    [
        command_buffers,
        dispatches,
        wait_ns,
        crate::metal_backend::decode_profile_gpu_ns(),
    ]
}

#[cfg(not(gpu_metal))]
fn gpu_counters() -> [u64; 4] {
    [0; 4]
}

impl FrameProfile {
    pub(super) fn new() -> Self {
        let enabled = super::tts_internal_profile_enabled();
        let sites =
            enabled && crate::runtime_flags::env_flag("SARAGOSSA_RUST_DECODE_PROFILE_SITES", false);
        Self {
            enabled,
            totals: [Totals::default(); PHASES.len()],
            started: enabled.then(Instant::now),
            sites: sites.then(|| vec![SiteCounts::default(); PHASES.len()]),
        }
    }

    /// Exécute `work` en imputant son coût à `phase`.
    pub(super) fn time<T>(&mut self, phase: Phase, work: impl FnOnce() -> Result<T>) -> Result<T> {
        if !self.enabled {
            return work();
        }
        let sites_before = sites_snapshot(self.sites.is_some());
        let before = gpu_counters();
        let started = Instant::now();
        let out = work();
        let wall = started.elapsed();
        let after = gpu_counters();
        #[cfg(gpu_metal)]
        if let (Some(before), Some(sites)) = (sites_before, self.sites.as_mut()) {
            let counts = &mut sites[phase as usize];
            for (site, count) in crate::metal_backend::decode_profile_dispatch_sites_snapshot() {
                let delta = count.saturating_sub(before.get(&site).copied().unwrap_or(0));
                if delta > 0 {
                    *counts.entry(site).or_insert(0) += delta;
                }
            }
        }
        #[cfg(not(gpu_metal))]
        let _ = sites_before;
        let delta = |index: usize| after[index].saturating_sub(before[index]);
        let totals = &mut self.totals[phase as usize];
        totals.wall += wall;
        totals.command_buffers += delta(0);
        totals.dispatches += delta(1);
        totals.wait_ns += delta(2);
        totals.gpu_ns += delta(3);
        out
    }

    /// Imprime le coût moyen par trame de chaque phase sur stderr.
    pub(super) fn report(&self, mode: &str, frames: usize) {
        let Some(started) = self.started else {
            return;
        };
        let per = |value: f64| value / frames.max(1) as f64;
        let mut line = format!(
            "perf tts.frame mode={mode} frames={frames} total_ms_per_frame={:.3}",
            per(started.elapsed().as_secs_f64() * 1e3)
        );
        for (name, phase) in PHASES {
            let totals = self.totals[phase as usize];
            line.push_str(&format!(
                " {name}=wall:{:.3},cb:{:.2},disp:{:.0},wait:{:.3},gpu:{:.3}",
                per(totals.wall.as_secs_f64() * 1e3),
                per(totals.command_buffers as f64),
                per(totals.dispatches as f64),
                per(totals.wait_ns as f64 / 1e6),
                per(totals.gpu_ns as f64 / 1e6),
            ));
        }
        eprintln!("{line}");
        #[cfg(gpu_metal)]
        if let Some(sites) = &self.sites {
            for ((name, _), counts) in PHASES.iter().zip(sites) {
                let mut rows = counts.iter().collect::<Vec<_>>();
                rows.sort_by(|left, right| right.1.cmp(left.1));
                for (site, count) in rows {
                    eprintln!(
                        "perf tts.frame.site phase={name} per_frame={:.1} at {}:{}",
                        per(*count as f64),
                        site.file,
                        site.line
                    );
                }
            }
        }
    }
}
