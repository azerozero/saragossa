//! Arrival-Order Speaker Cache (AOSC) et file FIFO de Streaming Sortformer.
//!
//! Port ligne à ligne de `Nemotron3DiarizationSpeakerCache` (transformers 5.22),
//! batch 1. Le mode hors ligne l'utilise aussi : un enregistrement plus long
//! qu'un morceau (340 trames de 80 ms) est découpé et chaque morceau voit le
//! cache des locuteurs et la FIFO des précédents.

/// Politique du cache, lue dans `streaming_config` et les tailles hors ligne.
#[derive(Clone, Debug)]
pub(super) struct CachePolicy {
    pub cache_length: usize,
    pub fifo_length: usize,
    pub update_period: usize,
    pub silence_frames: usize,
    pub score_threshold: f32,
    pub latest_boost: f32,
    pub min_positive: usize,
    pub strong_boosted: usize,
    pub weak_boosted: usize,
    pub num_speakers: usize,
}

/// État du cache : embeddings d'entrée de l'encodeur (avant LayerNorm).
pub(super) struct SpeakerCache {
    policy: CachePolicy,
    hidden: usize,
    cache_embeds: Vec<f32>,
    cache_probs: Vec<f32>,
    fifo: Vec<f32>,
    compressed: bool,
}

impl SpeakerCache {
    pub fn new(policy: CachePolicy, hidden: usize) -> Self {
        Self {
            policy,
            hidden,
            cache_embeds: Vec::new(),
            cache_probs: Vec::new(),
            fifo: Vec::new(),
            compressed: false,
        }
    }

    fn cache_frames(&self) -> usize {
        self.cache_embeds.len() / self.hidden
    }

    fn fifo_frames(&self) -> usize {
        self.fifo.len() / self.hidden
    }

    /// Trames mises en tête de l'entrée : cache puis FIFO.
    pub fn prefix(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.cache_embeds.len() + self.fifo.len());
        out.extend_from_slice(&self.cache_embeds);
        out.extend_from_slice(&self.fifo);
        out
    }

    /// Pousse le morceau dans la FIFO et déverse son trop-plein dans le cache.
    ///
    /// `input` est l'entrée complète du pas (préfixe, retour arrière, morceau,
    /// anticipation), `probs` ses probabilités par trame encodeur `[frames, S]`.
    pub fn update(
        &mut self,
        input: &[f32],
        probs: &[f32],
        silence: &[f32],
        chunk_frames: usize,
        lookback_frames: usize,
    ) {
        let (h, s) = (self.hidden, self.policy.num_speakers);
        let (nc, nf) = (self.cache_frames(), self.fifo_frames());
        let chunk_start = nc + nf + lookback_frames;
        let mut fifo = std::mem::take(&mut self.fifo);
        fifo.extend_from_slice(&input[chunk_start * h..(chunk_start + chunk_frames) * h]);
        let n = fifo.len() / h;
        let popped = if n <= self.policy.fifo_length {
            0
        } else {
            self.policy
                .update_period
                .max(n - self.policy.fifo_length)
                .min(n)
        };
        if popped > 0 {
            let mut fifo_probs = probs[nc * s..(nc + nf) * s].to_vec();
            fifo_probs.extend_from_slice(&probs[chunk_start * s..(chunk_start + chunk_frames) * s]);
            // Un cache non compressé garde des trames ordinaires, réestimées à ce pas.
            let mut cache_probs = if self.compressed {
                std::mem::take(&mut self.cache_probs)
            } else {
                probs[..nc * s].to_vec()
            };
            let mut cache_embeds = std::mem::take(&mut self.cache_embeds);
            cache_embeds.extend_from_slice(&fifo[..popped * h]);
            cache_probs.extend_from_slice(&fifo_probs[..popped * s]);
            fifo.drain(..popped * h);
            if cache_embeds.len() / h > self.policy.cache_length {
                (cache_embeds, cache_probs) = self.compress(&cache_embeds, &cache_probs, silence);
                self.compressed = true;
            }
            self.cache_embeds = cache_embeds;
            self.cache_probs = cache_probs;
        }
        self.fifo = fifo;
    }

    fn frame_scores(&self, probs: &[f32], frames: usize) -> Vec<f32> {
        let s = self.policy.num_speakers;
        let thr = self.policy.score_threshold;
        let ln_half = 0.5_f32.ln();
        let mut scores = vec![0.0_f32; frames * s];
        for t in 0..frames {
            let row = &probs[t * s..(t + 1) * s];
            let complements: Vec<f32> = row.iter().map(|p| (1.0 - p).max(thr).ln()).collect();
            let sum: f32 = complements.iter().sum();
            for k in 0..s {
                let score = row[k].max(thr).ln() - complements[k] + sum - ln_half;
                scores[t * s + k] = if row[k] > 0.5 {
                    score
                } else {
                    f32::NEG_INFINITY
                };
            }
        }
        for k in 0..s {
            let positives = (0..frames).filter(|&t| scores[t * s + k] > 0.0).count();
            if positives >= self.policy.min_positive {
                for t in 0..frames {
                    let p = probs[t * s + k];
                    let v = &mut scores[t * s + k];
                    if *v <= 0.0 && p > 0.5 {
                        *v = f32::NEG_INFINITY;
                    }
                }
            }
        }
        scores
    }

    /// Ajoute `boost` aux `count` meilleurs scores de chaque locuteur.
    fn boost(&self, scores: &mut [f32], frames: usize, count: usize, boost: f32) {
        let s = self.policy.num_speakers;
        for k in 0..s {
            let mut order: Vec<usize> = (0..frames).collect();
            order.sort_by(|&a, &b| {
                scores[b * s + k]
                    .total_cmp(&scores[a * s + k])
                    .then(a.cmp(&b))
            });
            for &t in order.iter().take(count.min(frames)) {
                scores[t * s + k] += boost;
            }
        }
    }

    /// Garde les `cache_length` trames les plus utiles, groupées par locuteur
    /// dans leur ordre d'origine ; les créneaux de silence prennent `silence`.
    fn compress(&self, embeds: &[f32], probs: &[f32], silence: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (h, s) = (self.hidden, self.policy.num_speakers);
        let frames = probs.len() / s;
        let length = self.policy.cache_length;
        let mut scores = self.frame_scores(probs, frames);
        for v in scores.iter_mut().skip(length * s) {
            *v += self.policy.latest_boost;
        }
        let ln_half = 0.5_f32.ln();
        self.boost(
            &mut scores,
            frames,
            self.policy.strong_boosted,
            -2.0 * ln_half,
        );
        self.boost(&mut scores, frames, self.policy.weak_boosted, -ln_half);
        let scored = frames + self.policy.silence_frames;
        // Aplatissement locuteur-majeur, comme `scores.transpose(1, 2).reshape`.
        let flat_score = |i: usize| {
            let (k, t) = (i / scored, i % scored);
            if t < frames {
                scores[t * s + k]
            } else {
                f32::INFINITY
            }
        };
        let mut order: Vec<usize> = (0..scored * s).collect();
        order.sort_by(|&a, &b| flat_score(b).total_cmp(&flat_score(a)).then(a.cmp(&b)));
        let sentinel = scored * s;
        let mut picked: Vec<usize> = order
            .into_iter()
            .take(length)
            .map(|i| {
                if flat_score(i) == f32::NEG_INFINITY {
                    sentinel
                } else {
                    i
                }
            })
            .collect();
        picked.sort_unstable();
        let mut out_embeds = Vec::with_capacity(length * h);
        let mut out_probs = Vec::with_capacity(length * s);
        for i in picked {
            let frame = if i == sentinel {
                frames
            } else {
                (i % scored).min(frames)
            };
            if frame == frames {
                out_embeds.extend_from_slice(silence);
                out_probs.extend(std::iter::repeat_n(0.0, s));
            } else {
                out_embeds.extend_from_slice(&embeds[frame * h..(frame + 1) * h]);
                out_probs.extend_from_slice(&probs[frame * s..(frame + 1) * s]);
            }
        }
        (out_embeds, out_probs)
    }
}
