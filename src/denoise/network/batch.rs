//! Pas du réseau sur un bloc de trames, multi-cœur et identique bit à bit à
//! [`Network::step`] trame par trame.
//!
//! Seuls les GRU portent un état d'une trame à l'autre. L'encodeur (convolutions
//! causales sur des fenêtres d'entrée connues), les portes d'entrée des GRU et
//! les décodeurs se calculent donc pour toutes les trames en parallèle ; la
//! partie récurrente des GRU reste séquentielle. Chaque trame exécute les mêmes
//! fonctions sur les mêmes entrées que le chemin trame par trame : seul l'ordre
//! entre trames indépendantes change.

use std::thread;

use super::{ConvHist, Encoded, NetState, Network, StepOut};
use crate::denoise::dsp::{NB_DF, NB_ERB};

impl Network {
    /// Équivalent de `k` appels à [`Network::step`], `erb` et `cplx` concaténant
    /// les `k` trames d'entrée.
    pub(crate) fn step_batch(
        &self,
        st: &mut NetState,
        erb: &[f32],
        cplx: &[f32],
        threads: usize,
    ) -> Vec<StepOut> {
        let (k, math) = (erb.len() / NB_ERB, st.math);
        // Les convolutions internes à noyau temporel > 1 porteraient un état
        // d'une trame à l'autre : ce chemin ne couvre que DFN3 standard.
        debug_assert!(
            self.is_std(),
            "invariant: pas multi-cœur réservé à DFN3 standard"
        );
        if k == 0 {
            return Vec::new();
        }
        let enc: Vec<Encoded> = {
            let erb = window_source(&st.erb_hist, erb, NB_ERB);
            let spec = window_source(&st.spec_hist, cplx, 2 * NB_DF);
            let (ke, ks) = (st.erb_hist.len(), st.spec_hist.len());
            par_map(k, threads, |t| {
                self.encode(
                    &erb[t..t + ke],
                    &spec[t..t + ks],
                    &mut ConvHist::default(),
                    math,
                )
            })
        };
        slide(&mut st.erb_hist, erb, NB_ERB);
        slide(&mut st.spec_hist, cplx, 2 * NB_DF);

        let emb: Vec<Vec<f32>> = enc
            .iter()
            .map(|e| {
                let mut out = Vec::new();
                self.enc_gru.recur(&e.gates, &mut st.h_enc, &mut out, math);
                out
            })
            .collect();
        let gates = par_map(k, threads, |t| {
            (
                self.erb_gru.input_gates(&emb[t]),
                self.df_gru.input_gates(&emb[t]),
            )
        });
        // Les deux GRU de décodeur ne dépendent que de `emb` : un cœur chacun.
        let (h_erb, h_df) = (&mut st.h_erb, &mut st.h_df);
        let (d, c) = thread::scope(|scope| {
            let df = scope.spawn(|| {
                gates
                    .iter()
                    .map(|(_, g)| {
                        let mut out = Vec::new();
                        self.df_gru.recur(g, h_df, &mut out, math);
                        out
                    })
                    .collect::<Vec<_>>()
            });
            let d: Vec<Vec<f32>> = gates
                .iter()
                .map(|(g, _)| {
                    let mut out = Vec::new();
                    self.erb_gru.recur(g, h_erb, &mut out, math);
                    out
                })
                .collect();
            (d, join(df))
        });

        let c0: Vec<&[f32]> = st.c0_hist[1..]
            .iter()
            .map(Vec::as_slice)
            .chain(enc.iter().map(|e| e.c0.as_slice()))
            .collect();
        let kc = st.c0_hist.len();
        let out = par_map(k, threads, |t| {
            self.decode(
                &enc[t],
                &c0[t..t + kc],
                &emb[t],
                (&d[t], &c[t]),
                &mut ConvHist::default(),
                math,
            )
        });
        let tail: Vec<Vec<f32>> = c0[c0.len() - kc..].iter().map(|s| s.to_vec()).collect();
        st.c0_hist = tail;
        out
    }
}

/// Fenêtres causales : `hist[1..]` (les `kt − 1` trames passées) suivies des nouvelles trames.
fn window_source<'a>(hist: &'a [Vec<f32>], frames: &'a [f32], len: usize) -> Vec<&'a [f32]> {
    hist[1..]
        .iter()
        .map(Vec::as_slice)
        .chain(frames.chunks_exact(len))
        .collect()
}

/// Laisse dans `hist` les `kt` dernières trames, comme `k` appels à `push`.
fn slide(hist: &mut Vec<Vec<f32>>, frames: &[f32], len: usize) {
    let kt = hist.len();
    let tail: Vec<Vec<f32>> = window_source(hist, frames, len)
        .iter()
        .rev()
        .take(kt)
        .rev()
        .map(|s| s.to_vec())
        .collect();
    *hist = tail;
}

/// `(0..n).map(f)` réparti en tranches contiguës sur `threads` cœurs.
fn par_map<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let chunk = n.div_ceil(threads.max(1)).max(1);
    let f = &f;
    thread::scope(|scope| {
        let handles: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|lo| scope.spawn(move || (lo..(lo + chunk).min(n)).map(f).collect::<Vec<_>>()))
            .collect();
        handles.into_iter().flat_map(join).collect()
    })
}

/// Relance dans l'appelant la panique d'un fil de calcul.
fn join<T>(handle: thread::ScopedJoinHandle<'_, T>) -> T {
    handle
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}
