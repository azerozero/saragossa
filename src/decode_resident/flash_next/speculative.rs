//! Décodage glouton spéculatif MTP de Flash-Next, en pas mono (étape 6 C).
//!
//! Cycle, jeton en attente `y` à la position `p + 1`, résiduel du tronc `h_p` :
//! drafts `D_1 = MTP(h_p, y)` (ligne d'historique `p`, vérifiée) puis
//! `D_i = MTP(h'_{i−1}, D_{i−1})` (lignes spéculatives) ; vérification du tronc
//! sur `[y, D_1, …, D_d]` en pas mono dans un command buffer (mêmes bits que
//! l'AR) ; acceptation du plus long préfixe `D_i == a_{i−1}` ; retour arrière ;
//! historique MTP tronqué à `p + 1` lignes puis complété des paires vérifiées
//! `(h_{p+1+i}, D_{i+1})`. Le nouveau jeton en attente est `a_k`.
//!
//! Convention d'arrêt, comme l'AR : un jeton d'arrêt est émis, puis la
//! génération s'arrête ; un draft d'arrêt accepté coupe le cycle juste avant
//! lui et devient le jeton en attente, émis seul au tour suivant.

use super::session::{FlashNextSession, MtpInput};
use crate::{InferError, Result};

/// Sortie d'une génération : jetons émis et résiduels hc (avant mixer) des
/// positions engagées après le prompt, dans l'ordre.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "spéculation MTP : serve à venir")
)]
pub(crate) struct Generation {
    pub(crate) tokens: Vec<usize>,
    /// Résiduel du tronc à chaque position engagée à partir du dernier jeton
    /// du prompt (oracle d'identité des états).
    pub(crate) residuals: Vec<Vec<f32>>,
    /// Drafts acceptés par cycle (vide en AR).
    pub(crate) accepted: Vec<usize>,
}

/// Remplacement de drafts (oracles) : `(cycle, rang du draft, draft) → draft`.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "spéculation MTP : serve à venir")
)]
pub(crate) type DraftOverride<'a> = &'a dyn Fn(usize, usize, usize) -> usize;

/// AR glouton de référence : `max_tokens` jetons après `prompt`.
///
/// # Errors
///
/// Renvoie une erreur sur prompt vide ou erreur de session.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "spéculation MTP : serve à venir")
)]
pub(crate) fn generate_ar(
    session: &mut FlashNextSession,
    prompt: &[usize],
    max_tokens: usize,
    stop: &[usize],
) -> Result<Generation> {
    let (last, context) = prompt
        .split_last()
        .ok_or_else(|| InferError::Config("prompt vide".to_string()))?;
    session.reset()?;
    for &token in context {
        session.replay(token)?;
    }
    let mut out = Generation {
        tokens: Vec::new(),
        residuals: Vec::new(),
        accepted: Vec::new(),
    };
    let mut current = *last;
    while out.tokens.len() < max_tokens {
        let step = session.step(current, false, true)?;
        let layers = step
            .layers
            .ok_or_else(|| InferError::Metal("résiduels AR absents".to_string()))?;
        out.residuals
            .push(layers.last().cloned().unwrap_or_default());
        let next = step.argmax as usize;
        out.tokens.push(next);
        if stop.contains(&next) {
            break;
        }
        current = next;
    }
    Ok(out)
}

/// Décodage glouton spéculatif, profondeur `depth` (drafts par cycle, ≤ 3) :
/// mêmes jetons que [`generate_ar`], au bit près.
///
/// # Errors
///
/// Renvoie une erreur sur prompt vide, profondeur hors bornes, tête MTP
/// absente ou erreur de session.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "spéculation MTP : serve à venir")
)]
pub(crate) fn generate_speculative(
    session: &mut FlashNextSession,
    prompt: &[usize],
    max_tokens: usize,
    depth: usize,
    stop: &[usize],
    overrides: Option<DraftOverride<'_>>,
) -> Result<Generation> {
    if prompt.is_empty() {
        return Err(InferError::Config("prompt vide".to_string()));
    }
    if depth + 1 > super::session::VERIFY_ROWS {
        return Err(InferError::Config(format!(
            "profondeur {depth} au-delà de {} drafts",
            super::session::VERIFY_ROWS - 1
        )));
    }
    session.reset()?;
    // Prompt : un pas par jeton (vérification d'une ligne : résiduel relu),
    // historique MTP des paires (h_{j−1}, x_j).
    let mut h_prev: Option<Vec<f32>> = None;
    let mut pending = 0;
    let mut out = Generation {
        tokens: Vec::new(),
        residuals: Vec::new(),
        accepted: Vec::new(),
    };
    for (j, &token) in prompt.iter().enumerate() {
        if let Some(h) = &h_prev {
            session.mtp_step(MtpInput::Host(h), token, false, false)?;
        }
        let row = session.verify(&[token])?;
        session.accept(0)?;
        let row = row
            .into_iter()
            .next()
            .ok_or_else(|| InferError::Metal("ligne de vérification absente".to_string()))?;
        if j + 1 == prompt.len() {
            out.residuals.push(row.residual.clone());
        }
        pending = row.argmax as usize;
        h_prev = Some(row.residual);
    }
    let mut h = h_prev.ok_or_else(|| InferError::Config("prompt vide".to_string()))?;
    let mut cycle = 0;
    while out.tokens.len() < max_tokens {
        let y = pending;
        if stop.contains(&y) || out.tokens.len() + 1 == max_tokens {
            // Dernier jeton : émis sans cycle.
            out.tokens.push(y);
            break;
        }
        let room = session.capacity().saturating_sub(session.position());
        if room == 0 {
            return Err(InferError::Config("capacité atteinte".to_string()));
        }
        // Profondeur réduite par le budget de jetons et par la capacité.
        let d = depth.min(max_tokens - out.tokens.len() - 1).min(room - 1);
        let base_rows = session
            .mtp_rows()
            .ok_or_else(|| InferError::Config("tête MTP absente".to_string()))?;
        // Drafts : D_1 sur la paire vérifiée (h_p, y), puis enchaînés.
        let mut drafts = Vec::with_capacity(d);
        for i in 0..d {
            let step = if i == 0 {
                session.mtp_step(MtpInput::Host(&h), y, false, false)?
            } else {
                let previous = session
                    .mtp_hidden_buffer()
                    .cloned()
                    .ok_or_else(|| InferError::Config("tête MTP absente".to_string()))?;
                session.mtp_step(MtpInput::Resident(&previous), drafts[i - 1], false, false)?
            };
            let draft = step.argmax as usize;
            drafts.push(overrides.map_or(draft, |f| f(cycle, i, draft)));
        }
        if d == 0 {
            // Pas de draft possible : historique (h_p, y) puis un pas mono.
            session.mtp_step(MtpInput::Host(&h), y, false, false)?;
        }
        let mut rows = vec![y];
        rows.extend_from_slice(&drafts);
        let verified = session.verify(&rows)?;
        let mut k = 0;
        while k < d && drafts[k] == verified[k].argmax as usize {
            k += 1;
        }
        // Un draft d'arrêt accepté coupe le cycle juste avant lui.
        if let Some(i) = drafts[..k].iter().position(|t| stop.contains(t)) {
            k = i;
        }
        session.accept(k)?;
        // Historique : ligne p (vérifiée) conservée, lignes enchaînées retirées,
        // paires vérifiées (h_{p+1+i}, D_{i+1}) ajoutées.
        session.mtp_truncate(base_rows + 1)?;
        for i in 0..k {
            session.mtp_step(
                MtpInput::Host(&verified[i].residual),
                drafts[i],
                false,
                false,
            )?;
        }
        out.tokens.push(y);
        out.tokens.extend_from_slice(&drafts[..k]);
        for row in &verified[..=k] {
            out.residuals.push(row.residual.clone());
        }
        out.accepted.push(k);
        pending = verified[k].argmax as usize;
        h = verified[k].residual.clone();
        cycle += 1;
    }
    out.tokens.truncate(max_tokens);
    Ok(out)
}
