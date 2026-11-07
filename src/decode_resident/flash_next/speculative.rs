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
//! lui et devient le jeton en attente, émis seul au tour suivant. Comme en AR,
//! le dernier jeton émis n'est jamais consommé par le tronc : un cycle émet
//! au plus `budget restant − 1` jetons.
//!
//! Échec en cours de cycle : l'historique MTP revient au préfixe propre des
//! `p + 1` paires vérifiées (clés brutes de l'indexeur restaurées), le tronc
//! garde ce qu'il a engagé. Toute reprise passe par un `reset` (chaque
//! génération en commence un).

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
    /// Durée de la génération après le prompt (débit publié).
    pub(crate) decode: std::time::Duration,
}

/// Remplacement de drafts (oracles) : `(cycle, jetons déjà émis, rang du
/// draft, draft) → draft`. Le draft de rang `i` prédit le jeton d'indice
/// `émis + 1 + i` de la sortie.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "spéculation MTP : serve à venir")
)]
pub(crate) type DraftOverride<'a> = &'a dyn Fn(usize, usize, usize, usize) -> usize;

/// Oracles : mutation de l'historique MTP (puissance de l'oracle des drafts).
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SpecMutation {
    /// Les lignes des drafts enchaînés ne sont pas retirées.
    NoTruncate,
    /// Les paires ajoutées portent le résiduel d'avant le cycle au lieu des
    /// résiduels vérifiés.
    StaleHistory,
    /// La troncature ne restaure pas les clés brutes de l'indexeur MTP.
    NoRawRestore,
}

/// Options d'une génération spéculative.
#[derive(Default)]
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "spéculation MTP : serve à venir")
)]
pub(crate) struct SpecOptions<'a> {
    /// Remplacement de drafts (oracles, cas limites).
    pub(crate) overrides: Option<DraftOverride<'a>>,
    #[cfg(test)]
    pub(crate) mutation: Option<SpecMutation>,
    /// Oracles : échec d'encodage de la vérification injecté au cycle donné.
    #[cfg(test)]
    pub(crate) fail_verify_cycle: Option<usize>,
    /// Oracles : échec injecté au pas de reconstruction `(cycle, paire)` de
    /// l'historique MTP, après l'acceptation.
    #[cfg(test)]
    pub(crate) fail_rebuild: Option<(usize, usize)>,
    /// Mesure (étape 6 D) : drafts restreints à un sous-vocabulaire (masque
    /// par jeton), argmax des logits MTP pris sur l'hôte.
    #[cfg(test)]
    pub(crate) draft_vocab: Option<&'a [bool]>,
}

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
        decode: std::time::Duration::ZERO,
    };
    // Le pas du dernier jeton du prompt produit le premier jeton : il compte
    // dans la génération, comme le premier cycle du spéculatif.
    let started = std::time::Instant::now();
    let mut current = *last;
    while out.tokens.len() < max_tokens {
        let step = session.step(current, false, true)?;
        let layers = step
            .layers
            .ok_or_else(|| InferError::Metal("résiduels AR absents".to_string()))?;
        out.residuals.push(
            layers
                .last()
                .cloned()
                .ok_or_else(|| InferError::Metal("résiduel AR absent".to_string()))?,
        );
        let next = step.argmax as usize;
        out.tokens.push(next);
        if stop.contains(&next) {
            break;
        }
        current = next;
    }
    out.decode = started.elapsed();
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
    options: &SpecOptions<'_>,
) -> Result<Generation> {
    if prompt.is_empty() {
        return Err(InferError::Config("prompt vide".to_string()));
    }
    if depth >= super::session::VERIFY_ROWS {
        return Err(InferError::Config(format!(
            "profondeur {depth} au-delà de {} drafts",
            super::session::VERIFY_ROWS - 1
        )));
    }
    // Profondeur nulle : AR pur, sans tête MTP. Budget nul : rien à émettre,
    // l'AR laisse le dernier jeton du prompt non consommé.
    if depth == 0 || max_tokens == 0 {
        return generate_ar(session, prompt, max_tokens, stop);
    }
    if session.mtp_rows().is_none() {
        return Err(InferError::Config("tête MTP absente".to_string()));
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
        decode: std::time::Duration::ZERO,
    };
    let mut started = std::time::Instant::now();
    for (j, &token) in prompt.iter().enumerate() {
        if let Some(h) = &h_prev {
            session.mtp_step(MtpInput::Host(h), token, false, false)?;
        }
        // Chrono au pas du tronc du dernier jeton, comme l'AR.
        if j + 1 == prompt.len() {
            started = std::time::Instant::now();
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
        // Profondeur réduite par la capacité et par le budget : le cycle émet
        // `y` et jusqu'à `d` drafts, tous consommés ; le dernier jeton émis de
        // la génération ne doit pas l'être (≥ 2 jetons restants ici).
        let d = depth
            .min((max_tokens - out.tokens.len()).saturating_sub(2))
            .min(room - 1);
        let base_rows = session
            .mtp_rows()
            .ok_or_else(|| InferError::Config("tête MTP absente".to_string()))?;
        // Toute erreur après le premier pas MTP du cycle remet l'historique à
        // ses lignes vérifiées (`clean`) avant de remonter. L'instantané des
        // clés brutes est pris juste après ce premier pas : les drafts
        // enchaînés écrivent ensuite dans l'anneau de l'indexeur MTP.
        let mut clean = base_rows;
        let fail = |session: &mut FlashNextSession, clean: usize, error: InferError| {
            // La troncature est refusée sur une session invalidée ou en
            // attente de vérification ; l'erreur d'origine prime.
            let _ = session.mtp_truncate(clean);
            error
        };
        // Drafts : D_1 sur la paire vérifiée (h_p, y), puis enchaînés. Sans
        // draft possible (capacité, budget), la paire (h_p, y) est ajoutée seule.
        let mut drafts = Vec::with_capacity(d);
        #[cfg(test)]
        let logits = options.draft_vocab.is_some();
        #[cfg(not(test))]
        let logits = false;
        for i in 0..d.max(1) {
            let step = if i == 0 {
                session.mtp_step(MtpInput::Host(&h), y, logits, false)
            } else {
                match session.mtp_hidden_buffer().cloned() {
                    Some(previous) => session.mtp_step(
                        MtpInput::Resident(&previous),
                        drafts[i - 1],
                        logits,
                        false,
                    ),
                    None => Err(InferError::Config("tête MTP absente".to_string())),
                }
            };
            let step = match step {
                Ok(step) => step,
                Err(error) => return Err(fail(session, clean, error)),
            };
            if i == 0 {
                if let Err(error) = session.mtp_snapshot_raw() {
                    return Err(fail(session, clean, error));
                }
                clean = base_rows + 1;
            }
            if i < d {
                #[cfg(test)]
                let draft = match (options.draft_vocab, &step.logits) {
                    (Some(vocab), Some(logits)) => restricted_argmax(logits, vocab)?,
                    _ => step.argmax as usize,
                };
                #[cfg(not(test))]
                let draft = step.argmax as usize;
                let emitted = out.tokens.len();
                drafts.push(
                    options
                        .overrides
                        .map_or(draft, |f| f(cycle, emitted, i, draft)),
                );
            }
        }
        let mut rows = vec![y];
        rows.extend_from_slice(&drafts);
        #[cfg(test)]
        if options.fail_verify_cycle == Some(cycle) {
            session.fail_verify_row = Some(rows.len() - 1);
        }
        let verified = session.verify(&rows);
        #[cfg(test)]
        {
            session.fail_verify_row = None;
        }
        let verified = match verified {
            Ok(verified) => verified,
            Err(error) => return Err(fail(session, clean, error)),
        };
        let mut k = 0;
        while k < d && drafts[k] == verified[k].argmax as usize {
            k += 1;
        }
        // Un draft d'arrêt accepté coupe le cycle juste avant lui.
        if let Some(i) = drafts[..k].iter().position(|t| stop.contains(t)) {
            k = i;
        }
        if let Err(error) = session.accept(k) {
            return Err(fail(session, clean, error));
        }
        // Historique : ligne p (vérifiée) conservée, lignes enchaînées retirées,
        // paires vérifiées (h_{p+1+i}, D_{i+1}) ajoutées.
        #[cfg(test)]
        let truncate = options.mutation != Some(SpecMutation::NoTruncate);
        #[cfg(not(test))]
        let truncate = true;
        #[cfg(test)]
        {
            session.skip_mtp_raw_restore = options.mutation == Some(SpecMutation::NoRawRestore);
        }
        if truncate {
            let truncated = session.mtp_truncate(clean);
            #[cfg(test)]
            {
                session.skip_mtp_raw_restore = false;
            }
            truncated?;
        }
        for i in 0..k {
            #[cfg(test)]
            let source = if options.mutation == Some(SpecMutation::StaleHistory) {
                &h
            } else {
                &verified[i].residual
            };
            #[cfg(not(test))]
            let source = &verified[i].residual;
            #[cfg(test)]
            let step = if options.fail_rebuild == Some((cycle, i)) {
                Err(InferError::Metal(
                    "reconstruction de l'historique MTP en échec (injecté)".to_string(),
                ))
            } else {
                session.mtp_step(MtpInput::Host(source), drafts[i], false, false)
            };
            #[cfg(not(test))]
            let step = session.mtp_step(MtpInput::Host(source), drafts[i], false, false);
            // Les paires déjà ajoutées sont vérifiées, mais l'historique doit
            // rester un préfixe sans trou : retour aux `p + 1` lignes propres.
            if let Err(error) = step {
                return Err(fail(session, clean, error));
            }
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
    // Le résiduel d'un jeton en attente jamais émis (fin de budget) est retiré :
    // `residuals[i]` est celui de la position qui produit `tokens[i]`.
    out.residuals.truncate(out.tokens.len());
    out.decode = started.elapsed();
    Ok(out)
}

/// Argmax des logits restreint aux jetons du masque (premier indice en cas
/// d'égalité, comme l'argmax GPU).
#[cfg(test)]
fn restricted_argmax(logits: &[f32], vocab: &[bool]) -> Result<usize> {
    logits
        .iter()
        .zip(vocab)
        .enumerate()
        .filter(|(_, (_, &keep))| keep)
        .fold(
            None,
            |best: Option<(usize, f32)>, (i, (&value, _))| match best {
                Some((_, top)) if top >= value => best,
                _ => Some((i, value)),
            },
        )
        .map(|(i, _)| i)
        .ok_or_else(|| InferError::Config("sous-vocabulaire de draft vide".to_string()))
}
