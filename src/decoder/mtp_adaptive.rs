//! Contrôleur de profondeur de draft MTP **adaptative** (GammaTune-style).
//!
//! Le decode MTP à profondeur fixe D2 (`max_draft=2`) ≈ D1 : l'acceptance du 2ᵉ
//! token de draft (pos-2) n'est PAS uniforme — sur certains pas le draft profond
//! est accepté (gain d'un token), sur d'autres rejeté (l'overhead draft+verify du
//! 2ᵉ token est payé pour rien). Le levier : choisir la profondeur PAR PAS. On
//! draft profond quand la pos-2 est *probablement* acceptée (régime facile,
//! répétitif/code), court sinon.
//!
//! Politique (la plus simple qui capte l'auto-corrélation de l'acceptance) :
//! une EMA de la « deep-acceptance » (le 2ᵉ draft a-t-il été accepté ?) pilote la
//! décision, avec un **sondage périodique** qui force un pas profond même en
//! régime réputé court, pour garder l'EMA fraîche (sans probe, une fois passé en
//! court on n'observe plus jamais la pos-2 → l'estimateur gèle).
//!
//! **Byte-identité** : ce contrôleur ne décide QUE *combien* de tokens on
//! propose. Le vérifieur trunk exact décide *lesquels* sont acceptés. La séquence
//! générée est donc indépendante de la profondeur choisie (garantie par l'oracle
//! `oracle_ids` OFF ≡ profondeur fixe ≡ adaptatif). Zéro readback CPU ajouté : le
//! seul signal consommé (acceptance) est déjà calculé par le chemin fused.

/// Décide dynamiquement la profondeur de draft MTP par pas de decode.
///
/// L'état tient dans quelques scalaires (aucune allocation, aucun accès GPU) :
/// l'appel par pas est négligeable devant un forward de tête MTP.
#[derive(Clone, Debug)]
pub(super) struct AdaptiveDepthController {
    /// Plafond de profondeur (= `max_draft` du run). La profondeur choisie est
    /// dans `1..=cap`, re-bornée par le nombre de tokens restants.
    cap: usize,
    /// Lissage EMA (`alpha` ∈ ]0,1]) : plus grand = plus réactif, plus bruité.
    alpha: f32,
    /// Seuil de deep-acceptance au-delà duquel drafter profond « paie ».
    threshold: f32,
    /// Force un pas profond tous les `probe_period` pas courts (rafraîchit l'EMA).
    probe_period: u32,
    /// EMA de la deep-acceptance récente (init 1.0 = optimiste → démarre profond).
    ema_deep: f32,
    /// Nombre de pas courts consécutifs depuis le dernier pas profond.
    steps_since_deep: u32,
    /// Total des profondeurs planifiées (pour la profondeur moyenne effective).
    total_depth: u64,
    /// Nombre de pas planifiés.
    steps: u64,
    /// Nombre de pas où un draft profond (≥ 2) a réellement été observé.
    deep_steps: u64,
}

impl AdaptiveDepthController {
    /// Construit un contrôleur pour un plafond `cap` et des hyperparamètres.
    ///
    /// `alpha` est borné dans `]0,1]`, `probe_period` dans `>= 1`, pour éviter
    /// qu'une configuration d'environnement dégénérée fige la politique.
    pub(super) fn new(cap: usize, alpha: f32, threshold: f32, probe_period: u32) -> Self {
        Self {
            cap: cap.max(1),
            alpha: alpha.clamp(f32::MIN_POSITIVE, 1.0),
            threshold,
            probe_period: probe_period.max(1),
            ema_deep: 1.0,
            steps_since_deep: 0,
            total_depth: 0,
            steps: 0,
            deep_steps: 0,
        }
    }

    /// Renvoie la profondeur de draft à utiliser à ce pas (`1..=cap`).
    ///
    /// `remaining` = tokens encore à générer : la profondeur ne dépasse jamais ce
    /// budget (un draft profond n'a de sens que s'il reste la place de l'accepter).
    /// Enregistre le choix pour le retour d'expérience [`Self::observe`] et la
    /// profondeur moyenne.
    pub(super) fn plan(&mut self, remaining: usize) -> usize {
        self.steps = self.steps.saturating_add(1);
        let ceiling = self.cap.min(remaining.max(1));
        if ceiling <= 1 {
            // Pas de marge pour un draft profond : reste court, mais compte le pas
            // comme « court » pour la cadence de sondage.
            self.steps_since_deep = self.steps_since_deep.saturating_add(1);
            self.total_depth = self.total_depth.saturating_add(1);
            return 1;
        }
        // POURQUOI le probe : en régime court on n'observe plus la pos-2 ; sans un
        // pas profond forcé de temps en temps, l'EMA gèle et on rate le retour d'un
        // régime facile (bursty). `+1` car on décide AVANT d'incrémenter le compteur.
        let force_probe = self.steps_since_deep.saturating_add(1) >= self.probe_period;
        let go_deep = self.ema_deep >= self.threshold || force_probe;
        let depth = if go_deep { ceiling } else { 1 };
        if depth >= 2 {
            self.steps_since_deep = 0;
        } else {
            self.steps_since_deep = self.steps_since_deep.saturating_add(1);
        }
        self.total_depth = self.total_depth.saturating_add(depth as u64);
        depth
    }

    /// Intègre le résultat d'un pas : profondeur réellement proposée et nombre de
    /// tokens de draft acceptés (`0..=depth_used`).
    ///
    /// Ne met à jour l'EMA que sur un pas profond (`depth_used >= 2`), seul cas où
    /// la pos-2 est observable : `deep_paid = 1` si le 2ᵉ draft a été accepté,
    /// sinon `0` (y compris pos-1 rejetée = draft profond payé pour rien).
    pub(super) fn observe(&mut self, depth_used: usize, accepted_drafts: usize) {
        if depth_used < 2 {
            return;
        }
        self.deep_steps = self.deep_steps.saturating_add(1);
        let deep_paid = if accepted_drafts >= 2 { 1.0 } else { 0.0 };
        self.ema_deep = (1.0 - self.alpha) * self.ema_deep + self.alpha * deep_paid;
    }

    /// Renvoie la synthèse du run : (pas, pas profonds, profondeur moyenne, EMA).
    pub(super) fn summary(&self) -> AdaptiveDepthSummary {
        let avg_depth = if self.steps > 0 {
            self.total_depth as f32 / self.steps as f32
        } else {
            0.0
        };
        AdaptiveDepthSummary {
            steps: self.steps,
            deep_steps: self.deep_steps,
            avg_depth,
            ema_deep: self.ema_deep,
        }
    }
}

/// Synthèse d'un run adaptatif, pour la trace de profondeur moyenne effective.
#[derive(Clone, Copy, Debug)]
pub(super) struct AdaptiveDepthSummary {
    /// Nombre total de pas de decode planifiés.
    pub(super) steps: u64,
    /// Nombre de pas où un draft profond a été proposé (pos-2 observée).
    pub(super) deep_steps: u64,
    /// Profondeur moyenne effective (somme des profondeurs / pas).
    pub(super) avg_depth: f32,
    /// Valeur finale de l'EMA de deep-acceptance.
    pub(super) ema_deep: f32,
}

#[cfg(test)]
mod tests {
    use super::AdaptiveDepthController;

    fn ctl(cap: usize) -> AdaptiveDepthController {
        // alpha franc, seuil 0.5, probe rare pour isoler la logique d'EMA.
        AdaptiveDepthController::new(cap, 0.5, 0.5, 1_000)
    }

    #[test]
    fn plan_stays_shallow_when_no_budget() {
        let mut c = ctl(2);
        assert_eq!(c.plan(1), 1, "un seul token restant → jamais profond");
    }

    #[test]
    fn plan_starts_deep_optimistic() {
        // EMA init 1.0 ≥ seuil → premier pas profond quand le budget le permet.
        let mut c = ctl(2);
        assert_eq!(c.plan(8), 2);
    }

    #[test]
    fn ema_collapses_to_shallow_after_repeated_deep_rejections() {
        let mut c = ctl(2);
        // Régime difficile : la pos-2 est systématiquement rejetée.
        for _ in 0..10 {
            let d = c.plan(8);
            // Pos-1 acceptée mais pos-2 rejetée → accepted_drafts = 1.
            c.observe(d, 1);
        }
        // L'EMA a chuté sous le seuil → la politique bascule en court.
        assert_eq!(c.plan(8), 1, "après des rejets pos-2 répétés, reste court");
    }

    #[test]
    fn ema_recovers_to_deep_in_easy_regime() {
        let mut c = ctl(2);
        // On force l'EMA bas d'abord.
        for _ in 0..10 {
            let d = c.plan(8);
            c.observe(d, 1);
        }
        assert_eq!(c.plan(8), 1);
        // Sondage manuel : on observe une pos-2 acceptée plusieurs fois de suite.
        for _ in 0..10 {
            c.observe(2, 2);
        }
        assert_eq!(c.plan(8), 2, "régime facile retrouvé → redevient profond");
    }

    #[test]
    fn probe_forces_deep_after_period() {
        // Seuil impossible → jamais profond via l'EMA ; seul le probe déclenche.
        let mut c = AdaptiveDepthController::new(2, 0.5, 2.0, 3);
        // ema_deep=1.0 mais seuil 2.0 → go_deep uniquement par probe.
        // Pas 1 : steps_since_deep=0, +1=1 <3 → court.
        assert_eq!(c.plan(8), 1);
        // Pas 2 : 1+1=2 <3 → court.
        assert_eq!(c.plan(8), 1);
        // Pas 3 : 2+1=3 >=3 → probe profond.
        assert_eq!(c.plan(8), 2);
        // Après un pas profond, le compteur repart.
        assert_eq!(c.plan(8), 1);
    }

    #[test]
    fn observe_ignores_shallow_steps() {
        let mut c = ctl(2);
        let before = c.summary().ema_deep;
        c.observe(1, 1); // pas court → pas d'observation pos-2
        assert_eq!(c.summary().ema_deep, before);
        assert_eq!(c.summary().deep_steps, 0);
    }

    #[test]
    fn summary_tracks_average_depth() {
        let mut c = ctl(2);
        c.plan(8); // profond (2)
        c.plan(8); // profond (2)
        let s = c.summary();
        assert_eq!(s.steps, 2);
        assert!((s.avg_depth - 2.0).abs() < 1e-6);
    }

    #[test]
    fn cap_one_is_always_shallow() {
        let mut c = ctl(1);
        for _ in 0..5 {
            assert_eq!(c.plan(8), 1);
        }
    }
}
