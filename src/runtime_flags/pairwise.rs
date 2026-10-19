//! Overrides privés du harnais pairwise ; aucun défaut de production modifié.
use std::cell::Cell;
use std::marker::PhantomData;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Mode {
    #[default]
    D1,
    Fresh,
    D2,
    Adaptive,
    #[serde(rename = "adaptive_fresh")]
    AdaptiveFresh,
    Block2,
    Ar,
    /// Drafter DFlash 2 résident (phase 7) : boucle `generate_greedy_dflash_resident`,
    /// aucune variante MTP applicable.
    Dflash,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
pub(crate) struct Case {
    pub mode: Mode,
    pub hidden: bool,
    pub embedding: bool,
    pub overlap: bool,
    pub p1: bool,
    pub p2: bool,
    pub p3: bool,
    /// Drafts par lookup n-grammes (0 = tête MTP seule), cf.
    /// `SARAGOSSA_RUST_MTP_LOOKUP_DRAFT`.
    #[serde(default)]
    pub lookup: usize,
    /// Plafond de profondeur de draft (0 = celui du mode : 2 en adaptatif, 1 sinon).
    #[serde(default)]
    pub cap: usize,
}

impl Default for Case {
    fn default() -> Self {
        Self {
            mode: Mode::D1,
            hidden: false,
            embedding: false,
            overlap: false,
            p1: true,
            p2: false,
            p3: false,
            lookup: 0,
            cap: 0,
        }
    }
}

impl Case {
    pub fn depth(self) -> usize {
        if self.cap > 0 {
            self.cap
        } else if matches!(self.mode, Mode::D2 | Mode::Adaptive | Mode::AdaptiveFresh) {
            2
        } else {
            1
        }
    }
    pub fn fresh(self) -> bool {
        matches!(self.mode, Mode::Fresh | Mode::AdaptiveFresh | Mode::Block2)
    }
    pub fn validate(self) -> crate::Result<()> {
        if matches!(self.mode, Mode::Ar | Mode::Dflash)
            && (self.hidden || self.embedding || self.overlap)
        {
            return Err(crate::InferError::Config(
                "les variantes MTP ne sont pas applicables au contrôle AR ni au drafter DFlash"
                    .into(),
            ));
        }
        if self.mode == Mode::Block2 && (self.hidden || !self.embedding || self.overlap) {
            return Err(crate::InferError::Config(
                "bloc2 archivé : embedding GPU requis, hidden/overlap ordinaires non câblés".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Default)]
struct Active {
    case: Case,
    trace: bool,
}

#[derive(Clone, Copy)]
pub(crate) enum Hit {
    Hidden = 0,
    Embedding = 1,
    Overlap = 2,
    Block = 3,
    P1 = 4,
    P2 = 5,
    P3 = 6,
}

thread_local! {
    static ACTIVE: Cell<Option<Active>> = const { Cell::new(None) };
    static HITS: Cell<[u64; 7]> = const { Cell::new([0; 7]) };
    static IO: Cell<[u64; 4]> = const { Cell::new([0; 4]) };
}

pub(crate) fn current() -> Option<Case> {
    ACTIVE.get().map(|a| a.case)
}
pub(crate) fn traced() -> Option<bool> {
    ACTIVE.get().map(|a| a.trace)
}
pub(crate) fn hits() -> [u64; 7] {
    HITS.get()
}
pub(crate) fn io() -> [u64; 4] {
    IO.get()
}
pub(crate) fn record_io(upload: bool, bytes: usize) {
    if traced() == Some(true) {
        let mut io = IO.get();
        let index = if upload { 2 } else { 0 };
        io[index] += 1;
        io[index + 1] = io[index + 1].saturating_add(bytes as u64);
        IO.set(io);
    }
}
pub(crate) fn mark(hit: Hit) {
    if traced() == Some(true) {
        let mut h = HITS.get();
        h[hit as usize] += 1;
        HITS.set(h);
    }
}

/// Ne peut pas être déplacé vers un autre thread : le Drop restaure son TLS.
pub(crate) struct Guard {
    previous: Option<Active>,
    hits: [u64; 7],
    io: [u64; 4],
    _same_thread: PhantomData<Rc<()>>,
}
impl Guard {
    pub(crate) fn set(case: Case, trace: bool) -> crate::Result<Self> {
        case.validate()?;
        Ok(Self {
            previous: ACTIVE.replace(Some(Active { case, trace })),
            hits: HITS.replace([0; 7]),
            io: IO.replace([0; 4]),
            _same_thread: PhantomData,
        })
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        ACTIVE.set(self.previous);
        HITS.set(self.hits);
        IO.set(self.io);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scoped_override_restores_state_and_trace() -> crate::Result<()> {
        let before = current();
        {
            let _guard = Guard::set(Case::default(), true)?;
            mark(Hit::Hidden);
            record_io(false, 12);
            assert_eq!(io(), [1, 12, 0, 0]);
            assert_eq!(hits(), [1, 0, 0, 0, 0, 0, 0]);
            {
                let _nested = Guard::set(
                    Case {
                        mode: Mode::Fresh,
                        ..Case::default()
                    },
                    false,
                )?;
                mark(Hit::Embedding);
                assert_eq!(hits(), [0; 7]);
                assert_eq!(io(), [0; 4]);
                assert!(current().is_some_and(Case::fresh));
            }
            assert_eq!(current(), Some(Case::default()));
            assert_eq!(hits(), [1, 0, 0, 0, 0, 0, 0]);
            assert_eq!(io(), [1, 12, 0, 0]);
        }
        assert_eq!(current(), before);
        let adaptive_fresh = Case {
            mode: Mode::AdaptiveFresh,
            ..Case::default()
        };
        assert_eq!(adaptive_fresh.depth(), 2);
        assert!(adaptive_fresh.fresh());
        {
            let _guard = Guard::set(adaptive_fresh, false)?;
            assert!(crate::runtime_flags::mtp_adaptive_depth_enabled());
            assert!(crate::runtime_flags::mtp_fresh_cache_enabled());
        }
        assert!(Case {
            mode: Mode::Block2,
            ..Case::default()
        }
        .validate()
        .is_err());
        let dflash = Case {
            mode: Mode::Dflash,
            ..Case::default()
        };
        assert!(dflash.validate().is_ok() && dflash.depth() == 1 && !dflash.fresh());
        assert!(Case {
            overlap: true,
            ..dflash
        }
        .validate()
        .is_err());
        Ok(())
    }
}
