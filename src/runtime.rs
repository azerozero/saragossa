//! Sélection du runtime forward pour le décodeur expérimental.

#[cfg(not(all(target_os = "macos", feature = "metal")))]
use std::marker::PhantomData;

/// Runtime utilisé par les opérations de couche.
#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardRuntime<'a> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    metal: Option<&'a crate::MetalExecutor>,
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    _marker: PhantomData<&'a ()>,
}

impl<'a> ForwardRuntime<'a> {
    /// Renvoie le runtime CPU pur.
    #[must_use]
    pub fn cpu() -> Self {
        Self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            metal: None,
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            _marker: PhantomData,
        }
    }

    /// Renvoie le runtime Metal.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[must_use]
    pub fn metal(executor: &'a crate::MetalExecutor) -> Self {
        Self {
            metal: Some(executor),
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn metal_executor(self) -> Option<&'a crate::MetalExecutor> {
        self.metal
    }
}
