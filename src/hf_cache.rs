use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

/// Renvoie le répertoire du hub HF configuré par l'environnement.
pub(crate) fn hf_cache_dir_from_env() -> Option<PathBuf> {
    hf_cache_dir(
        std::env::var_os("HF_HUB_CACHE"),
        std::env::var_os("HF_HOME"),
        std::env::var_os("HOME"),
    )
}

fn hf_cache_dir(
    hub_cache: Option<OsString>,
    hf_home: Option<OsString>,
    home: Option<OsString>,
) -> Option<PathBuf> {
    non_empty(hub_cache.as_deref())
        .map(PathBuf::from)
        .or_else(|| non_empty(hf_home.as_deref()).map(|path| PathBuf::from(path).join("hub")))
        .or_else(|| {
            non_empty(home.as_deref())
                .map(|path| PathBuf::from(path).join(".cache/huggingface/hub"))
        })
}

fn non_empty(value: Option<&OsStr>) -> Option<&OsStr> {
    value.filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dir_prefers_hub_cache_then_hf_home_then_home() {
        assert_eq!(
            hf_cache_dir(
                Some(OsString::from("/hub-cache")),
                Some(OsString::from("/hf-home")),
                Some(OsString::from("/home")),
            ),
            Some(PathBuf::from("/hub-cache"))
        );
        assert_eq!(
            hf_cache_dir(
                Some(OsString::new()),
                Some(OsString::from("/hf-home")),
                Some(OsString::from("/home")),
            ),
            Some(PathBuf::from("/hf-home/hub"))
        );
        assert_eq!(
            hf_cache_dir(None, None, Some(OsString::from("/home"))),
            Some(PathBuf::from("/home/.cache/huggingface/hub"))
        );
        assert_eq!(hf_cache_dir(None, None, None), None);
    }
}
