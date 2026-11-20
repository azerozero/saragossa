//! Cohérence json/bin des goldens TTS et Whisper : un fixture ne dérive pas
//! de son en-tête (dtype, forme, taille) sans que la suite ordinaire le voie.

use std::path::Path;

#[test]
fn golden_headers_match_payload_sizes() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let mut checked = 0;
    let mut orphans = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("invariant: tests/golden lisible") {
        let path = entry.expect("invariant: entrée lisible").path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext == "json" {
            let header: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).expect("invariant: json lisible"))
                    .unwrap_or_else(|e| panic!("{}: json invalide: {e}", path.display()));
            let elements: u64 = header["shape"]
                .as_array()
                .unwrap_or_else(|| panic!("{}: shape absent", path.display()))
                .iter()
                .map(|d| d.as_u64().expect("invariant: dimension entière"))
                .product();
            let dtype = header["dtype"].as_str().unwrap_or("");
            let (payload, expected) = match dtype {
                "f32" | "i32" => (path.with_extension("bin"), vec![elements * 4]),
                // Les `.txt` portent un saut de ligne final facultatif.
                "utf8" => (path.with_extension("txt"), vec![elements, elements + 1]),
                other => panic!("{}: dtype inconnu {other}", path.display()),
            };
            let size = std::fs::metadata(&payload)
                .unwrap_or_else(|e| panic!("{}: charge absente: {e}", payload.display()))
                .len();
            assert!(
                expected.contains(&size),
                "{}: {size} octets, attendu {expected:?}",
                payload.display()
            );
            checked += 1;
        } else if (ext == "bin" || ext == "txt") && !path.with_extension("json").exists() {
            orphans.push(path.display().to_string());
        }
    }
    assert!(
        orphans.is_empty(),
        "charges sans en-tête json : {orphans:?}"
    );
    assert!(checked >= 20, "fixtures golden introuvables ({checked})");
}
