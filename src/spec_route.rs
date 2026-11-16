//! Routage spéculatif par classe de prompt : le code vers DFlash, la prose vers MTP.
//!
//! Campagne du 15 novembre (27B u4ls, 256 tokens) : le tour DFlash coûte
//! 87-126 ms quelle que soit l'acceptance. Le code en accepte 5,1-7,8 tokens
//! par tour (DFlash +18 à +36 % sur MTP), la prose 2,7-3,9, sous le point
//! mort d'environ 4 tokens (MTP +4 à +28 % sur DFlash). La bascule par
//! acceptance observée paie le départ DFlash lent (verdict A du 19 octobre) :
//! la décision se prend donc avant le prefill, sur la consigne.

/// Mots qui annoncent une réponse en code. Comparés à des mots entiers en
/// minuscules ; `implement`/`implément` et `program` valent aussi en préfixe.
const CODE_WORDS: &[&str] = &[
    "code",
    "python",
    "rust",
    "typescript",
    "javascript",
    "java",
    "kotlin",
    "swift",
    "golang",
    "sql",
    "bash",
    "shell",
    "html",
    "css",
    "php",
    "ruby",
    "haskell",
    "fonction",
    "function",
    "script",
    "regex",
    "refactor",
    "refactorise",
];

/// Mots qui, juste avant un marqueur, l'annulent (« sans code », « no code »).
const NEGATIONS: &[&str] = &["sans", "no", "without", "pas", "aucun"];

/// Formules qui excluent le code quel que soit le reste de la consigne.
const PROSE_ONLY: &[&str] = &["uniquement de la prose", "only prose", "prose only"];

/// Longueur maximale (en caractères) de la consigne examinée, prise en fin de
/// texte : sur un contexte long, seule la demande finale décrit la réponse.
const TAIL_CHARS: usize = 400;

/// Vrai si la consigne demande du code : DFlash y gagne, MTP partout ailleurs.
///
/// Seul le dernier paragraphe compte, borné à ses 400 derniers caractères :
/// un dossier technique à résumer contient du code sans en demander.
// ponytail: liste de mots fixe ; un cas mixte (prose avec court exemple) part
// en DFlash. Passer à un signal sur la sortie si la mesure le justifie.
pub fn prompt_prefers_dflash(text: &str) -> bool {
    let paragraph = text
        .trim_end()
        .rsplit("\n\n")
        .find(|part| !part.trim().is_empty())
        .unwrap_or("");
    let start = paragraph
        .char_indices()
        .rev()
        .nth(TAIL_CHARS - 1)
        .map_or(0, |(index, _)| index);
    let tail = paragraph[start..].to_lowercase();
    if PROSE_ONLY.iter().any(|phrase| tail.contains(phrase)) {
        return false;
    }
    if tail.contains("```") || tail.contains("c++") {
        return true;
    }
    let words: Vec<&str> = tail
        .split(|c: char| !(c.is_alphanumeric() || c == '#'))
        .filter(|word| !word.is_empty())
        .collect();
    words.iter().enumerate().any(|(index, word)| {
        let marker = CODE_WORDS.contains(word)
            || word.starts_with("implement")
            || word.starts_with("implément")
            || word.starts_with("program");
        let negated = words[index.saturating_sub(2)..index]
            .iter()
            .any(|previous| NEGATIONS.contains(previous));
        marker && !negated
    })
}

/// Le greedy peut-il prendre DFlash pour `text` ? Toujours sans l'opt-in
/// `SARAGOSSA_RUST_SPEC_ROUTE=1` ; avec, seulement si la consigne demande du code.
pub fn allows_dflash(text: &str) -> bool {
    !crate::runtime_flags::spec_route_enabled() || prompt_prefers_dflash(text)
}

#[cfg(test)]
mod tests {
    use super::prompt_prefers_dflash;

    /// Corpus des campagnes : classe de la sortie observée, pas du prompt.
    #[test]
    fn corpus_classes() {
        let code = [
            "Write a complete Python implementation of Dijkstra's shortest path algorithm, then explain the algorithm step by step.",
            "Écris en Rust un cache LRU générique de capacité fixe, avec les méthodes get et put. Donne le code complet puis explique les invariants.",
            "Implement a TypeScript async retry function with exponential backoff, jitter, a maximum attempt count and AbortSignal support. Include usage examples.",
            "Le journal décrit les opérations.\nLe contexte précédent décrit un journal de port. Écris un programme Python complet qui valide des événements JSON contenant numero, source et charge, détecte les doublons et résume les erreurs.",
            "Voici un fichier Python.\n\n```python\ndef valider(x):\n    return x\n```\n\nRéécris ce fichier intégralement, sans rien omettre, en renommant la fonction `valider` en `verifier_journal` partout où elle apparaît. Ne change rien d'autre.",
            "Écris une fonction Python de tri fusion avec une explication.",
        ];
        let prose = [
            "Explique en français comment les marées influencent la vie dans un port breton. Écris un texte suivi, accessible et concret, sans code.",
            "Write a short story about a lighthouse keeper who receives an unexpected letter during a winter storm. Use vivid prose and dialogue.",
            "Compare les avantages et les difficultés du travail à distance pour une petite équipe. Rédige une analyse nuancée en plusieurs paragraphes avec des exemples concrets.",
            "À partir du journal précédent, rédige un récit en français sur une journée dans ce port, depuis les premières livraisons jusqu’à la relève du soir. Utilise uniquement de la prose.",
            "Voici un dossier technique.\n\n```rust\nfn main() {}\n```\n\nRésume ce dossier en cinq points, une phrase par point.",
            "",
        ];
        for text in code {
            assert!(prompt_prefers_dflash(text), "code attendu : {text}");
        }
        for text in prose {
            assert!(!prompt_prefers_dflash(text), "prose attendue : {text}");
        }
    }
}
