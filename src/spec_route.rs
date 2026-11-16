//! Routage spéculatif par classe de prompt : le code vers DFlash, la prose vers MTP.
//!
//! Campagne du 15 novembre (27B u4ls, 256 tokens) : le tour DFlash coûte
//! 87-126 ms quelle que soit l'acceptance. Le code en accepte 5,1-7,8 tokens
//! par tour (DFlash +18 à +36 % sur MTP), la prose 2,7-3,9, sous le point
//! mort d'environ 4 tokens (MTP +4 à +28 % sur DFlash). La bascule par
//! acceptance observée paie le départ DFlash lent (verdict A du 19 octobre) :
//! la décision se prend donc avant le prefill, sur la conversation.
//!
//! Règle prudente : DFlash reste le choix par défaut, perdre un peu sur une
//! prose mal classée coûte moins que perdre le gain du code. MTP n'est pris que
//! si rien dans la requête ne ressemble à du code (docs/perf/spec-route.md).

/// Radicaux d'un mot de code, comparés en préfixe (`cod` couvre code, coder,
/// codage ; `compil` couvre « erreur de compilation »).
const CODE_STEMS: &[&str] = &[
    "cod",
    "fonction",
    "function",
    "script",
    "regex",
    "refactor",
    "program",
    "implement",
    "implément",
    "algorith",
    "sql",
    "compil",
    "debug",
    "débog",
];

/// Langages, formats et frameworks, comparés en mots entiers.
const CODE_WORDS: &[&str] = &[
    "python",
    "rust",
    "typescript",
    "javascript",
    "js",
    "ts",
    "java",
    "kotlin",
    "swift",
    "go",
    "golang",
    "c",
    "c#",
    "c++",
    "cpp",
    "lua",
    "perl",
    "scala",
    "zig",
    "dart",
    "r",
    "julia",
    "ocaml",
    "elixir",
    "erlang",
    "haskell",
    "ruby",
    "php",
    "bash",
    "sh",
    "zsh",
    "fish",
    "shell",
    "powershell",
    "awk",
    "sed",
    "jq",
    "html",
    "css",
    "xml",
    "yaml",
    "yml",
    "json",
    "toml",
    "dockerfile",
    "docker",
    "makefile",
    "cmake",
    "terraform",
    "latex",
    "tex",
    "react",
    "vue",
    "angular",
    "svelte",
    "django",
    "flask",
    "rails",
    "swiftui",
    "node",
    "nodejs",
    "wasm",
    "cuda",
    "api",
    "bug",
    "git",
];

/// Fragments de syntaxe qui trahissent un code collé sans clôture.
const CODE_SYNTAX: &[&str] = &[
    "`", "{", "}", "==", "!=", "=>", "->", "::", "()", "[]", "</", "/>", "&&", "||", "+=", ";\n",
    "#include", "#!/",
];

/// Mots qui, dans les deux mots précédents, annulent un marqueur (« sans code »).
const NEGATIONS: &[&str] = &["sans", "no", "without", "pas", "aucun"];

/// Formules qui excluent les mots de code (pas la syntaxe d'un code collé).
const PROSE_ONLY: &[&str] = &["uniquement de la prose", "only prose", "prose only"];

/// Vrai si la requête doit passer par DFlash.
///
/// `messages` donne `(rôle, texte)` pour toute la conversation, dans l'ordre.
/// MTP n'est choisi que si : pas d'outils, aucune clôture ```` ``` ```` dans la
/// conversation (assistant compris), un dernier message utilisateur en écriture
/// latine et sans indice de code, lu en entier.
pub fn prefers_dflash<R: AsRef<str>, S: AsRef<str>>(
    messages: impl IntoIterator<Item = (R, S)>,
    has_tools: bool,
) -> bool {
    if has_tools {
        return true;
    }
    let mut last_user = None;
    for (role, text) in messages {
        if text.as_ref().contains("```") {
            return true;
        }
        if role.as_ref() == "user" {
            last_user = Some(text);
        }
    }
    let Some(text) = last_user else {
        return true;
    };
    let text = text.as_ref();
    text.trim().is_empty() || !is_latin(text) || has_code_hint(text)
}

/// Le greedy peut-il prendre DFlash ? Toujours si le routage est coupé
/// (`SARAGOSSA_RUST_SPEC_ROUTE=0`), sans lire la conversation.
pub fn allows_dflash<R: AsRef<str>, S: AsRef<str>>(
    messages: impl IntoIterator<Item = (R, S)>,
    has_tools: bool,
) -> bool {
    !crate::runtime_flags::spec_route_enabled() || prefers_dflash(messages, has_tools)
}

/// Toutes les lettres sont latines (ASCII, Latin-1, Latin étendu A/B et
/// additionnel) ; chiffres, ponctuation et emoji ne comptent pas.
fn is_latin(text: &str) -> bool {
    text.chars().filter(|c| c.is_alphabetic()).all(|c| {
        c.is_ascii_alphabetic()
            || matches!(c, 'ª' | 'º' | '\u{00C0}'..='\u{024F}' | '\u{1E00}'..='\u{1EFF}')
    })
}

fn has_code_hint(text: &str) -> bool {
    let lower = text.to_lowercase();
    if CODE_SYNTAX.iter().any(|fragment| lower.contains(fragment))
        || lower
            .lines()
            .any(|line| line.starts_with("    ") || line.starts_with('\t'))
        || lower
            .as_bytes()
            .windows(2)
            .any(|pair| pair[1] == b'(' && pair[0].is_ascii_alphanumeric())
    {
        return true;
    }
    if PROSE_ONLY.iter().any(|phrase| lower.contains(phrase)) {
        return false;
    }
    // Un mot suivi d'une apostrophe est une élision (« c'est », « l'API ») : il
    // ne compte pas comme mot entier, ses radicaux restent testés.
    let mut words: Vec<(&str, bool)> = Vec::new();
    let mut start = None;
    for (index, c) in lower.char_indices() {
        if c.is_alphanumeric() || c == '#' || c == '+' {
            start.get_or_insert(index);
            continue;
        }
        if let Some(begin) = start.take() {
            words.push((&lower[begin..index], matches!(c, '\'' | '’')));
        }
    }
    if let Some(begin) = start {
        words.push((&lower[begin..], false));
    }
    words.iter().enumerate().any(|(index, &(word, elided))| {
        let marker = (!elided && CODE_WORDS.contains(&word))
            || CODE_STEMS.iter().any(|stem| word.starts_with(stem));
        let negated = words[index.saturating_sub(2)..index]
            .iter()
            .any(|(previous, _)| NEGATIONS.contains(previous));
        marker && !negated
    })
}

#[cfg(test)]
mod tests {
    use super::prefers_dflash;

    fn user(text: &str) -> bool {
        prefers_dflash([("user", text)], false)
    }

    /// Corpus des campagnes : la prose part en MTP, le code et l'ambigu en DFlash.
    #[test]
    fn corpus_classes() {
        let code = [
            "Write a complete Python implementation of Dijkstra's shortest path algorithm, then explain the algorithm step by step.",
            "Écris en Rust un cache LRU générique de capacité fixe, avec les méthodes get et put. Donne le code complet puis explique les invariants.",
            "Implement a TypeScript async retry function with exponential backoff, jitter, a maximum attempt count and AbortSignal support. Include usage examples.",
            "Le journal décrit les opérations.\nLe contexte précédent décrit un journal de port. Écris un programme Python complet qui valide des événements JSON contenant numero, source et charge, détecte les doublons et résume les erreurs.",
            "Voici un fichier Python.\n\n```python\ndef valider(x):\n    return x\n```\n\nRéécris ce fichier intégralement en renommant la fonction.",
            "Écris une fonction Python de tri fusion avec une explication.",
            "Explique la différence entre un processus et un thread sous Linux, avec un court exemple en Python.",
            "Voici un dossier technique.\n\n```rust\nfn main() {}\n```\n\nRésume ce dossier en cinq points, une phrase par point.",
            "",
        ];
        let prose = [
            "Explique en français comment les marées influencent la vie dans un port breton. Écris un texte suivi, accessible et concret, sans code.",
            "Write a short story about a lighthouse keeper who receives an unexpected letter during a winter storm. Use vivid prose and dialogue.",
            "Compare les avantages et les difficultés du travail à distance pour une petite équipe. Rédige une analyse nuancée en plusieurs paragraphes avec des exemples concrets.",
            "Le journal décrit les opérations quotidiennes du port.\nÀ partir du journal précédent, rédige un récit en français sur une journée dans ce port, depuis les premières livraisons jusqu’à la relève du soir. Utilise uniquement de la prose.",
            "C'est l'hiver : raconte l'arrivée d'un bateau au port, depuis le quai.",
        ];
        for text in code {
            assert!(user(text), "DFlash attendu : {text}");
        }
        for text in prose {
            assert!(!user(text), "MTP attendu : {text}");
        }
    }

    /// Cas hors corpus de la relecture : chacun doit rester en DFlash.
    #[test]
    fn ambiguous_requests_keep_dflash() {
        assert!(user("Écris une fonction de tri fusion.\n\nMerci !"));
        assert!(user("Que fait ceci ?\n\nfor i in range(10):\n    print(i)"));
        assert!(user("corrige l'erreur de compilation"));
        assert!(user("用Python写一个排序函数"));
        assert!(user("Écris une classe en C# pour gérer un panier."));
        assert!(user(
            "Résume en prose uniquement : x = f(y); z = g(x);\nreturn z;"
        ));
        assert!(prefers_dflash([("user", "Raconte une histoire.")], true));
        assert!(prefers_dflash(
            [
                ("user", "Raconte une histoire."),
                ("assistant", "Voici :```sh\necho fin\n```"),
                ("user", "Merci, une autre histoire."),
            ],
            false,
        ));
        assert!(prefers_dflash::<&str, &str>([], false));
        // Prix de la prudence : « vue » (Vue.js) envoie aussi cette prose en DFlash.
        assert!(user("Décris la vue depuis le phare."));
    }
}
