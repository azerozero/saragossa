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

#[cfg(all(feature = "embed", feature = "decision"))]
pub mod embed;

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
    "`",
    "{",
    "}",
    "==",
    "!=",
    "=>",
    "->",
    "::",
    "()",
    "[]",
    "</",
    "/>",
    "&&",
    "||",
    "+=",
    ";\n",
    "#include",
    "#!/",
    "def ",
    "fn ",
    "function(",
];

/// Mots qui, dans les deux mots précédents, annulent le radical `cod` (« sans code »).
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
    let lower = text.to_lowercase();
    if text.trim().is_empty()
        || has_syntax_hint(&lower)
        || has_language_name(&lower)
        || has_word_hint(&lower)
    {
        return true;
    }
    // Classifieur opt-in : il remplace la règle « écriture latine » et peut
    // donc envoyer aussi la prose non latine vers MTP.
    #[cfg(all(feature = "embed", feature = "decision"))]
    if let Some(router) = embed::router() {
        return !router.is_prose(text);
    }
    !is_latin(text)
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
    text.chars()
        .filter(|c| c.is_alphabetic())
        .all(is_latin_letter)
}

/// Chiffre ASCII ou lettre latine ; vrai aussi pour un non-alphabétique ASCII.
fn is_latin_letter(c: char) -> bool {
    c.is_ascii() || matches!(c, 'ª' | 'º' | '\u{00C0}'..='\u{024F}' | '\u{1E00}'..='\u{1EFF}')
}

/// Syntaxe de code cherchée en sous-chaînes, quelle que soit la langue : le
/// CJK et le coréen n'ont pas d'espaces entre les mots.
fn has_syntax_hint(lower: &str) -> bool {
    CODE_SYNTAX.iter().any(|fragment| lower.contains(fragment))
        || lower
            .lines()
            .any(|line| line.starts_with("    ") || line.starts_with('\t'))
        || lower
            .as_bytes()
            .windows(2)
            .any(|pair| pair[1] == b'(' && pair[0].is_ascii_alphanumeric())
        || lower
            .find("select ")
            .is_some_and(|at| lower[at..].contains(" from "))
}

/// Nom de langage ou de format en ASCII, borné par des caractères qui ne sont
/// ni alphanumériques ASCII ni `#`/`+`, ni suivi d'une apostrophe d'élision
/// (« c'est » n'est pas du C). « 用python写 » et « Pythonで » comptent.
fn has_language_name(lower: &str) -> bool {
    // Une lettre latine (accentuée comprise) prolonge le mot : « rédige » ne
    // contient pas le langage R. Le CJK, le cyrillique, etc. le bornent.
    let joins = |c: char| (c.is_alphanumeric() && is_latin_letter(c)) || c == '#' || c == '+';
    CODE_WORDS.iter().any(|name| {
        lower.match_indices(name).any(|(at, _)| {
            let rest = &lower[at + name.len()..];
            !lower[..at].chars().next_back().is_some_and(joins)
                && !rest.chars().next().is_some_and(joins)
                && !rest.starts_with(['\'', '’'])
        })
    })
}

/// Radicaux de mots de code (écriture latine), annulables par une négation
/// proche ou une formule « uniquement de la prose ».
fn has_word_hint(lower: &str) -> bool {
    if PROSE_ONLY.iter().any(|phrase| lower.contains(phrase)) {
        return false;
    }
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect();
    words.iter().enumerate().any(|(index, word)| {
        // Seul « code » se nie (« sans code ») : « il n'y a pas de fonction
        // main » parle toujours de code.
        let negated = word.starts_with("cod")
            && words[index.saturating_sub(2)..index]
                .iter()
                .any(|previous| NEGATIONS.contains(previous));
        CODE_STEMS.iter().any(|stem| word.starts_with(stem)) && !negated
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

    /// Indices de code hors écriture latine, cherchés en sous-chaînes : le
    /// classifieur doit les voir même si la règle « écriture latine » tombe.
    #[test]
    fn multilingual_code_hints() {
        use super::{has_language_name, has_syntax_hint};
        let code = [
            "用python写一个函数",
            "Pythonで関数を書いて",
            "Python으로 함수 작성",
            "напиши функцию на Rust",
            "اكتب دالة بلغة JavaScript",
            "写一个 SQL 查询：select name from users",
            "このコードを説明して：def main():",
            "escribe una función en Go",
            "schreibe ein Skript in Bash",
        ];
        for text in code {
            let lower = text.to_lowercase();
            assert!(
                has_language_name(&lower) || has_syntax_hint(&lower),
                "indice attendu : {text}"
            );
            assert!(user(text), "DFlash attendu : {text}");
        }
        // Prose non latine : DFlash tant que la règle « écriture latine » tient.
        for text in [
            "写一个关于灯塔的短篇故事",
            "灯台守の物語を書いて",
            "등대지기에 관한 이야기를 써 줘",
            "Напиши рассказ о смотрителе маяка",
            "اكتب قصة عن حارس المنارة",
        ] {
            let lower = text.to_lowercase();
            assert!(
                !has_language_name(&lower) && !has_syntax_hint(&lower),
                "{text}"
            );
            assert!(user(text), "DFlash attendu (écriture non latine) : {text}");
        }
        // Prose latine (es, de) sans indice : MTP.
        for text in [
            "Escribe un cuento corto sobre un faro en invierno.",
            "Schreibe eine kurze Geschichte über einen Leuchtturm im Winter.",
        ] {
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
        // La négation ne vaut que pour « code » : les autres radicaux restent.
        assert!(user("Il n'y a pas de fonction main dans mon fichier."));
        assert!(user("Pourquoi pas un script de sauvegarde ?"));
        assert!(user("No refactoring today, just explain the algorithm."));
        assert!(!user("Explique les marées, sans code."));
        assert!(!user("Raconte une légende, pas de code."));
        // Prix de la prudence : « vue » (Vue.js) envoie aussi cette prose en DFlash.
        assert!(user("Décris la vue depuis le phare."));
    }
}
