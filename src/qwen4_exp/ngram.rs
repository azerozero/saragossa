//! Hachage n-gram du PLE `qwen4_exp` (transformers `Qwen4ExpTextNGramEmbedding`).
//!
//! Identifiants : pour chaque n ∈ 2..=ngram_size et chaque tête de ce n,
//! `id = (⊕_{j<n} tok_{t−j}·mult_j) mod premier_tête + offset_tête`, les
//! tokens décalés étant remplacés par EOS au-delà du début de segment (le
//! segment démarre après le dernier EOS strictement antérieur).

/// Seed du hachage : défaut de `Qwen4ExpTextConfig`, absent de `config.json`.
pub const NGRAM_SEED: u64 = 1234;

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
const PRIME_1: u64 = 10_007;

fn splitmix64(value: u64) -> u64 {
    let value = value.wrapping_add(SPLITMIX_GAMMA);
    let value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_M1);
    let value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_M2);
    value ^ (value >> 31)
}

/// Multiplicateurs de couche (`_build_layer_multipliers`).
pub fn layer_multipliers(
    unigram_vocab: u64,
    ngram_size: usize,
    ple_layer_index: u64,
    seed: u64,
) -> Vec<i64> {
    let max_long = i64::MAX as u64;
    let multiplier_max = max_long / unigram_vocab.max(1);
    let half_bound = (multiplier_max / 2).max(1);
    let base_seed = seed.wrapping_add(PRIME_1.wrapping_mul(ple_layer_index));
    (0..ngram_size as u64)
        .map(|index| {
            let value = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index + 1));
            // Borné par 2·half_bound+1 ≤ i64::MAX : la conversion est exacte.
            (2 * (splitmix64(value) % half_bound) + 1) as i64
        })
        .collect()
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value % 2 == 0 {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor * divisor <= value {
        if value % divisor == 0 {
            return false;
        }
        divisor += 2;
    }
    true
}

/// Tailles de vocabulaire par tête : `(i+1)`-ième premier après `base − 1`.
pub fn head_vocab_sizes(base: u64, heads: usize) -> Vec<i64> {
    let mut sizes = Vec::with_capacity(heads);
    let mut prime = base.saturating_sub(1);
    for _ in 0..heads {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
        sizes.push(prime as i64);
    }
    sizes
}

/// `_shift_right_ignore_eos` : token à `t − shift` s'il reste dans le segment
/// de `t`, EOS sinon.
fn shifted(history: &[u64], eos: u64, shift: usize) -> Vec<u64> {
    let mut previous_eos: Option<usize> = None;
    let mut out = Vec::with_capacity(history.len());
    for (position, &token) in history.iter().enumerate() {
        // Segment : après le dernier EOS strictement antérieur à `position`.
        let segment_start = previous_eos.map_or(0, |eos_position| eos_position + 1);
        let valid = position >= shift && position - shift >= segment_start;
        out.push(if valid {
            history[position - shift]
        } else {
            eos
        });
        if token == eos {
            previous_eos = Some(position);
        }
    }
    out
}

/// Identifiants de lignes `[tokens.len(), heads]` (têtes des bigrammes puis
/// des trigrammes…), contexte initial rempli d'EOS (pas de cache).
///
/// `multipliers`, `vocab_sizes` et `offsets` sont les buffers I64 du
/// checkpoint ; les produits et le XOR sont faits en u64 (les multiplicateurs
/// sont bornés par `2^63 / vocab`, le produit ne déborde pas), le modulo par
/// un premier positif est donc positif.
pub fn ngram_ids(
    tokens: &[u64],
    eos: u64,
    ngram_size: usize,
    heads_per_ngram: usize,
    multipliers: &[i64],
    vocab_sizes: &[i64],
    offsets: &[i64],
) -> Vec<u64> {
    let context = ngram_size.saturating_sub(1);
    let mut history = vec![eos; context];
    history.extend_from_slice(tokens);
    let shifts: Vec<Vec<u64>> = (0..ngram_size)
        .map(|shift| shifted(&history, eos, shift))
        .collect();
    let heads = context * heads_per_ngram;
    let mut ids = Vec::with_capacity(tokens.len() * heads);
    for position in context..history.len() {
        for n in 2..=ngram_size {
            let mut mixed = 0_u64;
            for (j, shift) in shifts.iter().enumerate().take(n) {
                mixed ^= shift[position].wrapping_mul(multipliers[j] as u64);
            }
            let first = (n - 2) * heads_per_ngram;
            for head in first..first + heads_per_ngram {
                let size = vocab_sizes[head] as u64;
                ids.push(mixed % size + offsets[head] as u64);
            }
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_checkpoint_hash_goldens() {
        // Valeurs lues dans le checkpoint officiel (rev 236dfdf2).
        assert_eq!(
            layer_multipliers(248_320, 3, 0, NGRAM_SEED),
            vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
        );
        let sizes = head_vocab_sizes(20_000_000, 16);
        assert_eq!(sizes[..4], [20_000_003, 20_000_023, 20_000_033, 20_000_047]);
        assert_eq!(sizes[15], 20_000_171);
    }

    #[test]
    fn eos_restarts_context_but_keeps_its_own_history() {
        let eos = 5;
        // Historique effectif : [eos, eos, 7, 8, eos, 9].
        let history = [eos, eos, 7, 8, eos, 9];
        let one = shifted(&history, eos, 1);
        // L'EOS (position 4) voit encore 8 ; le 9 qui le suit repart après lui.
        assert_eq!(one, vec![eos, eos, eos, 7, 8, eos]);
        let two = shifted(&history, eos, 2);
        assert_eq!(two, vec![eos, eos, eos, eos, 7, eos]);
    }

    #[test]
    fn ids_are_offset_by_head_and_ordered_bigrams_then_trigrams() {
        let mult = [3_i64, 5, 7];
        let sizes = [11_i64, 13, 17, 19];
        let offsets = [0_i64, 11, 24, 41];
        let ids = ngram_ids(&[2, 4], 1, 3, 2, &mult, &sizes, &offsets);
        assert_eq!(ids.len(), 2 * 4);
        // Token 4 en position 1 : bigramme 4·3 ⊕ 2·5, trigramme ⊕ 1·7 (EOS initial).
        let bigram = (4 * 3) ^ (2 * 5);
        let trigram = bigram ^ 7;
        assert_eq!(
            ids[4..],
            [
                bigram % 11,
                bigram % 13 + 11,
                trigram % 17 + 24,
                trigram % 19 + 41
            ]
        );
    }
}
