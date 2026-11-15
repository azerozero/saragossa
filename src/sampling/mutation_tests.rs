use super::*;

#[test]
fn invalid_speculative_distributions_fail_before_consuming_rng() {
    let valid = [(7, 1.0)];
    let invalid = [
        vec![],
        vec![(7, 0.0)],
        vec![(7, f32::EPSILON)],
        vec![(7, -0.25), (9, 1.0)],
        vec![(7, f32::NAN)],
        vec![(7, f32::INFINITY)],
        vec![(7, f32::NEG_INFINITY)],
    ];
    for distribution in invalid {
        assert!(validate_distribution(&distribution, "probe").is_err());
        for (target, draft) in [
            (&distribution[..], &valid[..]),
            (&valid[..], &distribution[..]),
        ] {
            let mut sampler = DeterministicSampler::new(73);
            let before = sampler.state();
            assert!(lossless_speculative_sample(target, draft, 7, &mut sampler).is_err());
            assert_eq!(
                sampler.state(),
                before,
                "une entrée refusée ne doit pas tirer"
            );
        }
    }
    // Le zéro est permis dans le support ; seule la masse totale est bornée.
    assert!(validate_distribution(&[(0, -0.0), (7, 2.0 * f32::EPSILON)], "probe").is_ok());
}

#[test]
fn degenerate_sampling_rejects_before_consuming_rng() {
    for probs in [vec![], vec![0.0], vec![f32::EPSILON]] {
        let sparse: Vec<_> = probs.iter().copied().enumerate().collect();
        let mut sampler = DeterministicSampler::new(73);
        assert!(sample_from_token_distribution(&sparse, &mut sampler).is_err());
        assert_eq!(sampler.state(), 73);
        assert!(sample_from_dense_distribution(&probs, &mut sampler).is_err());
        assert_eq!(sampler.state(), 73);
    }
    for probs in [[f32::NAN, 1.0], [f32::INFINITY, 1.0]] {
        let mut sampler = DeterministicSampler::new(73);
        assert!(sample_from_dense_distribution(&probs, &mut sampler).is_err());
        assert_eq!(sampler.state(), 73);
    }
    let mut sampler = DeterministicSampler::new(73);
    assert_eq!(
        sample_from_dense_distribution(&[0.0, 2.0 * f32::EPSILON], &mut sampler)
            .expect("invariant: masse valide"),
        1
    );
    assert_eq!(
        sample_from_token_distribution(&[(7, 2.0 * f32::EPSILON)], &mut sampler)
            .expect("invariant: masse valide"),
        7
    );
}

#[test]
fn sampling_is_invariant_to_positive_mass_scaling() {
    for seed in 0..64 {
        let mut normalized = DeterministicSampler::new(seed);
        let mut scaled = DeterministicSampler::new(seed);
        let dense = sample_from_dense_distribution(&[0.0, 0.25, 0.75], &mut normalized)
            .expect("invariant: masse normalisée");
        assert_eq!(
            dense,
            sample_from_dense_distribution(&[0.0, 1.0, 3.0], &mut scaled)
                .expect("invariant: masse positive")
        );
        let mut normalized = DeterministicSampler::new(seed);
        let mut scaled = DeterministicSampler::new(seed);
        let sparse = sample_from_token_distribution(&[(7, 0.25), (11, 0.75)], &mut normalized)
            .expect("invariant: masse normalisée");
        assert_eq!(
            sparse,
            sample_from_token_distribution(&[(7, 1.0), (11, 3.0)], &mut scaled)
                .expect("invariant: masse positive")
        );
        assert_eq!(sparse, if dense == 1 { 7 } else { 11 });
    }
}
