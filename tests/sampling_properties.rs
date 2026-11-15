use proptest::prelude::*;
use proptest::test_runner::{Config, FileFailurePersistence};
use saragossa::sampling::{sample_from_token_distribution, token_distribution_top_k_top_p};
use saragossa::DeterministicSampler;

proptest! {
    #![proptest_config(Config {
        cases: 128,
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct(
            "proptest-regressions/sampling.txt"
        ))),
        ..Config::default()
    })]

    #[test]
    fn distribution_and_seed_preserve_their_contract(
        logits in prop::collection::vec(-20.0_f32..20.0, 1..65),
        seed in any::<u64>(),
        k in 1_usize..65,
    ) {
        let distribution = token_distribution_top_k_top_p(&logits, 0.7, 0.9, k)
            .expect("invariant: logits finis non vides");
        prop_assert!(!distribution.is_empty());
        prop_assert!(distribution.len() <= k.min(logits.len()));
        prop_assert!(distribution.iter().all(|(id, p)| *id < logits.len() && p.is_finite() && *p > 0.0));
        let sum: f32 = distribution.iter().map(|(_, p)| p).sum();
        prop_assert!((sum - 1.0).abs() <= 1e-6);
        let mut left = DeterministicSampler::new(seed);
        let mut right = DeterministicSampler::new(seed);
        for _ in 0..16 {
            let a = sample_from_token_distribution(&distribution, &mut left)
                .expect("invariant: distribution valide");
            let b = sample_from_token_distribution(&distribution, &mut right)
                .expect("invariant: distribution valide");
            prop_assert_eq!(a, b);
            prop_assert!(distribution.iter().any(|(id, _)| *id == a));
        }
    }
}
