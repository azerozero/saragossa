use super::*;

#[test]
fn repeat_frame_stop_env_defaults_and_allows_disable() {
    assert_eq!(
        tts_repeat_frame_stop_from_env(None),
        DEFAULT_REPEAT_FRAME_STOP
    );
    assert_eq!(
        tts_repeat_frame_stop_from_env(Some("bad")),
        DEFAULT_REPEAT_FRAME_STOP
    );
    assert_eq!(tts_repeat_frame_stop_from_env(Some(" 12 ")), 12);
    assert_eq!(tts_repeat_frame_stop_from_env(Some("0")), 0);
}

#[test]
fn repeat_frame_stop_trips_only_at_threshold() {
    assert!(!repeat_frame_stop_tripped(32, 0));
    assert!(!repeat_frame_stop_tripped(7, 8));
    assert!(repeat_frame_stop_tripped(8, 8));
    assert!(repeat_frame_stop_tripped(9, 8));
}

#[test]
fn clone_frame_cap_matches_legacy_icl_bound() {
    assert_eq!(clone_effective_frame_cap(160, 4), 75);
    assert_eq!(clone_effective_frame_cap(160, 40), 160);
    assert_eq!(
        clone_effective_frame_cap(1_000, 200),
        CLONE_GENERATION_HARD_CAP
    );
}

#[test]
fn sampled_talker_penalizes_repeated_cb0() -> Result<()> {
    let params = TtsSampleParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 2.0,
        seed: 0,
    };
    let mut sampler = DeterministicSampler::new(0);
    let token = sample_talker_token(&[0.0, 10.0, 9.0], &params, &[], &[1], None, &mut sampler)?;

    assert_eq!(token, 2);
    Ok(())
}
