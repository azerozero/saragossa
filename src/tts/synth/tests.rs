use super::*;
use crate::tts_codec::TtsCodecStreamState;

// Un filtre causal à mémoire rend visibles les pertes/duplications aux frontières.
fn fake_batch(codes: &[Vec<i32>]) -> Vec<f32> {
    let mut previous = 0.0;
    codes
        .iter()
        .flat_map(|frame| {
            let sample = frame[0] as f32 / 64.0;
            let pair = [previous + sample, sample];
            previous = sample;
            pair
        })
        .collect()
}

#[test]
fn prefix_fallbacks_keep_doubling() {
    assert!(!uses_fixed_lots(&TtsStreamDecodeState::FullPrefixDelta {
        emitted: 0,
        exact: false
    }));
    assert!(!uses_fixed_lots(&TtsStreamDecodeState::Incremental(
        TtsCodecStreamState::default(),
    )));
}

#[test]
fn fixed_and_doubled_lots_match_batch_with_fake_codec() -> Result<()> {
    for (fixed, lot, expected) in [
        (true, 4, vec![4, 8, 12, 16, 17]),
        (false, 4, vec![4, 8, 16, 17]),
        (true, 7, vec![7, 14, 17]),
        (false, 7, vec![7, 14, 17]),
    ] {
        let codes: Vec<_> = (1..=17).map(|n| vec![n]).collect();
        let mut decoded = 0;
        let mut previous = 0.0;
        let mut ends = Vec::new();
        let mut chunks = Vec::new();
        let output = synthesize_streaming_from_frames(
            |on_frame| {
                for end in 1..=codes.len() {
                    on_frame(&codes[..end])?;
                }
                Ok(GeneratedCodes {
                    codes: codes.clone(),
                    stop_reason: TtsStopReason::Cap,
                })
            },
            |pcm| {
                chunks.extend_from_slice(pcm);
                Ok(())
            },
            |prefix| {
                if prefix.len() == decoded {
                    return Ok(Vec::new());
                }
                ends.push(prefix.len());
                let pcm = if fixed {
                    prefix[decoded..]
                        .iter()
                        .flat_map(|frame| {
                            let sample = frame[0] as f32 / 64.0;
                            let pair = [previous + sample, sample];
                            previous = sample;
                            pair
                        })
                        .collect()
                } else {
                    fake_batch(prefix)[decoded * 2..].to_vec()
                };
                decoded = prefix.len();
                Ok(pcm)
            },
            lot,
            fixed,
            24_000,
        )?;
        assert_eq!(ends, expected);
        assert_eq!(chunks, fake_batch(&codes));
        assert_eq!(output.samples, chunks);
        assert_eq!(output.codes, codes);
        assert_eq!(output.sample_rate, 24_000);
        assert!(output.frame_limit_reached);
    }
    Ok(())
}

#[test]
fn final_flush_covers_empty_short_and_exact_lots() -> Result<()> {
    for fixed in [false, true] {
        for count in [0, 1, 3, 4, 8] {
            let codes: Vec<_> = (1..=count).map(|n| vec![n]).collect();
            let mut emitted = 0;
            let mut chunks = Vec::new();
            let output = synthesize_streaming_from_frames(
                |on_frame| {
                    for end in 1..=codes.len() {
                        on_frame(&codes[..end])?;
                    }
                    Ok(GeneratedCodes {
                        codes: codes.clone(),
                        stop_reason: TtsStopReason::Eos,
                    })
                },
                |pcm| {
                    assert!(!pcm.is_empty());
                    chunks.extend_from_slice(pcm);
                    Ok(())
                },
                |prefix| {
                    let all = fake_batch(prefix);
                    let delta = all[emitted..].to_vec();
                    emitted = all.len();
                    Ok(delta)
                },
                4,
                fixed,
                24_000,
            )?;
            assert_eq!(chunks, fake_batch(&codes));
            assert_eq!(output.samples, chunks);
            assert!(!output.frame_limit_reached);
        }
    }
    Ok(())
}

#[test]
fn callback_failure_stops_generation_without_final_decode() {
    let mut generated = 0;
    let mut decodes = 0;
    let result = synthesize_streaming_from_frames(
        |on_frame| {
            let mut codes = Vec::new();
            for n in 1..=17 {
                codes.push(vec![n]);
                generated += 1;
                on_frame(&codes)?;
            }
            Ok(GeneratedCodes {
                codes,
                stop_reason: TtsStopReason::Cap,
            })
        },
        |_| Err(InferError::Config("client déconnecté".into())),
        |prefix| {
            decodes += 1;
            Ok(fake_batch(prefix))
        },
        4,
        true,
        24_000,
    );
    assert!(result.is_err());
    assert_eq!(generated, 4);
    assert_eq!(decodes, 1);
}

#[test]
fn stop_reason_survives_streaming_even_at_the_frame_cap() -> Result<()> {
    for reason in [
        TtsStopReason::Eos,
        TtsStopReason::Repeat,
        TtsStopReason::Cap,
    ] {
        let output = synthesize_streaming_from_frames(
            |on_frame| {
                let codes = vec![vec![1]];
                on_frame(&codes)?;
                Ok(GeneratedCodes {
                    codes,
                    stop_reason: reason,
                })
            },
            |_| Ok(()),
            |_| Ok(Vec::new()),
            4,
            true,
            24_000,
        )?;
        assert_eq!(output.stop_reason, reason);
        assert_eq!(output.frame_limit_reached, reason == TtsStopReason::Cap);
    }
    Ok(())
}

#[test]
fn overlapped_lots_match_batch_with_fake_codec() -> Result<()> {
    for (fixed, lot) in [(true, 4), (false, 4), (true, 7), (false, 1)] {
        let codes: Vec<_> = (1..=17).map(|n| vec![n]).collect();
        let mut decoded = 0;
        let mut previous = 0.0;
        let mut ends = Vec::new();
        let mut chunks = Vec::new();
        let output = synthesize_streaming_overlapped(
            |on_frame| {
                for end in 1..=codes.len() {
                    on_frame(&codes[..end])?;
                }
                Ok(GeneratedCodes {
                    codes: codes.clone(),
                    stop_reason: TtsStopReason::Cap,
                })
            },
            |pcm| {
                chunks.extend_from_slice(pcm);
                Ok(())
            },
            |prefix: &[Vec<i32>]| {
                ends.push(prefix.len());
                let pcm = if fixed {
                    prefix[decoded..]
                        .iter()
                        .flat_map(|frame| {
                            let sample = frame[0] as f32 / 64.0;
                            let pair = [previous + sample, sample];
                            previous = sample;
                            pair
                        })
                        .collect()
                } else {
                    fake_batch(prefix)[decoded * 2..].to_vec()
                };
                decoded = prefix.len();
                Ok(pcm)
            },
            lot,
            fixed,
            24_000,
        )?;
        // Les lots fusionnés dépendent du minutage : seuls l'ordre et la
        // couverture complète sont garantis, et le PCM reste celui du batch.
        assert!(ends.windows(2).all(|pair| pair[0] < pair[1]), "{ends:?}");
        assert_eq!(ends.last(), Some(&17));
        assert_eq!(chunks, fake_batch(&codes));
        assert_eq!(output.samples, chunks);
        assert_eq!(output.codes, codes);
        assert!(output.frame_limit_reached);
    }
    Ok(())
}

#[test]
fn overlapped_failures_surface_on_the_calling_thread() {
    let codes: Vec<_> = (1..=17).map(|n| vec![n]).collect();
    let run = |audio_fails: bool, codec_fails: bool| {
        synthesize_streaming_overlapped(
            |on_frame| {
                for end in 1..=codes.len() {
                    on_frame(&codes[..end])?;
                }
                Ok(GeneratedCodes {
                    codes: codes.clone(),
                    stop_reason: TtsStopReason::Eos,
                })
            },
            |_| {
                if audio_fails {
                    Err(InferError::Config("client déconnecté".into()))
                } else {
                    Ok(())
                }
            },
            |prefix: &[Vec<i32>]| {
                if codec_fails {
                    Err(InferError::Metal("codec en échec".into()))
                } else {
                    Ok(fake_batch(prefix))
                }
            },
            4,
            true,
            24_000,
        )
    };
    assert!(run(true, false).is_err());
    assert!(run(false, true).is_err());
    assert!(run(false, false).is_ok());
}
