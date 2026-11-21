use super::*;

#[test]
fn pcm_conversion_clips_and_preserves_sign_and_scale() {
    for (sample, expected) in [
        (f32::NEG_INFINITY, -32767),
        (-2.0, -32767),
        (-1.0, -32767),
        (-0.5, -16383),
        (-0.0, 0),
        (0.0, 0),
        (0.5, 16383),
        (1.0, 32767),
        (2.0, 32767),
        (f32::INFINITY, 32767),
        (f32::NAN, 0),
    ] {
        assert_eq!(pcm_i16(sample), expected, "sample={sample}");
    }
}
