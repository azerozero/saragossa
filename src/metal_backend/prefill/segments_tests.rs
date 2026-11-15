//! Oracle synthétique de l'empaquetage des décisions : N séquences dans un
//! prefill contre N prefills séparés, couches gated-delta et pleine gated.

use super::tests::{assert_bit_exact, deterministic_norm, deterministic_tensor, test_executor};
use super::*;

#[test]
fn decision_segments_match_separate_prefills_bit_exact() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    const HIDDEN: usize = 64;
    const INTER: usize = 32;
    let (q_heads, kv_heads, head_dim) = (16, 2, 256);
    // Formes du chemin de production : dk128/dv128, noyau conv 4, d256 gated.
    let linear_spec = LinearAttentionStepSpec {
        num_key_heads: 1,
        num_value_heads: 2,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_dim: 4,
        rms_eps: 1.0e-6,
        l2_qk_sigmoid_gate: false,
    };
    let dims = LinearAttnResidentDims {
        in_dim: HIDDEN,
        conv_dim: 512,
        value_dim: 256,
        key_dim: 128,
    };
    let input_norm = deterministic_norm(HIDDEN, 607)?;
    let post_norm = deterministic_norm(HIDDEN, 613)?;
    let gate = Linear::new(deterministic_tensor(INTER, HIDDEN, 617, 0.1)?, None)?;
    let up = Linear::new(deterministic_tensor(INTER, HIDDEN, 619, 0.1)?, None)?;
    let down = Linear::new(deterministic_tensor(HIDDEN, INTER, 631, 0.1)?, None)?;
    let in_proj_qkv = Linear::new(deterministic_tensor(512, HIDDEN, 641, 0.08)?, None)?;
    let in_proj_z = Linear::new(deterministic_tensor(256, HIDDEN, 643, 0.08)?, None)?;
    let in_proj_b = Linear::new(deterministic_tensor(2, HIDDEN, 647, 0.08)?, None)?;
    let in_proj_a = Linear::new(deterministic_tensor(2, HIDDEN, 653, 0.08)?, None)?;
    let out_proj = Linear::new(deterministic_tensor(HIDDEN, 256, 659, 0.05)?, None)?;
    let conv_weight = Tensor::from_vec(
        vec![512, 4, 1],
        deterministic_tensor(512, 4, 661, 0.3)?.data().to_vec(),
    )?;
    let (a_log, dt_bias) = ([0.1_f32, -0.2], [0.05_f32, 0.15]);
    let norm_weight = deterministic_norm(128, 673)?;
    let q_norm = deterministic_norm(head_dim, 677)?;
    let k_norm = deterministic_norm(head_dim, 683)?;
    let q_proj = Linear::new(
        deterministic_tensor(2 * q_heads * head_dim, HIDDEN, 691, 0.06)?,
        None,
    )?;
    let k_proj = Linear::new(
        deterministic_tensor(kv_heads * head_dim, HIDDEN, 701, 0.06)?,
        None,
    )?;
    let v_proj = Linear::new(
        deterministic_tensor(kv_heads * head_dim, HIDDEN, 709, 0.06)?,
        None,
    )?;
    let o_proj = Linear::new(
        deterministic_tensor(HIDDEN, q_heads * head_dim, 719, 0.02)?,
        None,
    )?;
    let tail = PrefillMoeTail::Dense {
        gate_proj: &gate,
        up_proj: &up,
        down_proj: &down,
    };
    let spec = |seq| PrefillAttentionSpec {
        seq,
        hidden_dim: HIDDEN,
        q_heads,
        kv_heads,
        head_dim,
        rope_dims: 64,
        rope_frequency_dim: 64,
        rope_theta: 10_000_000.0,
        attn_scalar: head_dim as f32,
        window: None,
        k_eq_v: false,
        value_norm: false,
        eps: 1.0e-6,
    };
    let layers = |spec: PrefillAttentionSpec| {
        let linear = PrefillMoeLayer {
            input_norm: &input_norm,
            attention: PrefillAttentionLayer::Linear {
                weights: LinearAttnResidentWeights {
                    in_proj_qkv: &in_proj_qkv,
                    in_proj_z: &in_proj_z,
                    in_proj_b: &in_proj_b,
                    in_proj_a: &in_proj_a,
                    out_proj: &out_proj,
                    conv_weight: &conv_weight,
                    a_log: &a_log,
                    dt_bias: &dt_bias,
                    norm_weight: norm_weight.data(),
                },
                spec: linear_spec,
                dims,
            },
            attention_spec: spec,
            post_norm: &post_norm,
            post_norm_before_residual: false,
            tail,
        };
        let full = PrefillMoeLayer {
            attention: PrefillAttentionLayer::Full {
                q_proj: &q_proj,
                k_proj: &k_proj,
                v_proj: &v_proj,
                o_proj: &o_proj,
                q_norm: &q_norm,
                k_norm: &k_norm,
                gated: true,
                sinks: None,
            },
            ..linear
        };
        // Gated-delta, pleine, gated-delta : l'état d'une séquence ne doit
        // pas fuir vers la suivante, ni avant ni après l'attention.
        [linear, full, linear]
    };
    let prefill = |input: &Tensor| -> Result<Tensor> {
        let spec = spec(input.shape()[0]);
        let (hidden, _, _) = executor.qwen_moe_prefill_resident_with_captures(
            PrefillInput::Hidden(input),
            &layers(spec),
            spec,
            None,
            &[],
            usize::MAX,
            0,
            false,
        )?;
        hidden.into_tensor(HIDDEN)
    };
    // Longueurs inégales, hors multiples des tuiles de 64.
    let lengths = [97, 130, 64];
    let inputs = lengths
        .iter()
        .enumerate()
        .map(|(i, &len)| deterministic_tensor(len, HIDDEN, 601 + i, 0.5))
        .collect::<Result<Vec<_>>>()?;
    crate::runtime_flags::with_decision_noncausal(|| -> Result<()> {
        let separate = inputs.iter().map(prefill).collect::<Result<Vec<_>>>()?;
        let packed_input = Tensor::from_vec(
            vec![lengths.iter().sum(), HIDDEN],
            inputs
                .iter()
                .flat_map(|t| t.data().iter().copied())
                .collect(),
        )?;
        let packed = with_decision_segments(&lengths, || prefill(&packed_input))?;
        let mut first = 0;
        for (i, (expected, &len)) in separate.iter().zip(&lengths).enumerate() {
            let rows = packed.data()[first * HIDDEN..(first + len) * HIDDEN].to_vec();
            assert_bit_exact(
                &Tensor::from_vec(vec![len, HIDDEN], rows)?,
                expected,
                &format!("séquence {i}"),
            );
            first += len;
        }
        // Le découpage doit couvrir le prefill entier.
        assert!(with_decision_segments(&[97, 130], || prefill(&packed_input)).is_err());
        Ok(())
    })
}
