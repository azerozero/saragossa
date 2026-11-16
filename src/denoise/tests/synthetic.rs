//! Poids nuls déterministes : masque ERB à 0,5, filtre profond nul.
use super::*;
use std::collections::HashMap;

pub(super) fn model() -> DeepFilterNet3 {
    let mut weights = HashMap::new();
    let mut add = |name: String, shape: &[usize], value: f32| {
        weights.insert(
            name,
            crate::Tensor::from_vec(shape.to_vec(), vec![value; shape.iter().product()]).unwrap(),
        );
    };
    // Préfixe, canaux entrée/sortie, groupes, noyau temps/fréquence, séparable.
    for (p, ci, co, g, kt, kf, separable) in [
        ("enc.erb_conv0", 1, 64, 1, 3, 3, false),
        ("enc.erb_conv1", 64, 64, 64, 1, 3, true),
        ("enc.erb_conv2", 64, 64, 64, 1, 3, true),
        ("enc.erb_conv3", 64, 64, 64, 1, 3, true),
        ("enc.df_conv0", 2, 64, 2, 3, 3, true),
        ("enc.df_conv1", 64, 64, 64, 1, 3, true),
        ("erb_dec.conv0p", 64, 64, 64, 1, 1, false),
        ("erb_dec.conv1p", 64, 64, 64, 1, 1, false),
        ("erb_dec.conv2p", 64, 64, 64, 1, 1, false),
        ("erb_dec.conv3p", 64, 64, 64, 1, 1, false),
        ("erb_dec.convt1", 64, 64, 64, 1, 3, true),
        ("erb_dec.convt2", 64, 64, 64, 1, 3, true),
        ("erb_dec.convt3", 64, 64, 64, 1, 3, true),
        ("erb_dec.conv0_out", 64, 1, 1, 1, 3, false),
        ("df_dec.df_convp", 64, 10, 2, 5, 1, true),
    ] {
        let mut idx = usize::from(kt > 1);
        add(format!("{p}.{idx}.weight"), &[co, ci / g, kt, kf], 0.0);
        idx += 1;
        if separable {
            add(format!("{p}.{idx}.weight"), &[co, co, 1, 1], 0.0);
            idx += 1;
        }
        for (name, value) in [
            ("weight", 1.0),
            ("bias", 0.0),
            ("running_mean", 0.0),
            ("running_var", 1.0),
        ] {
            add(format!("{p}.{idx}.{name}"), &[co], value);
        }
    }
    for (p, layers, groups, output) in [
        ("enc.emb_gru", 1, 16, true),
        ("erb_dec.emb_gru", 2, 16, true),
        ("df_dec.df_gru", 2, 8, false),
    ] {
        add(
            format!("{p}.linear_in.0.weight"),
            &[groups, 512 / groups, 256 / groups],
            0.0,
        );
        if output {
            add(
                format!("{p}.linear_out.0.weight"),
                &[groups, 256 / groups, 512 / groups],
                0.0,
            );
        }
        for layer in 0..layers {
            for kind in ["ih", "hh"] {
                add(format!("{p}.gru.weight_{kind}_l{layer}"), &[768, 256], 0.0);
                add(format!("{p}.gru.bias_{kind}_l{layer}"), &[768], 0.0);
            }
        }
    }
    for (name, shape) in [
        ("enc.df_fc_emb.0.weight", vec![32, 96, 16]),
        ("df_dec.df_skip.weight", vec![16, 32, 16]),
        ("df_dec.df_out.0.weight", vec![16, 16, 60]),
        ("enc.lsnr_fc.0.weight", vec![1, 512]),
        ("enc.lsnr_fc.0.bias", vec![1]),
    ] {
        add(name.into(), &shape, 0.0);
    }
    let widths = dsp::erb_widths();
    let mut fb = vec![0.0; FREQ_BINS * NB_ERB];
    let mut inv = vec![0.0; FREQ_BINS * NB_ERB];
    let mut start = 0;
    for (b, &width) in widths.iter().enumerate() {
        for f in start..start + width {
            fb[f * NB_ERB + b] = 1.0 / width as f32;
            inv[b * FREQ_BINS + f] = 1.0;
        }
        start += width;
    }
    weights.insert(
        "erb_fb".into(),
        crate::Tensor::from_vec(vec![FREQ_BINS, NB_ERB], fb).unwrap(),
    );
    weights.insert(
        "mask.erb_inv_fb".into(),
        crate::Tensor::from_vec(vec![NB_ERB, FREQ_BINS], inv).unwrap(),
    );
    DeepFilterNet3 {
        net: Network::load(weights, FREQ_BINS).unwrap(),
        widths,
        window: dsp::vorbis_window(),
    }
}
