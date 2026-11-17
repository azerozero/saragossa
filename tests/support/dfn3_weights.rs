//! Poids synthétiques communs à l'oracle moteur et aux tests de transport.
use std::collections::HashMap;
const FREQ_BINS: usize = 481;
const NB_ERB: usize = 32;
pub fn weights(recurrent: bool, widths: &[usize]) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let mut weights = HashMap::new();
    let mut add = |name: String, shape: &[usize], value: f32| {
        let size = shape.iter().product();
        let values = if recurrent && shape.len() >= 2 {
            (0..size)
                .map(|i| ((i * 17 % 31) as f32 - 15.0) * 0.001)
                .collect()
        } else if recurrent && name.contains(".gru.bias_") {
            (0..size).map(|i| (i % 13) as f32 * 0.003).collect()
        } else {
            vec![value; size]
        };
        weights.insert(name, (shape.to_vec(), values));
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
    weights.insert("erb_fb".into(), (vec![FREQ_BINS, NB_ERB], fb));
    weights.insert("mask.erb_inv_fb".into(), (vec![NB_ERB, FREQ_BINS], inv));
    weights
}
