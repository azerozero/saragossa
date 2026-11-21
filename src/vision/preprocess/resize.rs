//! Bicubique séparable uint8 : noyau de Keys a=-0,5 et support antialiasé,
//! avec les poids int16 de torchvision (ATen `UpSampleKernelAVXAntialias`).

use super::{allocate, Result, RgbImage};

/// Borne int16 des poids : la précision est la plus grande qui y tient.
const WEIGHT_LIMIT: f64 = 32768.0;

#[derive(Clone, Default)]
struct Contributions {
    start: usize,
    weights: Vec<i32>,
}

struct Axis {
    contributions: Vec<Contributions>,
    precision: u32,
}

fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5
    } else {
        0.0
    }
}

fn coefficients(input: usize, output: usize) -> Result<Axis> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let mut contributions = allocate::<Contributions>(output)?;
    let mut normalized = allocate::<Vec<f64>>(output)?;
    let mut maximum = 0.0_f64;
    for (position, contribution) in contributions.iter_mut().enumerate() {
        let center = (position as f64 + 0.5) * scale;
        let start = ((center - support + 0.5) as isize).max(0) as usize;
        let end = ((center + support + 0.5) as usize).min(input);
        let mut weights = allocate::<f64>(end - start)?;
        for (offset, weight) in weights.iter_mut().enumerate() {
            *weight = cubic((offset as f64 + start as f64 - center + 0.5) / filter_scale);
        }
        let sum: f64 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= sum;
            maximum = maximum.max(weight.abs());
        }
        contribution.start = start;
        normalized[position] = weights;
    }
    // torchvision quantifie sur int16 avec une précision commune à l'axe,
    // arrondit chaque passe à l'entier et sature en uint8 ; PIL fixe 22 bits,
    // d'où jusqu'à 2 niveaux d'écart entre les deux références.
    let mut precision = 0_u32;
    while maximum * f64::from(1_u32 << (precision + 1)) < WEIGHT_LIMIT {
        precision += 1;
    }
    for (contribution, weights) in contributions.iter_mut().zip(normalized) {
        contribution.weights = allocate::<i32>(weights.len())?;
        for (fixed, weight) in contribution.weights.iter_mut().zip(weights) {
            *fixed = (weight * f64::from(1_u32 << precision)).round() as i32;
        }
    }
    Ok(Axis {
        contributions,
        precision,
    })
}

fn sample(
    data: &[u8],
    base: usize,
    stride: usize,
    contribution: &Contributions,
    precision: u32,
) -> u8 {
    let mut sum = 1_i64 << (precision - 1);
    for (offset, &weight) in contribution.weights.iter().enumerate() {
        sum += i64::from(data[base + (contribution.start + offset) * stride]) * i64::from(weight);
    }
    (sum >> precision).clamp(0, 255) as u8
}

pub(super) fn bicubic(image: &RgbImage, height: u32, width: u32) -> Result<RgbImage> {
    let sw = image.width as usize;
    let sh = image.height as usize;
    let dw = width as usize;
    let dh = height as usize;
    let mut horizontal;
    let intermediate = if sw == dw {
        &image.pixels
    } else {
        horizontal = allocate::<u8>(super::rgb_len(image.height, width)?)?;
        let axis = coefficients(sw, dw)?;
        for y in 0..sh {
            for (x, contribution) in axis.contributions.iter().enumerate() {
                for c in 0..3 {
                    horizontal[(y * dw + x) * 3 + c] = sample(
                        &image.pixels,
                        y * sw * 3 + c,
                        3,
                        contribution,
                        axis.precision,
                    );
                }
            }
        }
        &horizontal
    };
    let mut pixels = allocate::<u8>(super::rgb_len(height, width)?)?;
    if sh == dh {
        pixels.copy_from_slice(intermediate);
    } else {
        let axis = coefficients(sh, dh)?;
        for (y, contribution) in axis.contributions.iter().enumerate() {
            for x in 0..dw {
                for c in 0..3 {
                    pixels[(y * dw + x) * 3 + c] = sample(
                        intermediate,
                        x * 3 + c,
                        dw * 3,
                        contribution,
                        axis.precision,
                    );
                }
            }
        }
    }
    RgbImage::from_rgb8(height, width, pixels)
}
