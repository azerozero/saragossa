//! Sous-commandes `saragossa enhance` (DeepFilterNet3) et `saragossa separate` (MossFormer2).
use std::path::PathBuf;
use std::time::Instant;

use crate::{cli_error, CliResult};
use saragossa::denoise::resample;
use saragossa::denoise::{decode_wav, DeepFilterNet3, SAMPLE_RATE};

struct IoArgs {
    model: PathBuf,
    input: PathBuf,
    output: PathBuf,
}

fn parse(args: impl Iterator<Item = String>, usage: &str) -> CliResult<Option<IoArgs>> {
    let (mut model, mut input, mut output) = (None, None, None);
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .map(PathBuf::from)
                .ok_or_else(|| cli_error(format!("{arg} exige un chemin")))
        };
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{usage}");
                return Ok(None);
            }
            "--model" => model = Some(value()?),
            "--input" | "-i" => input = Some(value()?),
            "--output" | "-o" => output = Some(value()?),
            _ => return Err(cli_error(format!("argument inconnu : {arg}\n{usage}"))),
        }
    }
    let need = |v: Option<PathBuf>, name: &str| {
        v.ok_or_else(|| cli_error(format!("{name} requis\n{usage}")))
    };
    Ok(Some(IoArgs {
        model: need(model, "--model")?,
        input: need(input, "--input")?,
        output: need(output, "--output")?,
    }))
}

fn read_wav(path: &PathBuf, rate: u32) -> CliResult<(Vec<f32>, u32)> {
    let (samples, sr) = decode_wav(&std::fs::read(path)?)?;
    Ok((resample(&samples, sr, rate)?, sr))
}

fn write_wav(path: &PathBuf, samples: &[f32], rate: u32) -> CliResult<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| cli_error(format!("écriture {}: {e}", path.display())))?;
    for &s in samples {
        writer
            .write_sample(s)
            .map_err(|e| cli_error(format!("écriture {}: {e}", path.display())))?;
    }
    writer
        .finalize()
        .map_err(|e| cli_error(format!("écriture {}: {e}", path.display())))?;
    Ok(())
}

pub(crate) fn run_enhance(args: impl Iterator<Item = String>) -> CliResult<()> {
    const USAGE: &str = "saragossa enhance --model dfn3.safetensors --input bruit.wav --output propre.wav\nDébruite avec DeepFilterNet3 sur CPU (48 kHz ; l'entrée est rééchantillonnée, la sortie reste à 48 kHz).";
    let Some(io) = parse(args, USAGE)? else {
        return Ok(());
    };
    let (audio, sr) = read_wav(&io.input, SAMPLE_RATE)?;
    let model = DeepFilterNet3::load(&io.model)?;
    let started = Instant::now();
    let out = model.enhance(&audio)?;
    let elapsed = started.elapsed().as_secs_f64();
    let secs = audio.len() as f64 / f64::from(SAMPLE_RATE);
    eprintln!(
        "enhance: {secs:.2} s d'audio ({sr} Hz) en {elapsed:.3} s, RTF {:.4}",
        elapsed / secs.max(1e-9)
    );
    write_wav(&io.output, &out, SAMPLE_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enhance_rejects_zero_rate_before_loading_model() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.wav");
        let output = dir.path().join("output.wav");
        write_wav(&input, &[0.25], SAMPLE_RATE).unwrap();
        let mut wav = std::fs::read(&input).unwrap();
        wav[24..32].fill(0);
        std::fs::write(&input, wav).unwrap();
        let error = run_enhance(
            [
                "--model".into(),
                dir.path()
                    .join("absent.safetensors")
                    .to_string_lossy()
                    .into_owned(),
                "--input".into(),
                input.to_string_lossy().into_owned(),
                "--output".into(),
                output.to_string_lossy().into_owned(),
            ]
            .into_iter(),
        )
        .unwrap_err();
        assert!(matches!(
            error.downcast_ref::<saragossa::InferError>(),
            Some(saragossa::InferError::Audio(_))
        ));
        assert!(error.to_string().contains("taux WAV"));
        assert!(!output.exists());
    }
}
