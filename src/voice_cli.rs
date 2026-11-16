//! Sous-commandes `saragossa enhance` (DeepFilterNet3) et `saragossa separate` (MossFormer2).
use std::path::PathBuf;
use std::time::Instant;

use crate::{cli_error, CliResult};
use saragossa::denoise::{DeepFilterNet3, SAMPLE_RATE};
use saragossa::tts_clone::{decode_wav_mono, resample};

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
    let (samples, sr) = decode_wav_mono(&std::fs::read(path)?)?;
    Ok((resample(&samples, sr, rate), sr))
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
    let model = DeepFilterNet3::load(&io.model)?;
    let (audio, sr) = read_wav(&io.input, SAMPLE_RATE)?;
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
