//! Sous-commandes `saragossa enhance` (DeepFilterNet3) et `saragossa separate` (MossFormer2).
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::time::Instant;

use crate::{cli_error, CliResult};
use saragossa::denoise::wav::{output_header, WavStream};
use saragossa::denoise::{DeepFilterNet3, DenoiseStream, StreamResampler, SAMPLE_RATE};

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

#[cfg(test)]
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
    if io.input != std::path::Path::new("-")
        && io.output.exists()
        && std::fs::canonicalize(&io.input)? == std::fs::canonicalize(&io.output)?
    {
        return Err(cli_error("entrée et sortie doivent être distinctes"));
    }
    let input: Box<dyn Read> = if io.input == std::path::Path::new("-") {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(&io.input)?)
    };
    let mut reader = WavStream::new(input).map_err(saragossa::InferError::Audio)?;
    let sr = reader.sample_rate();
    let model = DeepFilterNet3::load(&io.model)?;
    let started = Instant::now();
    let samples = if io.output == std::path::Path::new("-") {
        #[cfg(unix)]
        {
            use std::os::fd::AsFd;
            let file = std::fs::File::from(std::io::stdout().as_fd().try_clone_to_owned()?);
            enhance_to_file(&mut reader, &model, file)?
        }
        #[cfg(not(unix))]
        {
            enhance_to_writer(&mut reader, &model, &mut std::io::stdout().lock())?
        }
    } else {
        enhance_to_file(&mut reader, &model, std::fs::File::create(&io.output)?)?
    };
    let elapsed = started.elapsed().as_secs_f64();
    let secs = samples as f64 / f64::from(SAMPLE_RATE);
    eprintln!(
        "enhance: {secs:.2} s d'audio ({sr} Hz) en {elapsed:.3} s, RTF {:.4}",
        elapsed / secs.max(1e-9)
    );
    Ok(())
}

fn enhance_to_file<R: Read>(
    reader: &mut WavStream<R>,
    model: &DeepFilterNet3,
    mut file: std::fs::File,
) -> CliResult<u64> {
    let start = file.stream_position().ok();
    let count = enhance_to_writer(reader, model, &mut file)?;
    if let Some(start) = start {
        file.seek(SeekFrom::Start(start))?;
        file.write_all(&output_header(Some(count))?)?;
        file.seek(SeekFrom::End(0))?;
        file.flush()?;
    }
    Ok(count)
}

fn enhance_to_writer<R: Read>(
    reader: &mut WavStream<R>,
    model: &DeepFilterNet3,
    writer: &mut impl Write,
) -> CliResult<u64> {
    let mut writer = BufWriter::with_capacity(8192, writer);
    writer.write_all(&output_header(None)?)?;
    let mut resampler = StreamResampler::new(reader.sample_rate(), SAMPLE_RATE)?;
    let threads = saragossa::runtime_flags::denoise_threads();
    let mut stream = DenoiseStream::with_threads(model, threads);
    let mut count = 0_u64;
    let mut emit = |samples: &[f32]| -> CliResult<()> {
        for sample in samples {
            writer.write_all(&sample.to_le_bytes())?;
        }
        count = count
            .checked_add(samples.len() as u64)
            .ok_or_else(|| cli_error("compteur WAV débordé"))?;
        Ok(())
    };
    loop {
        let chunk = reader.read_chunk().map_err(saragossa::InferError::Audio)?;
        if chunk.is_empty() {
            break;
        }
        emit(&stream.push(&resampler.push(&chunk)?)?)?;
    }
    reader.finish_input()?;
    emit(&stream.push(&resampler.finish())?)?;
    emit(&stream.finish()?)?;
    writer.flush()?;
    Ok(count)
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
