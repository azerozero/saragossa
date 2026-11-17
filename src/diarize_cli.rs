//! `saragossa diarize` : diarisation hors ligne d'un WAV en RTTM ou JSON.
use crate::{cli_error, CliResult};
use saragossa::diarization::{to_json, to_rttm, DiarizationModel};
use saragossa::ForwardRuntime;
use std::path::PathBuf;

const HELP: &str = "saragossa diarize --model-dir DOSSIER [--format rttm|json] [--cpu] AUDIO.wav\n\
Diarisation hors ligne nvidia/Nemotron-3-Diarization (8 locuteurs au plus, ordre d'arrivée).\n\
WAV mono ou multicanal, rééchantillonné à 16 kHz ; segments au seuil 0,5 sur stdout.";

pub(crate) fn run(args: impl Iterator<Item = String>) -> CliResult<()> {
    let mut model_dir = None;
    let mut audio = None;
    let mut json = false;
    let mut metal = true;
    let mut args = args;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(());
            }
            "--model-dir" => {
                model_dir = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| cli_error("--model-dir exige un chemin"))?,
                ));
            }
            "--format" => match args.next().as_deref() {
                Some("rttm") => json = false,
                Some("json") => json = true,
                other => return Err(cli_error(format!("--format rttm|json, reçu {other:?}"))),
            },
            "--cpu" => metal = false,
            _ if !arg.starts_with('-') && audio.is_none() => audio = Some(PathBuf::from(arg)),
            _ => return Err(cli_error(format!("argument diarize inconnu : {arg}"))),
        }
    }
    let model_dir = model_dir.ok_or_else(|| cli_error("--model-dir requis"))?;
    let audio = audio.ok_or_else(|| cli_error("fichier audio requis"))?;
    let bytes =
        std::fs::read(&audio).map_err(|e| cli_error(format!("{}: {e}", audio.display())))?;
    let samples = saragossa::wire_audio::wav_to_mono_16k(&bytes)?;
    let model = DiarizationModel::from_model_dir(&model_dir)?;
    #[cfg(all(target_os = "macos", feature = "metal"))]
    let executor = if metal {
        Some(saragossa::MetalExecutor::new()?)
    } else {
        None
    };
    #[cfg(all(target_os = "macos", feature = "metal"))]
    let runtime = executor
        .as_ref()
        .map_or_else(ForwardRuntime::cpu, ForwardRuntime::metal);
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    let runtime = if metal {
        return Err(cli_error("backend metal diarisation indisponible : utilisez --cpu ou recompilez avec --features metal"));
    } else {
        ForwardRuntime::cpu()
    };
    let out = model.diarize(&samples, runtime)?;
    let uri = audio
        .file_stem()
        .map_or_else(|| "audio".to_string(), |s| s.to_string_lossy().into_owned());
    if json {
        println!(
            "{}",
            serde_json::json!({"uri": uri, "duration": samples.len() as f64 / 16_000.0, "segments": to_json(&out.segments)})
        );
    } else {
        print!("{}", to_rttm(&out.segments, &uri));
    }
    Ok(())
}
