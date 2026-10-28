//! Speaker-attributed transcription speed and output on a directory of 16 kHz mono
//! wavs, for scoring against WhisperX
//!
//! Loads every file first, runs the first once untimed as warmup, then times each
//! file from preloaded audio to attributed turns. Writes, per file, `<id>.json`
//! (turns: speaker, start, end, text) and `<id>.rttm` (diarization with overlap,
//! for DER), plus `texts.json` ({id: text}, for scriptrs' `long_wer.py`) and
//! `summary.json`; the WhisperX baseline in `bench/whisperx` writes the same layout
//!
//! ```sh
//! cargo run --release --features cuda --example bench -- \
//!     --models /shared/models/scriptrs-native --wavs earnings21/wav --out out/e21
//! ```

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use dialogrs::{Dialog, DialogPipeline};
use eyre::{Result, bail, eyre};
use scriptrs::{LongFormTranscriptionPipeline, ModelBundle, ParakeetDeployment, Precision};
use serde_json::json;
use speakrs::{ExecutionMode, OwnedDiarizationPipeline};

struct Args {
    models: PathBuf,
    deployment: ParakeetDeployment,
    precision: Precision,
    mode: ExecutionMode,
    wavs: PathBuf,
    out: PathBuf,
    limit: Option<usize>,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out)?;
    let bundle = ModelBundle::from_native_dir(&args.models, args.deployment.clone())
        .with_precision(args.precision);
    let transcriber = LongFormTranscriptionPipeline::from_bundle(bundle)?;
    let diarizer = OwnedDiarizationPipeline::from_pretrained(args.mode)?;
    let mut pipeline = DialogPipeline::new(transcriber, diarizer);

    let mut paths: Vec<PathBuf> = fs::read_dir(&args.wavs)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<_, _>>()?;
    paths.retain(|path| path.extension().is_some_and(|ext| ext == "wav"));
    paths.sort();
    paths.truncate(args.limit.unwrap_or(paths.len()));
    let recordings: Vec<(String, Vec<f32>)> = paths
        .iter()
        .map(|path| Ok((stem(path)?, load_wav(path)?)))
        .collect::<Result<_>>()?;
    let Some((_, first)) = recordings.first() else {
        bail!("no wav files in {}", args.wavs.display());
    };
    pipeline.run(first)?;

    let (mut texts, mut walls) = (BTreeMap::new(), BTreeMap::new());
    for (id, audio) in &recordings {
        let start = Instant::now();
        let dialog = pipeline.run(audio)?;
        let wall = start.elapsed().as_secs_f64();

        write_outputs(&args.out, id, &dialog)?;
        println!(
            "{id}: {:.0} s in {wall:.2} s, {} turns",
            dialog.duration_seconds,
            dialog.turns.len()
        );
        texts.insert(id.clone(), dialog.text);
        walls.insert(id.clone(), wall);
    }

    let audio_s: f64 = recordings
        .iter()
        .map(|(_, audio)| audio.len() as f64 / 16_000.0)
        .sum();
    let wall_s: f64 = walls.values().sum();
    let summary = json!({
        "backend": format!(
            "dialogrs-{:?}-{}-{:?}",
            args.deployment.version(),
            match args.precision {
                Precision::Fp32 => "fp32",
                Precision::Tf32 => "tf32",
            },
            args.mode
        ),
        "files": recordings.len(),
        "audio_s": audio_s,
        "wall_s": wall_s,
        "rtfx": audio_s / wall_s,
    });
    println!("{summary}");
    fs::write(
        args.out.join("texts.json"),
        serde_json::to_string_pretty(&texts)?,
    )?;
    let mut summary = summary;
    summary["walls_s"] = json!(walls);
    fs::write(
        args.out.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    Ok(())
}

fn write_outputs(out: &Path, id: &str, dialog: &Dialog) -> Result<()> {
    let turns: Vec<_> = dialog
        .turns
        .iter()
        .map(|turn| {
            json!({
                "speaker": turn.speaker,
                "start": turn.start,
                "end": turn.end,
                "text": turn.text(),
            })
        })
        .collect();
    fs::write(
        out.join(format!("{id}.json")),
        serde_json::to_string_pretty(&json!({ "turns": turns }))?,
    )?;

    let rttm: String = dialog
        .segments
        .iter()
        .map(|segment| {
            format!(
                "SPEAKER {id} 1 {:.3} {:.3} <NA> <NA> {} <NA> <NA>\n",
                segment.start,
                segment.end - segment.start,
                segment.speaker
            )
        })
        .collect();
    fs::write(out.join(format!("{id}.rttm")), rttm)?;
    Ok(())
}

fn stem(path: &Path) -> Result<String> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(str::to_owned)
        .ok_or_else(|| eyre!("{} has no file stem", path.display()))
}

fn load_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 16_000 {
        bail!("{}: need 16 kHz mono", path.display());
    }

    Ok(match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    })
}

fn parse_args() -> Result<Args> {
    let mut models = None;
    let mut deployment = ParakeetDeployment::v3();
    let mut precision = Precision::default();
    let mut mode = ExecutionMode::Cuda;
    let mut wavs = None;
    let mut out = None;
    let mut limit = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| eyre!("{flag} needs a value"));
        match flag.as_str() {
            "--models" => models = Some(PathBuf::from(value()?)),
            "--version" => {
                deployment = match value()?.as_str() {
                    "v2" => ParakeetDeployment::v2(),
                    "v3" => ParakeetDeployment::v3(),
                    other => bail!("unknown version {other}"),
                }
            }
            "--fp32" => precision = Precision::Fp32,
            "--diarization" => {
                mode = match value()?.as_str() {
                    "cuda" => ExecutionMode::Cuda,
                    "cuda-fast" => ExecutionMode::CudaFast,
                    other => bail!("unknown diarization mode {other}"),
                }
            }
            "--wavs" => wavs = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--limit" => limit = Some(value()?.parse()?),
            other => bail!("unknown argument {other}"),
        }
    }

    Ok(Args {
        models: models.ok_or_else(|| eyre!("--models is required"))?,
        deployment,
        precision,
        mode,
        wavs: wavs.ok_or_else(|| eyre!("--wavs is required"))?,
        out: out.ok_or_else(|| eyre!("--out is required"))?,
        limit,
    })
}
