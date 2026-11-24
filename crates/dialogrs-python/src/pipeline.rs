//! The native pipeline behind `dialogrs.Pipeline`

use std::path::PathBuf;
use std::sync::Mutex;

use dialogrs::DialogPipeline;
use pyo3::prelude::*;
use scriptrs::{
    LanguageSelection, LongFormTranscriptionPipeline, ModelBundle, ParakeetDeployment,
    ParakeetLanguage, ParakeetVersion,
};
use speakrs_sdk::speakrs::{ExecutionMode, OwnedDiarizationPipeline};
use speakrs_sdk::{mode_name, parse_mode};

use crate::results::PyDialog;
use crate::{deployment, dialogrs_err, lock, samples};

/// Native pipeline; `dialogrs.Pipeline` wraps it
#[pyclass(name = "_Pipeline", module = "dialogrs._native", frozen)]
pub(crate) struct PyPipeline {
    // `DialogPipeline::run` takes `&mut self`, and the lock also runs one call at a
    // time, which Core ML needs for predictions on one model
    inner: Mutex<DialogPipeline>,
    settings: Settings,
}

impl PyPipeline {
    fn run(&self, py: Python<'_>, audio: &[f32]) -> PyResult<PyDialog> {
        py.detach(|| lock(&self.inner).run(audio))
            .map(Into::into)
            .map_err(crate::dialog_err)
    }
}

#[pymethods]
impl PyPipeline {
    #[getter]
    fn model(&self) -> &'static str {
        model_name(&self.settings.deployment)
    }

    #[getter]
    fn mode(&self) -> &'static str {
        mode_name(self.settings.mode)
    }

    #[getter]
    fn language(&self) -> Option<&'static str> {
        self.settings.language.map(|language| language.code())
    }

    #[getter]
    fn languages(&self) -> Vec<&'static str> {
        self.settings
            .deployment
            .languages()
            .iter()
            .map(|language| language.code())
            .collect()
    }

    fn transcribe(&self, py: Python<'_>, samples: &Bound<'_, PyAny>) -> PyResult<PyDialog> {
        let audio = samples::extract(samples)?;
        self.run(py, &audio)
    }

    fn transcribe_file(&self, py: Python<'_>, path: PathBuf) -> PyResult<PyDialog> {
        let audio = py
            .detach(|| speakrs_sdk::audio::decode_file_to_mono_16khz(&path))
            .map_err(|error| dialogrs_err("audio_decode", error))?;
        self.run(py, &audio.samples)
    }
}

/// Execution modes compiled into this build, such as `["CPU", "CUDA", "CUDAFast"]`
#[pyfunction]
pub(crate) fn available_modes() -> Vec<&'static str> {
    speakrs_sdk::available_modes()
        .into_iter()
        .map(mode_name)
        .collect()
}

/// Check load settings without loading anything, so a bad argument is reported
/// before the CUDA driver check and any download
#[pyfunction]
#[pyo3(signature = (model, mode, precision, language=None))]
pub(crate) fn _check_settings(
    model: &str,
    mode: &str,
    precision: &str,
    language: Option<&str>,
) -> PyResult<()> {
    Settings::parse(model, mode, precision, language).map(drop)
}

/// Download both models for this platform's backends, then load them
#[pyfunction]
#[pyo3(signature = (model, mode, precision, language=None))]
pub(crate) fn _from_pretrained(
    py: Python<'_>,
    model: &str,
    mode: &str,
    precision: &str,
    language: Option<&str>,
) -> PyResult<PyPipeline> {
    let settings = Settings::parse(model, mode, precision, language)?;
    let (deployment, mode) = (settings.deployment.clone(), settings.mode);
    load(py, settings, move || {
        let bundle = backend::download(deployment)?;
        let diarizer =
            OwnedDiarizationPipeline::from_pretrained(mode).map_err(crate::diarization_err)?;
        Ok((bundle, diarizer))
    })
}

/// Load the ASR model and the diarization models from local directories
#[pyfunction]
#[pyo3(signature = (transcription_dir, diarization_dir, model, mode, precision, language=None))]
pub(crate) fn _from_dir(
    py: Python<'_>,
    transcription_dir: PathBuf,
    diarization_dir: PathBuf,
    model: &str,
    mode: &str,
    precision: &str,
    language: Option<&str>,
) -> PyResult<PyPipeline> {
    let settings = Settings::parse(model, mode, precision, language)?;
    let (deployment, mode) = (settings.deployment.clone(), settings.mode);
    load(py, settings, move || {
        let bundle = backend::from_dir(transcription_dir, deployment);
        let diarizer = OwnedDiarizationPipeline::from_dir(diarization_dir, mode)
            .map_err(crate::diarization_err)?;
        Ok((bundle, diarizer))
    })
}

/// Fetch the models with the GIL released, then build the pipeline from them
fn load(
    py: Python<'_>,
    settings: Settings,
    fetch: impl FnOnce() -> PyResult<(ModelBundle, OwnedDiarizationPipeline)> + Send,
) -> PyResult<PyPipeline> {
    let precision = settings.precision;
    let language = settings.language;
    let pipeline = py.detach(move || {
        let (bundle, diarizer) = fetch()?;
        let transcriber =
            LongFormTranscriptionPipeline::from_bundle(backend::with_precision(bundle, precision))
                .map_err(crate::transcription_err)?;
        Ok::<_, PyErr>(with_language(transcriber, diarizer, language))
    })?;

    Ok(PyPipeline {
        inner: Mutex::new(pipeline),
        settings,
    })
}

/// Pair the two pipelines, keeping the transcriber's own config unless a language
/// was chosen, so the default path matches `DialogPipeline::new`
fn with_language(
    transcriber: LongFormTranscriptionPipeline,
    diarizer: OwnedDiarizationPipeline,
    language: Option<ParakeetLanguage>,
) -> DialogPipeline {
    let Some(language) = language else {
        return DialogPipeline::new(transcriber, diarizer);
    };

    let mut config = transcriber.config().clone();
    config.transcription.language = LanguageSelection::Language(language);
    DialogPipeline::new(transcriber, diarizer).with_transcription_config(config)
}

fn model_name(deployment: &ParakeetDeployment) -> &'static str {
    match deployment.version() {
        ParakeetVersion::V2 => "parakeet/v2",
        ParakeetVersion::V3 => "parakeet/v3",
    }
}

/// Everything a load takes, checked before any download starts
struct Settings {
    deployment: ParakeetDeployment,
    mode: ExecutionMode,
    precision: PrecisionName,
    language: Option<ParakeetLanguage>,
}

impl Settings {
    fn parse(model: &str, mode: &str, precision: &str, language: Option<&str>) -> PyResult<Self> {
        let deployment = deployment(model)?;
        let precision = parse_precision(precision)?;
        let language = language
            .map(|code| parse_language(&deployment, code))
            .transpose()?;

        let mode =
            parse_mode(mode).map_err(|error| dialogrs_err("configuration", error.message))?;
        if !speakrs_sdk::available_modes().contains(&mode) {
            return Err(dialogrs_err(
                "unsupported_mode",
                format!(
                    "diarization mode {} is not in this build; it has {}",
                    mode_name(mode),
                    available_modes().join(", ")
                ),
            ));
        }

        Ok(Self {
            deployment,
            mode,
            precision,
            language,
        })
    }
}

fn parse_language(deployment: &ParakeetDeployment, code: &str) -> PyResult<ParakeetLanguage> {
    ParakeetLanguage::from_code(&code.trim().to_ascii_lowercase())
        .filter(|language| deployment.supports_language(*language))
        .ok_or_else(|| {
            let supported: Vec<_> = deployment
                .languages()
                .iter()
                .map(|language| language.code())
                .collect();
            dialogrs_err(
                "unsupported_language",
                format!(
                    "model {} does not transcribe language `{code}`; it supports {}",
                    model_name(deployment),
                    supported.join(", ")
                ),
            )
        })
}

#[derive(Clone, Copy)]
enum PrecisionName {
    Fp16,
    Tf32,
    Fp32,
}

fn parse_precision(name: &str) -> PyResult<PrecisionName> {
    match name.to_ascii_lowercase().as_str() {
        "fp16" => Ok(PrecisionName::Fp16),
        "tf32" => Ok(PrecisionName::Tf32),
        "fp32" => Ok(PrecisionName::Fp32),
        _ => Err(dialogrs_err(
            "configuration",
            format!("unknown precision `{name}`; expected `fp16`, `tf32` or `fp32`"),
        )),
    }
}

#[cfg(any(
    target_os = "macos",
    all(
        any(target_os = "linux", target_os = "windows"),
        target_arch = "x86_64"
    )
))]
fn download_err(error: impl std::fmt::Display) -> PyErr {
    dialogrs_err("model_download", format!("model download failed: {error}"))
}

/// Core ML models on macOS
#[cfg(target_os = "macos")]
mod backend {
    use std::path::PathBuf;

    use pyo3::PyResult;
    use scriptrs::{ModelBundle, ParakeetDeployment};

    use super::{PrecisionName, download_err};

    /// Core ML runs its own precision, so the setting is only validated
    pub(super) fn with_precision(bundle: ModelBundle, _precision: PrecisionName) -> ModelBundle {
        bundle
    }

    pub(super) fn download(deployment: ParakeetDeployment) -> PyResult<ModelBundle> {
        ModelBundle::from_pretrained_with_deployment(deployment).map_err(download_err)
    }

    pub(super) fn from_dir(path: PathBuf, deployment: ParakeetDeployment) -> ModelBundle {
        ModelBundle::from_dir_with_deployment(path, deployment)
    }
}

/// Native CUDA weights on Linux and Windows x86_64
#[cfg(all(
    any(target_os = "linux", target_os = "windows"),
    target_arch = "x86_64"
))]
mod backend {
    use std::path::PathBuf;

    use pyo3::PyResult;
    use scriptrs::{ModelBundle, ParakeetDeployment, Precision};

    use super::{PrecisionName, download_err};

    pub(super) fn with_precision(bundle: ModelBundle, precision: PrecisionName) -> ModelBundle {
        bundle.with_precision(match precision {
            PrecisionName::Fp16 => Precision::Fp16,
            PrecisionName::Tf32 => Precision::Tf32,
            PrecisionName::Fp32 => Precision::Fp32,
        })
    }

    pub(super) fn download(deployment: ParakeetDeployment) -> PyResult<ModelBundle> {
        ModelBundle::from_pretrained_native(deployment).map_err(download_err)
    }

    pub(super) fn from_dir(path: PathBuf, deployment: ParakeetDeployment) -> ModelBundle {
        ModelBundle::from_native_dir(path, deployment)
    }
}

/// No ASR backend: the package imports, and loading a model reports the platform
#[cfg(not(any(
    target_os = "macos",
    all(
        any(target_os = "linux", target_os = "windows"),
        target_arch = "x86_64"
    )
)))]
mod backend {
    use std::path::PathBuf;

    use pyo3::PyResult;
    use scriptrs::{ModelBundle, ParakeetDeployment};

    use super::PrecisionName;
    use crate::dialogrs_err;

    pub(super) fn with_precision(bundle: ModelBundle, _precision: PrecisionName) -> ModelBundle {
        bundle
    }

    pub(super) fn download(_deployment: ParakeetDeployment) -> PyResult<ModelBundle> {
        Err(dialogrs_err(
            "unsupported_platform",
            "dialogrs runs on macOS (Core ML) and Linux or Windows x86_64 (CUDA) only",
        ))
    }

    pub(super) fn from_dir(path: PathBuf, deployment: ParakeetDeployment) -> ModelBundle {
        ModelBundle::from_dir_with_deployment(path, deployment)
    }
}
