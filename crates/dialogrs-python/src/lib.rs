//! The `dialogrs._native` extension module
//!
//! The Python package in `python/dialogrs` wraps the pipeline class and
//! re-exports the result types and functions from here unchanged

use std::sync::{Mutex, MutexGuard, PoisonError};

use dialogrs::DialogError;
use pyo3::prelude::*;
use scriptrs::{ParakeetDeployment, TranscriptionError};
use speakrs_sdk::speakrs::PipelineError;

pyo3::import_exception!(dialogrs._errors, DialogrsError);

mod attribution;
mod pipeline;
mod results;
mod samples;

use attribution::{assign_speakers, words_from_tokens};
use pipeline::{_check_settings, _from_dir, _from_pretrained, PyPipeline, available_modes};
use results::{PyDialog, PySegment, PySpeakerCentroid, PyTurn, PyWord};

/// Transcription backend compiled into this build, or `None` where scriptrs has none
const BACKEND: Option<&str> = if cfg!(target_os = "macos") {
    Some("coreml")
} else if cfg!(all(
    any(target_os = "linux", target_os = "windows"),
    target_arch = "x86_64"
)) {
    Some("cuda")
} else {
    None
};

pub(crate) fn dialogrs_err(code: &str, message: impl ToString) -> PyErr {
    DialogrsError::new_err((code.to_owned(), message.to_string()))
}

pub(crate) fn transcription_err(error: TranscriptionError) -> PyErr {
    dialogrs_err(transcription_code(&error), error)
}

pub(crate) fn diarization_err(error: PipelineError) -> PyErr {
    dialogrs_err(diarization_code(&error), error)
}

pub(crate) fn dialog_err(error: DialogError) -> PyErr {
    match error {
        DialogError::Transcription(error) => transcription_err(error),
        DialogError::Diarization(error) => diarization_err(error),
    }
}

/// Stable code for a transcription failure, so Python callers can branch on it
/// without parsing messages
fn transcription_code(error: &TranscriptionError) -> &'static str {
    match error {
        TranscriptionError::EmptyAudio => "empty_audio",
        TranscriptionError::AudioTooLong { .. } => "audio_too_long",
        TranscriptionError::InvalidAudioSample { .. } => "invalid_audio",
        TranscriptionError::UnsupportedLanguage { .. } => "unsupported_language",
        TranscriptionError::InvalidConfiguration(_) => "configuration",
        TranscriptionError::MissingModelAsset { .. }
        | TranscriptionError::LayoutMismatch { .. } => "missing_model",
        TranscriptionError::InvalidVocabulary(_) | TranscriptionError::InvalidWeights(_) => {
            "invalid_model"
        }
        TranscriptionError::UnsupportedPlatform => "unsupported_platform",
        #[cfg(all(
            any(target_os = "linux", target_os = "windows"),
            target_arch = "x86_64"
        ))]
        TranscriptionError::Cuda(_) => "cuda",
        TranscriptionError::CoreMl(_) => "coreml",
        TranscriptionError::Io(_) => "io",
        // a request in a failed batch reports why the batch failed
        TranscriptionError::BatchFailed(error) => transcription_code(error),
        _ => "transcription",
    }
}

/// Stable code for a diarization failure
fn diarization_code(error: &PipelineError) -> &'static str {
    match error {
        PipelineError::ModelLoad(_) => "model_load",
        PipelineError::UnsupportedExecutionMode(_) => "unsupported_mode",
        _ => "diarization",
    }
}

pub(crate) fn deployment(model: &str) -> PyResult<ParakeetDeployment> {
    match model {
        "parakeet/v2" => Ok(ParakeetDeployment::v2()),
        "parakeet/v3" => Ok(ParakeetDeployment::v3()),
        other => Err(dialogrs_err(
            "configuration",
            format!("unknown model `{other}`; expected `parakeet/v2` or `parakeet/v3`"),
        )),
    }
}

// a panic while holding the pipeline lock leaves the pipeline itself unchanged
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Language codes a model transcribes, such as `["en"]` for `parakeet/v2`
#[pyfunction]
fn languages(model: &str) -> PyResult<Vec<&'static str>> {
    Ok(deployment(model)?
        .languages()
        .iter()
        .map(|language| language.code())
        .collect())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("BACKEND", BACKEND)?;
    m.add_class::<PyWord>()?;
    m.add_class::<PyTurn>()?;
    m.add_class::<PySegment>()?;
    m.add_class::<PySpeakerCentroid>()?;
    m.add_class::<PyDialog>()?;
    m.add_class::<PyPipeline>()?;
    m.add_function(wrap_pyfunction!(_check_settings, m)?)?;
    m.add_function(wrap_pyfunction!(_from_pretrained, m)?)?;
    m.add_function(wrap_pyfunction!(_from_dir, m)?)?;
    m.add_function(wrap_pyfunction!(languages, m)?)?;
    m.add_function(wrap_pyfunction!(available_modes, m)?)?;
    m.add_function(wrap_pyfunction!(words_from_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(assign_speakers, m)?)?;
    Ok(())
}
