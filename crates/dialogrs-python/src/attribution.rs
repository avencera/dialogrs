//! The word merging and speaker assignment `DialogPipeline` runs, for results that
//! come from elsewhere, such as the scriptrs and speakrs Python packages
//!
//! Inputs are read by attribute, so any object with the right fields works: a
//! `scriptrs.TimedToken`, a `speakrs.Segment`, or the types in this module

use dialogrs::Word;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use scriptrs::TimedToken;
use speakrs_sdk::speakrs::Segment;

use crate::results::{PyTurn, PyWord};

/// Merge subword tokens, such as a scriptrs result's `tokens`, into words
#[pyfunction]
pub(crate) fn words_from_tokens(tokens: &Bound<'_, PyAny>) -> PyResult<Vec<PyWord>> {
    let tokens = read_all(tokens, "token", |token| {
        Ok(TimedToken {
            // dialogrs merges by text and never reads the id
            token_id: 0,
            text: field(token, "text")?,
            start: field(token, "start")?,
            end: field(token, "end")?,
            confidence: field(token, "confidence")?,
        })
    })?;

    Ok(dialogrs::words_from_tokens(&tokens)
        .into_iter()
        .map(Into::into)
        .collect())
}

/// Give each word the speaker of the segment it overlaps most and group the words
/// into turns
///
/// Raises `ValueError` unless `words` are sorted by start and `segments` are sorted
/// without overlap, as speakrs's `exclusive_segments` are, since out-of-order input
/// would silently give words the wrong speakers
#[pyfunction]
pub(crate) fn assign_speakers(
    words: &Bound<'_, PyAny>,
    segments: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyTurn>> {
    let words = read_all(words, "word", |word| {
        Ok(Word {
            text: field(word, "text")?,
            start: field(word, "start")?,
            end: field(word, "end")?,
            confidence: field(word, "confidence")?,
        })
    })?;
    let segments = read_all(segments, "segment", |segment| {
        Ok(Segment {
            start: field(segment, "start")?,
            end: field(segment, "end")?,
            speaker: field(segment, "speaker")?,
        })
    })?;

    if let Some(index) = words
        .windows(2)
        .position(|pair| pair[1].start < pair[0].start)
    {
        return Err(PyValueError::new_err(format!(
            "words must be sorted by start; word {} starts before word {index}",
            index + 1
        )));
    }

    if let Some(index) = segments
        .windows(2)
        .position(|pair| pair[1].start < pair[0].end)
    {
        return Err(PyValueError::new_err(format!(
            "segments must be sorted without overlap, as speakrs's exclusive_segments are; \
             segment {} starts before segment {index} ends",
            index + 1
        )));
    }

    Ok(dialogrs::assign_speakers(&words, &segments)
        .into_iter()
        .map(Into::into)
        .collect())
}

/// Read every item of an iterable, naming the item that fails
fn read_all<T>(
    items: &Bound<'_, PyAny>,
    what: &str,
    read: impl Fn(&Bound<'_, PyAny>) -> PyResult<T>,
) -> PyResult<Vec<T>> {
    items
        .try_iter()?
        .enumerate()
        .map(|(index, item)| {
            read(&item?).map_err(|error| PyTypeError::new_err(format!("{what} {index}: {error}")))
        })
        .collect()
}

fn field<'py, T: FromPyObjectOwned<'py>>(item: &Bound<'py, PyAny>, name: &str) -> PyResult<T> {
    item.getattr(name)?.extract().map_err(Into::into)
}
