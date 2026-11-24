//! Result types returned to Python

use dialogrs::{Dialog, SpeakerCentroid, Turn, Word};
use pyo3::prelude::*;
use speakrs_sdk::speakrs::Segment;

/// A word with its time span, merged from the ASR's subword tokens
#[pyclass(name = "Word", module = "dialogrs", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyWord {
    /// The word as transcribed, punctuation included
    text: String,
    /// Start time in seconds
    start: f64,
    /// End time in seconds
    end: f64,
    /// The lowest confidence of the word's tokens
    confidence: f32,
}

#[pymethods]
impl PyWord {
    #[new]
    fn new(text: String, start: f64, end: f64, confidence: f32) -> Self {
        Self {
            text,
            start,
            end,
            confidence,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Word(text={:?}, start={:.3}, end={:.3}, confidence={:.3})",
            self.text, self.start, self.end, self.confidence
        )
    }
}

impl From<Word> for PyWord {
    fn from(word: Word) -> Self {
        Self {
            text: word.text,
            start: word.start,
            end: word.end,
            confidence: word.confidence,
        }
    }
}

/// Consecutive words from one speaker
#[pyclass(name = "Turn", module = "dialogrs", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyTurn {
    /// The diarization label, such as `SPEAKER_00`; `None` when diarization found
    /// no speech at all
    speaker: Option<String>,
    /// Start of the first word in seconds
    start: f64,
    /// End of the last word in seconds
    end: f64,
    /// The words in time order
    words: Vec<PyWord>,
}

#[pymethods]
impl PyTurn {
    /// The turn's words joined by spaces
    #[getter]
    fn text(&self) -> String {
        let words: Vec<&str> = self.words.iter().map(|word| word.text.as_str()).collect();
        words.join(" ")
    }

    fn __str__(&self) -> String {
        self.text()
    }

    fn __repr__(&self) -> String {
        format!(
            "Turn(speaker={}, start={:.3}, end={:.3}, text={:?})",
            self.speaker
                .as_deref()
                .map_or_else(|| "None".to_owned(), |speaker| format!("{speaker:?}")),
            self.start,
            self.end,
            self.text()
        )
    }
}

impl From<Turn> for PyTurn {
    fn from(turn: Turn) -> Self {
        Self {
            speaker: turn.speaker,
            start: turn.start,
            end: turn.end,
            words: turn.words.into_iter().map(Into::into).collect(),
        }
    }
}

/// A span of speech from one speaker
#[pyclass(name = "Segment", module = "dialogrs", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PySegment {
    /// Start time in seconds
    start: f64,
    /// End time in seconds
    end: f64,
    /// Speaker label, such as `SPEAKER_00`
    speaker: String,
}

#[pymethods]
impl PySegment {
    #[new]
    fn new(start: f64, end: f64, speaker: String) -> Self {
        Self {
            start,
            end,
            speaker,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Segment(start={:.3}, end={:.3}, speaker={:?})",
            self.start, self.end, self.speaker
        )
    }
}

impl From<Segment> for PySegment {
    fn from(segment: Segment) -> Self {
        Self {
            start: segment.start,
            end: segment.end,
            speaker: segment.speaker,
        }
    }
}

/// A speaker's voice embedding, for matching the same person across recordings
#[pyclass(name = "SpeakerCentroid", module = "dialogrs", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PySpeakerCentroid {
    /// The label this speaker's turns and segments carry, such as `SPEAKER_00`
    label: String,
    /// L2-normalised mean of the speaker's 256-d chunk embeddings, so the cosine
    /// similarity of two centroids is their dot product
    embedding: Vec<f32>,
    /// Number of chunk embeddings averaged into `embedding`
    chunk_count: usize,
    /// Seconds of the speaker's speech, overlapped speech included
    speech_seconds: f64,
}

#[pymethods]
impl PySpeakerCentroid {
    fn __repr__(&self) -> String {
        format!(
            "SpeakerCentroid(label={:?}, chunk_count={}, speech_seconds={:.3})",
            self.label, self.chunk_count, self.speech_seconds
        )
    }
}

impl From<SpeakerCentroid> for PySpeakerCentroid {
    fn from(centroid: SpeakerCentroid) -> Self {
        Self {
            label: centroid.label,
            embedding: centroid.embedding,
            chunk_count: centroid.chunk_count,
            speech_seconds: centroid.speech_seconds,
        }
    }
}

/// Speaker-attributed transcript of one recording
#[pyclass(name = "Dialog", module = "dialogrs", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyDialog {
    /// Consecutive words from one speaker, in time order
    turns: Vec<PyTurn>,
    /// The transcript without speakers
    text: String,
    /// Diarization with overlapping speech kept, for RTTM output and DER
    segments: Vec<PySegment>,
    /// Voice embeddings of the speakers that have one, in label order
    speakers: Vec<PySpeakerCentroid>,
    /// Input duration in seconds
    duration_seconds: f64,
}

#[pymethods]
impl PyDialog {
    fn __str__(&self) -> &str {
        &self.text
    }

    fn __repr__(&self) -> String {
        format!(
            "Dialog(turns={}, speakers={}, duration_seconds={:.3})",
            self.turns.len(),
            self.speakers.len(),
            self.duration_seconds
        )
    }
}

impl From<Dialog> for PyDialog {
    fn from(dialog: Dialog) -> Self {
        Self {
            turns: dialog.turns.into_iter().map(Into::into).collect(),
            text: dialog.text,
            segments: dialog.segments.into_iter().map(Into::into).collect(),
            speakers: dialog.speakers.into_iter().map(Into::into).collect(),
            duration_seconds: dialog.duration_seconds,
        }
    }
}
