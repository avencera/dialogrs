//! Speaker-attributed transcription: scriptrs for what was said, speakrs for who
//! said it
//!
//! [`DialogPipeline`] runs Parakeet ASR and speaker diarization over the same
//! 16 kHz mono audio at the same time, merges the ASR's subword tokens into
//! [`Word`]s, and gives each word the speaker whose turn it falls in, as [`Turn`]s.
//! Each speaker with a usable embedding also gets a [`SpeakerCentroid`], a voice
//! embedding for matching the same person across recordings

mod attribution;
mod pipeline;
mod words;

pub use attribution::{Turn, assign_speakers};
pub use pipeline::{Dialog, DialogError, DialogPipeline};
pub use speakrs::SpeakerCentroid;
pub use words::{Word, words_from_tokens};
