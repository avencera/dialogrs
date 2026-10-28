use std::panic;
use std::thread;

use scriptrs::{LongFormTranscriptionPipeline, TranscriptionError};
use speakrs::{OwnedDiarizationPipeline, PipelineError, Segment};

use crate::attribution::{Turn, assign_speakers};
use crate::words::words_from_tokens;

/// Speaker-attributed transcript of one recording
#[derive(Debug, Clone)]
pub struct Dialog {
    /// Consecutive words from one speaker, in time order
    pub turns: Vec<Turn>,
    /// The ASR transcript without speakers
    pub text: String,
    /// Diarization with overlapping speech kept, for RTTM output and DER
    pub segments: Vec<Segment>,
    /// Input duration in seconds
    pub duration_seconds: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum DialogError {
    #[error("transcription failed: {0}")]
    Transcription(#[from] TranscriptionError),
    #[error("diarization failed: {0}")]
    Diarization(#[from] PipelineError),
}

/// Long-form Parakeet ASR and speaker diarization over the same audio
pub struct DialogPipeline {
    transcriber: LongFormTranscriptionPipeline,
    diarizer: OwnedDiarizationPipeline,
}

impl DialogPipeline {
    pub fn new(
        transcriber: LongFormTranscriptionPipeline,
        diarizer: OwnedDiarizationPipeline,
    ) -> Self {
        Self {
            transcriber,
            diarizer,
        }
    }

    /// Transcribes and diarizes `audio` (mono f32 at 16 kHz) at the same time, then
    /// gives each word a speaker
    ///
    /// On one GPU the two pipelines run on their own CUDA streams, so the device
    /// overlaps their kernels and the CPU stages of each hide behind the other
    pub fn run(&mut self, audio: &[f32]) -> Result<Dialog, DialogError> {
        // the transcriber is `Send` but not `Sync`, so its thread borrows it mutably
        let (transcriber, diarizer) = (&mut self.transcriber, &mut self.diarizer);
        let (transcription, diarization) = thread::scope(|scope| {
            let asr = scope.spawn(move || transcriber.run(audio));
            let diarization = diarizer.run(audio);
            let transcription = asr
                .join()
                .unwrap_or_else(|payload| panic::resume_unwind(payload));
            (transcription, diarization)
        });
        let (transcription, diarization) = (transcription?, diarization?);

        let words = words_from_tokens(&transcription.tokens);
        Ok(Dialog {
            turns: assign_speakers(&words, &diarization.exclusive_segments),
            text: transcription.text,
            segments: diarization.segments,
            duration_seconds: transcription.duration_seconds,
        })
    }
}
