# Changelog

All notable changes to dialogrs are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `DialogPipeline` transcribes with scriptrs and diarizes with speakrs over the same
  16 kHz mono audio at the same time, and returns the transcript as speaker turns.
- `words_from_tokens` merges scriptrs's subword tokens into `Word`s with timestamps,
  and `assign_speakers` gives each word the speaker whose turn it falls in.
- The `cuda` feature runs both on an NVIDIA GPU on Linux or Windows x86_64.
- `Dialog::speakers` gives each speaker label a `SpeakerCentroid` from speakrs:
  the L2-normalised mean of the speaker's 256-d chunk embeddings, with the number
  of embeddings and seconds of speech behind it, for matching the same person
  across recordings. The bench example writes them to each `<id>.json`.

[Unreleased]: https://github.com/avencera/dialogrs/commits/master
