use speakrs::Segment;

use crate::words::Word;

/// A lone word this close to turns on both sides is timestamp jitter; a real reply
/// like "Yeah?" usually has a pause before or after it
const JITTER_GAP: f64 = 0.6;

/// Consecutive words from one speaker
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    /// The diarization label, such as `SPEAKER_00`; `None` when diarization found
    /// no speech at all
    pub speaker: Option<String>,
    /// Start of the first word in seconds
    pub start: f64,
    /// End of the last word in seconds
    pub end: f64,
    /// The words in time order
    pub words: Vec<Word>,
}

impl Turn {
    /// The turn's words joined by spaces
    pub fn text(&self) -> String {
        let words: Vec<&str> = self.words.iter().map(|word| word.text.as_str()).collect();
        words.join(" ")
    }

    fn push(&mut self, word: Word) {
        self.end = word.end;
        self.words.push(word);
    }
}

/// Gives each word the speaker of the diarization segment it overlaps most and
/// groups the words into turns
///
/// `segments` must be sorted by time without overlap, as speakrs'
/// `exclusive_segments` are, and `words` sorted by start. A word in a pause between
/// segments goes to the nearest one, and a lone word squeezed between two turns of
/// the same speaker joins that speaker, since word times drift at speaker changes
pub fn assign_speakers(words: &[Word], segments: &[Segment]) -> Vec<Turn> {
    let mut turns: Vec<Turn> = Vec::new();
    let mut index = 0;
    for word in words {
        // both lists are sorted by time, so the search only moves forward
        while index < segments.len() && segments[index].end <= word.start {
            index += 1;
        }

        let speaker = best_speaker(segments, index, word);
        if let Some(turn) = turns.last_mut()
            && turn.speaker == speaker
        {
            turn.push(word.clone());
        } else if is_jitter(&turns, speaker.as_deref(), word.start) {
            // word times drift at speaker changes, so the lone word between two
            // turns of this speaker is theirs
            let lone = turns.pop().expect("is_jitter needs two turns");
            let before = turns.last_mut().expect("is_jitter needs two turns");
            before.words.extend(lone.words);
            before.push(word.clone());
        } else {
            turns.push(Turn {
                speaker,
                start: word.start,
                end: word.end,
                words: vec![word.clone()],
            });
        }
    }
    turns
}

fn is_jitter(turns: &[Turn], speaker: Option<&str>, start: f64) -> bool {
    let [.., before, lone] = turns else {
        return false;
    };
    lone.words.len() == 1
        && before.speaker.as_deref() == speaker
        && (lone.start - before.end).max(start - lone.end) < JITTER_GAP
}

fn best_speaker(segments: &[Segment], index: usize, word: &Word) -> Option<String> {
    let mut best: Option<(&Segment, f64)> = None;
    for segment in &segments[index..] {
        if segment.start >= word.end {
            break;
        }

        let overlap = segment.end.min(word.end) - segment.start.max(word.start);
        if overlap > best.map_or(0.0, |(_, best)| best) {
            best = Some((segment, overlap));
        }
    }
    if let Some((segment, _)) = best {
        return Some(segment.speaker.clone());
    }

    // a word in a pause between speakers goes to whoever is closest in time
    let distance = |segment: &Segment| (segment.start - word.end).max(word.start - segment.end);
    segments[index.saturating_sub(1)..(index + 1).min(segments.len())]
        .iter()
        .min_by(|a, b| distance(a).total_cmp(&distance(b)))
        .map(|segment| segment.speaker.clone())
}

#[cfg(test)]
mod tests {
    use speakrs::Segment;

    use super::assign_speakers;
    use crate::words::Word;

    fn word(text: &str, start: f64, end: f64) -> Word {
        Word {
            text: text.to_owned(),
            start,
            end,
            confidence: 1.0,
        }
    }

    fn segment(speaker: &str, start: f64, end: f64) -> Segment {
        Segment {
            start,
            end,
            speaker: speaker.to_owned(),
        }
    }

    fn summary(turns: &[super::Turn]) -> Vec<(Option<&str>, String)> {
        turns
            .iter()
            .map(|turn| (turn.speaker.as_deref(), turn.text()))
            .collect()
    }

    #[test]
    fn assigns_by_overlap_then_proximity_and_absorbs_jitter() {
        let segments = [
            segment("A", 0.0, 2.0),
            segment("B", 2.0, 4.0),
            segment("A", 4.0, 8.0),
            segment("B", 9.0, 10.0),
        ];
        let words = [
            word("hello", 0.1, 0.5),
            // mostly inside B's turn
            word("there", 1.8, 2.6),
            word("ok", 2.8, 3.2),
            word("so", 4.0, 4.1),
            word("hmm", 4.3, 4.4),
            // in the pause after A's segment, nearest to B
            word("yes", 8.8, 8.9),
        ];
        let segments_b_jitter = [
            segment("A", 0.0, 2.0),
            segment("B", 2.0, 3.0),
            segment("A", 3.1, 3.3),
            segment("B", 3.3, 6.0),
        ];
        // "two" overlaps the short A segment, but sits 0.2 s and 0.15 s from B's words
        let jitter_words = [
            word("one", 2.6, 2.95),
            word("two", 3.15, 3.25),
            word("three", 3.4, 3.6),
        ];

        assert_eq!(
            summary(&assign_speakers(&words, &segments)),
            [
                (Some("A"), "hello".to_owned()),
                (Some("B"), "there ok".to_owned()),
                (Some("A"), "so hmm".to_owned()),
                (Some("B"), "yes".to_owned()),
            ]
        );
        assert_eq!(
            summary(&assign_speakers(&jitter_words, &segments_b_jitter)),
            [(Some("B"), "one two three".to_owned())]
        );
        assert_eq!(
            summary(&assign_speakers(&words[..1], &[])),
            [(None, "hello".to_owned())]
        );
    }
}
