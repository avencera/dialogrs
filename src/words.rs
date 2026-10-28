use scriptrs::TimedToken;

/// A word with its time span, merged from the ASR's subword tokens
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    /// The word as transcribed, punctuation included
    pub text: String,
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// The lowest confidence of its tokens, so one doubtful piece flags the word
    pub confidence: f32,
}

/// Merges subword tokens into words
///
/// A token whose text starts with a space begins a word, as SentencePiece marks
/// word starts; any other token, punctuation included, extends the word before it
pub fn words_from_tokens(tokens: &[TimedToken]) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    // a bare word-start token carries its boundary to the next piece
    let mut boundary = true;
    for token in tokens {
        boundary |= token.text.starts_with(' ');
        let text = token.text.trim_start();
        if text.is_empty() {
            continue;
        }

        match words.last_mut() {
            Some(word) if !boundary => {
                word.text.push_str(text);
                word.end = token.end;
                word.confidence = word.confidence.min(token.confidence);
            }
            _ => words.push(Word {
                text: text.to_owned(),
                start: token.start,
                end: token.end,
                confidence: token.confidence,
            }),
        }
        boundary = false;
    }
    words
}

#[cfg(test)]
mod tests {
    use scriptrs::TimedToken;

    use super::words_from_tokens;

    fn token(text: &str, start: f64, confidence: f32) -> TimedToken {
        TimedToken {
            token_id: 0,
            text: text.to_owned(),
            start,
            end: start + 0.08,
            confidence,
        }
    }

    #[test]
    fn merges_pieces_and_punctuation_into_words() {
        let tokens = [
            token(" Good", 0.0, 0.9),
            token(" after", 0.2, 0.8),
            token("noon", 0.3, 0.6),
            token(",", 0.4, 0.95),
            token(" ", 0.6, 0.9),
            token("everyone", 0.7, 0.7),
        ];
        let words = words_from_tokens(&tokens);
        let texts: Vec<&str> = words.iter().map(|word| word.text.as_str()).collect();
        assert_eq!(texts, ["Good", "afternoon,", "everyone"]);
        assert_eq!((words[1].start, words[1].end), (0.2, 0.4 + 0.08));
        assert_eq!(words[1].confidence, 0.6);
        assert_eq!(words[2].start, 0.7);
    }
}
