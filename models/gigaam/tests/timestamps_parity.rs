use gigaam_rs::decode::words_from_frames;
use gigaam_rs::tokenizer::Tokenizer;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    pieces: Vec<String>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    ids: Vec<usize>,
    frames: Vec<usize>,
    sample_count: usize,
    encoded_frames: usize,
    words: Vec<ExpectedWord>,
}

#[derive(Deserialize)]
struct ExpectedWord {
    text: String,
    start: f64,
    end: f64,
}

#[test]
fn upstream_python_frames_to_words_parity() {
    let fixture: Reference =
        serde_json::from_str(include_str!("fixtures/timestamps_reference_cases.json")).unwrap();
    let tokenizer = Tokenizer::Character(fixture.pieces);
    for (index, case) in fixture.cases.iter().enumerate() {
        let words = words_from_frames(
            &tokenizer,
            &case.ids,
            &case.frames,
            case.sample_count,
            case.encoded_frames,
        )
        .unwrap();
        assert_eq!(words.len(), case.words.len(), "case {index}");
        for (word, expected) in words.iter().zip(&case.words) {
            assert_eq!(word.text, expected.text, "case {index}");
            assert!(
                (word.start - expected.start).abs() <= 1e-12,
                "case {index}: start"
            );
            assert!(
                (word.end - expected.end).abs() <= 1e-12,
                "case {index}: end"
            );
        }
    }
}
