use gigaam_rs::tokenizer::Tokenizer;
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Case {
    ids: Vec<usize>,
    text: String,
}

#[test]
fn decode_matches_sentencepiece_reference() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let tokenizer = Tokenizer::load(
        Vec::new(),
        Some(&root.join("sentencepiece_reference.model")),
    )
    .unwrap();
    let data = std::fs::read(root.join("sentencepiece_reference_cases.json")).unwrap();
    let cases: Vec<Case> = serde_json::from_slice(&data).unwrap();
    assert_eq!(cases.len(), 308);
    for (index, case) in cases.iter().enumerate() {
        assert_eq!(
            tokenizer.decode(&case.ids).unwrap(),
            case.text,
            "case {index}"
        );
    }
}
