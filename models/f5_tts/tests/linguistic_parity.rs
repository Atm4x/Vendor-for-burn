use f5_tts_native::{pinyin::ChinesePinyin, text_frontend::F5TextFrontend, vocab::Vocabulary};
use jieba_rs::Jieba;
use serde_json::Value;
use std::collections::BTreeSet;

fn han_only(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|c| ('\u{3100}'..='\u{9fff}').contains(&c))
}

#[test]
fn compare_jieba_and_pypinyin_against_python_corpus() {
    let jieba = Jieba::new();
    let dictionary = ChinesePinyin::bundled().unwrap();
    let corpus = include_str!("fixtures/rjieba_reference.jsonl");
    let mut segmentation_cases = 0;
    let mut segmentation_matches = 0;
    let mut pronunciation_cases = 0;
    let mut pronunciation_matches = 0;
    let mut bad_segments = Vec::new();
    let mut bad_pronunciations = Vec::new();
    for line in corpus.lines() {
        let row: Value = serde_json::from_str(line).unwrap();
        let text = row["text"].as_str().unwrap();
        let expected: Vec<_> = row["segments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap())
            .collect();
        let actual = jieba.cut(text, true);
        segmentation_cases += 1;
        if actual.iter().map(|s| s.word).eq(expected.iter().copied()) {
            segmentation_matches += 1;
        } else if bad_segments.len() < 6 {
            bad_segments.push((
                text.to_owned(),
                expected.join("|"),
                actual.iter().map(|s| s.word).collect::<Vec<_>>().join("|"),
            ));
        }
        for (i, word) in expected.iter().enumerate() {
            if !han_only(word) {
                continue;
            }
            let reference: Vec<_> = row["syllables"][i]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            let ours = dictionary.syllables_for_han(word);
            pronunciation_cases += 1;
            if let Ok(actual) = &ours {
                if actual
                    .iter()
                    .map(String::as_str)
                    .eq(reference.iter().copied())
                {
                    pronunciation_matches += 1;
                    continue;
                }
            }
            if bad_pronunciations.len() < 6 {
                bad_pronunciations.push((
                    word.to_string(),
                    reference.join(" "),
                    format!("{ours:?}"),
                ));
            }
        }
    }
    println!(
        "SEGMENTATION: {segmentation_matches}/{segmentation_cases}; examples: {bad_segments:?}"
    );
    println!(
        "PRONUNCIATION: {pronunciation_matches}/{pronunciation_cases}; examples: {bad_pronunciations:?}"
    );
    assert_eq!(segmentation_cases, 1809);
    assert_eq!(segmentation_matches, segmentation_cases);
    assert_eq!(pronunciation_cases, 5859);
    assert_eq!(pronunciation_matches, pronunciation_cases);
}

#[test]
fn complete_f5_pinyin_token_sequence_matches_original_python() {
    let corpus: Vec<Value> = include_str!("fixtures/rjieba_f5_tokens_expected.jsonl")
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let mut tokens = BTreeSet::new();
    for item in &corpus {
        for value in item["tokens"].as_array().unwrap() {
            tokens.insert(value.as_str().unwrap().to_owned());
        }
    }
    tokens.remove(" ");
    let vocabulary = Vocabulary::from_lines(
        std::iter::once(" ".to_owned())
            .chain(tokens)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let dictionary = ChinesePinyin::bundled().unwrap();
    let frontend = F5TextFrontend::new(&vocabulary, &dictionary);
    let mut matched = 0;
    let mut mismatches = Vec::new();
    for item in &corpus {
        let text = item["text"].as_str().unwrap();
        let expected = item["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| vocabulary.token_id(s.as_str().unwrap()))
            .collect::<Vec<_>>();
        let actual = frontend.encode(text);
        if actual.as_ref().is_ok_and(|ids| ids == &expected) {
            matched += 1;
        } else if mismatches.len() < 6 {
            mismatches.push((text.to_owned(), expected, actual));
        }
    }
    println!(
        "TOKEN SEQUENCE: {matched}/{}; examples: {mismatches:?}",
        corpus.len()
    );
    assert_eq!(corpus.len(), 1809);
    assert_eq!(matched, corpus.len());
}
