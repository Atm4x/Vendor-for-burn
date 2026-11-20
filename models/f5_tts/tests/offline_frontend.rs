#[derive(Debug, Clone, PartialEq, Eq)]
pub enum F5Error {
    InvalidConfiguration(&'static str),
    InvalidShape(&'static str),
    InvalidCheckpoint(String),
    ModelExecution(String),
}
pub type Result<T> = std::result::Result<T, F5Error>;

#[path = "../src/cfm.rs"]
mod cfm;
#[path = "../src/standard_chunking.rs"]
mod standard_chunking;
#[path = "../src/text_frontend.rs"]
mod text_frontend;
#[path = "../src/vocab.rs"]
mod vocab;

#[test]
fn source_derived_reference_text_includes_inference_trailing_space() {
    assert_eq!(
        text_frontend::effective_reference_transcript("Hello").unwrap(),
        "Hello.  "
    );
    assert_eq!(
        text_frontend::effective_reference_transcript("你好。").unwrap(),
        "你好。"
    );
}

fn from_hex(hex: &str) -> String {
    let bytes = hex
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect::<Vec<u8>>();
    String::from_utf8(bytes).unwrap()
}

#[test]
fn thousand_original_python_chunk_cases_match_byte_for_byte() {
    let fixture = include_str!("fixtures/standard_chunk_python_expected.tsv");
    let mut cases = 0;
    for line in fixture.lines() {
        let mut cols = line.splitn(4, '\t');
        let size = cols.next().unwrap().parse::<usize>().unwrap();
        let input = from_hex(cols.next().unwrap());
        let length = cols.next().unwrap().parse::<usize>().unwrap();
        let packed = cols.next().unwrap();
        let expected = if length == 0 {
            Vec::new()
        } else {
            packed.split('|').map(from_hex).collect::<Vec<String>>()
        };
        let actual = standard_chunking::chunk_text(&input, size);
        assert_eq!(
            actual, expected,
            "case {cases}, max_chars={size}, input={input:?}"
        );
        cases += 1;
    }
    assert_eq!(cases, 1_006);
}

#[test]
fn fifty_two_original_pytorch_cfm_timelines_match() {
    let fixture = include_str!("fixtures/cfm_pytorch_expected.tsv");
    let mut count = 0;
    for row in fixture.lines() {
        let mut parts = row.splitn(3, '\t');
        let steps = parts.next().unwrap().parse::<usize>().unwrap();
        let sway = match parts.next().unwrap() {
            "none" => None,
            value => Some(value.parse::<f32>().unwrap()),
        };
        let actual = cfm::SamplingConfig {
            steps,
            cfg_strength: 2.0,
            sway_sampling_coef: sway,
            empirically_pruned_steps: true,
        }
        .timeline()
        .unwrap();
        let expected = parts
            .next()
            .unwrap()
            .split(',')
            .map(|x| x.parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len(), "Steps {steps}, sway {sway:?}");
        for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() < 2e-6,
                "Steps {steps}, sway {sway:?}, t[{index}]: {a}, {b}"
            );
        }
        count += 1;
    }
    assert_eq!(count, 52);
}

#[test]
fn pytorch_cfm_trajectory_with_cfg_and_inpainting_matches() {
    let fixture = include_str!("fixtures/cfm_trajectory_pytorch_expected.tsv");
    let condition = [
        0.45, 0.1, -0.8, 0.91, -0.4, 0.24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let starting = [
        0.2, -0.4, 0.1, 0.3, 0.55, -0.8, 0.91, 0.13, -0.6, -0.42, 0.15, 0.3, 0.21, -0.77, 0.48,
    ];
    let mut cases = 0;
    for row in fixture.lines() {
        let mut parts = row.splitn(4, '\t');
        let steps = parts.next().unwrap().parse::<usize>().unwrap();
        let sway = match parts.next().unwrap() {
            "none" => None,
            value => Some(value.parse::<f32>().unwrap()),
        };
        let cfg = parts.next().unwrap().parse::<f32>().unwrap();
        let target = parts
            .next()
            .unwrap()
            .split(',')
            .map(|v| v.parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        let timeline = cfm::SamplingConfig {
            steps,
            cfg_strength: cfg,
            sway_sampling_coef: sway,
            empirically_pruned_steps: true,
        }
        .timeline()
        .unwrap();
        let mut samples = starting;
        for times in timeline.windows(2) {
            let t = times[0];
            let conditional = samples
                .iter()
                .enumerate()
                .map(|(i, &x)| x * 0.031 + condition[i] * 0.021 + t * 0.17)
                .collect::<Vec<_>>();
            let unconditional = samples
                .iter()
                .map(|&x| x * 0.013 - t * 0.04)
                .collect::<Vec<_>>();
            let velocity = conditional
                .iter()
                .zip(unconditional.iter())
                .map(|(&c, &u)| {
                    if cfg < 1e-5 {
                        c
                    } else {
                        cfm::classifier_free_guidance(c, u, cfg)
                    }
                })
                .collect::<Vec<_>>();
            cfm::euler_step(&mut samples, &velocity, times[1] - times[0]).unwrap();
        }
        samples[..6].copy_from_slice(&condition[..6]);
        for (i, (&a, &b)) in samples.iter().zip(&target).enumerate() {
            assert!(
                (a - b).abs() <= 2e-6,
                "case={cases}, element={i}, {a} vs {b}"
            );
        }
        cases += 1;
    }
    assert_eq!(cases, 4);
}

struct FixturePinyin;
impl text_frontend::PinyinProvider for FixturePinyin {
    fn syllables(&self, word: &str) -> Result<Vec<String>> {
        if word == "你好" {
            return Ok(vec!["ni2".to_owned(), "hao3".to_owned()]);
        }
        word.chars()
            .map(|ch| {
                let value = match ch {
                    '你' => "ni3",
                    '好' => "hao3",
                    '世' => "shi4",
                    '界' => "jie4",
                    '测' => "ce4",
                    '试' => "shi4",
                    '中' => "zhong1",
                    '国' => "guo2",
                    _ => {
                        return Err(F5Error::InvalidConfiguration(
                            "Pinyin fixture unexpected Han",
                        ));
                    }
                };
                Ok(value.to_owned())
            })
            .collect()
    }
}

#[test]
fn five_hundred_original_python_pinyin_segment_cases_match_token_ids() {
    let vocab_symbols =
        " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"é，。";
    let vocab = vocab::Vocabulary::from_lines(
        vocab_symbols.chars().map(|c| c.to_string()).chain(
            [
                "ni2", "ni3", "hao3", "shi4", "jie4", "ce4", "zhong1", "guo2",
            ]
            .into_iter()
            .map(str::to_owned),
        ),
    )
    .unwrap();
    let tokenizer = text_frontend::F5TextFrontend::new(&vocab, FixturePinyin);
    let mut count = 0;
    for line in include_str!("fixtures/pinyin_segments_python_expected.tsv").lines() {
        let (left, right) = line.split_once('\t').unwrap();
        let segments = left.split('|').map(from_hex).collect::<Vec<String>>();
        let refs = segments.iter().map(String::as_str).collect::<Vec<_>>();
        let tokens = right.split('|').map(from_hex).collect::<Vec<String>>();
        let expected = vocab.encode_tokens(tokens.iter().map(String::as_str));
        let actual = tokenizer.encode_segments(&refs).unwrap();
        assert_eq!(
            actual, expected,
            "case {count}, segments {segments:?}, python tokens {tokens:?}"
        );
        count += 1;
    }
    assert_eq!(count, 505);
}
