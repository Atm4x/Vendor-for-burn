use gigaam_rs::audio::{SpeechRegion, VadGrouping, speech_chunks_grouped_vad};
use serde::Deserialize;

#[derive(Deserialize)]
struct VadCase {
    name: String,
    regions: Vec<[usize; 2]>,
    expected: Vec<[usize; 2]>,
}

#[test]
fn vad_chunk_boundaries_match_upstream_python() {
    let fixture: Vec<VadCase> =
        serde_json::from_str(include_str!("fixtures/upstream_vad_grouping.json")).unwrap();
    let audio = vec![0.0_f32; 180 * 16_000];
    for case in fixture {
        let regions: Vec<SpeechRegion> = case
            .regions
            .into_iter()
            .map(|[start_sample, end_sample]| SpeechRegion {
                start_sample,
                end_sample,
            })
            .collect();
        let result = speech_chunks_grouped_vad(&audio, &regions, VadGrouping::default())
            .unwrap_or_else(|error| panic!("{}: {}", case.name, error));
        assert_eq!(
            result.len(),
            case.expected.len(),
            "{}: chunk count",
            case.name
        );
        for (index, (chunk, expected)) in result.iter().zip(case.expected).enumerate() {
            let left = chunk.start_sample;
            let right = left + chunk.samples.len();
            assert!(
                left.abs_diff(expected[0]) <= 2 && right.abs_diff(expected[1]) <= 2,
                "{} segment {}: Rust [{}, {}], Python {:?}",
                case.name,
                index,
                left,
                right,
                expected
            );
        }
    }
}
