use fish_speech_core::firefly_cpu::AudioTensor;
use fish_speech_core::fsq_cpu::{FsqCodebook, FsqGroupProjection, GroupedResidualFsqEncoder};

fn weight(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 13 + seed) % 37) as f32 / 40.0 - 0.45)
        .collect()
}
fn bias(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + seed) % 13) as f32 / 30.0 - 0.2)
        .collect()
}
fn verify(name: &str) {
    let line = include_str!("../fixtures/generate_fsq_encode_golden.txt")
        .lines()
        .find(|line| line.split_whitespace().next() == Some(name))
        .unwrap();
    let mut fields = line.split_whitespace();
    assert_eq!(fields.next(), Some(name));
    let spec = fields
        .next()
        .unwrap()
        .split(',')
        .map(|number| number.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let (batch, groups, q, frames, width, seed) =
        (spec[0], spec[1], spec[2], spec[3], spec[4], spec[5]);
    let expected = fields
        .next()
        .unwrap()
        .split(',')
        .map(|number| number.parse::<i32>().unwrap())
        .collect::<Vec<_>>();
    let data = (0..batch * groups * width * frames)
        .map(|i| ((i * 11 + seed) % 103) as f32 / 22.0 - 2.2)
        .collect();
    let input = AudioTensor::new(batch, groups * width, frames, data).unwrap();
    let encoder = GroupedResidualFsqEncoder {
        codebook: FsqCodebook::new(&[8, 5, 5, 5]).unwrap(),
        num_quantizers: q,
        projections: (0..groups)
            .map(|g| FsqGroupProjection {
                weights: weight(width * 4, seed + g),
                bias: bias(4, seed + g + 1),
            })
            .collect(),
    };
    assert_eq!(encoder.encode(&input).unwrap(), expected);
}

#[test]
fn fsq_encode_eight_groups_one_quantizer() {
    verify("q1_g8");
}
#[test]
fn fsq_encode_two_groups_three_quantizers() {
    verify("q3_g2");
}
#[test]
fn fsq_encode_one_group_two_quantizers() {
    verify("q2_g1");
}

#[test]
fn fsq_rejects_nonfinite_input_and_wrong_width() {
    let encoder = GroupedResidualFsqEncoder {
        codebook: FsqCodebook::new(&[8, 5, 5, 5]).unwrap(),
        num_quantizers: 1,
        projections: vec![FsqGroupProjection {
            weights: vec![0.5; 4 * 2],
            bias: vec![0.0; 4],
        }],
    };
    assert!(encoder
        .encode(&AudioTensor::new(1, 1, 3, vec![1.0; 3]).unwrap())
        .is_err());
    assert!(encoder
        .encode(&AudioTensor::new(1, 2, 1, vec![f32::NAN, 1.0]).unwrap())
        .is_err());
}
