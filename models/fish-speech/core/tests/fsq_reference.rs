use fish_speech_core::fsq_cpu::{FsqCodebook, FsqGroupProjection, GroupedResidualFsqDecoder};

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
    let source = include_str!("../fixtures/generate_fsq_golden.txt");
    let line = source
        .lines()
        .find(|l| l.split_whitespace().next() == Some(name))
        .unwrap();
    let mut pieces = line.split_whitespace();
    assert_eq!(pieces.next(), Some(name));
    let dims = pieces
        .next()
        .unwrap()
        .split(',')
        .map(|p| p.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let (batch, groups, q, frames, width, seed) =
        (dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
    let codes = pieces
        .next()
        .unwrap()
        .split(',')
        .map(|x| x.parse::<i32>().unwrap())
        .collect::<Vec<_>>();
    let expected = pieces
        .map(|v| v.parse::<f32>().unwrap())
        .collect::<Vec<_>>();
    let model = GroupedResidualFsqDecoder {
        codebook: FsqCodebook::new(&[8, 5, 5, 5]).unwrap(),
        num_quantizers: q,
        projections: (0..groups)
            .map(|g| FsqGroupProjection {
                weights: weight(width * 4, seed + g),
                bias: bias(width, seed + g + 1),
            })
            .collect(),
    };
    let output = model.decode(&codes, batch, frames).unwrap();
    assert_eq!(
        (output.batch, output.channels, output.frames),
        (batch, groups * width, frames)
    );
    assert_eq!(output.data.len(), expected.len());
    for (i, (&val, &exp)) in output.data.iter().zip(&expected).enumerate() {
        assert!(
            (val - exp).abs() < 2e-6 + 1e-5 * exp.abs(),
            "{name} mismatch at {i}: {val} vs {exp}"
        );
    }
}
#[test]
fn fsq_grouped_eight_groups_single_quantizer() {
    verify("fsq_8group_q1");
}
#[test]
fn fsq_grouped_two_groups_three_residual_quantizers() {
    verify("fsq_2group_q3");
}
