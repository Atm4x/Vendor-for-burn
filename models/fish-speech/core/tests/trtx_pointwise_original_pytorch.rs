use fish_speech_core::firefly_cpu::{gelu, layer_norm_channels_first, silu, AudioTensor};

fn floats(line: &str, label: &str) -> Vec<f32> {
    let (prefix, values) = line.split_once(' ').expect("named fixture tensor");
    assert_eq!(prefix, label);
    values
        .split_whitespace()
        .map(|v| v.parse::<f32>().unwrap())
        .collect()
}

#[test]
fn original_pytorch_pointwise_equations_match_rust_reference() {
    let lines = include_str!("../fixtures/trtx_pointwise_original_pytorch.txt")
        .lines()
        .collect::<Vec<_>>();
    assert_eq!(lines.len() % 5, 0);
    assert_eq!(lines.len() / 5, 8);
    for case in lines.chunks_exact(5) {
        let header = case[0].split_whitespace().collect::<Vec<_>>();
        assert_eq!(header.len(), 5);
        let name = header[1];
        let n = header[2].parse::<usize>().unwrap();
        let c = header[3].parse::<usize>().unwrap();
        let t = header[4].parse::<usize>().unwrap();
        let x = floats(case[1], "input");
        let other = floats(case[2], "extra1");
        let third = floats(case[3], "extra2");
        let expected = floats(case[4], "output");
        assert_eq!(x.len(), n * c * t);
        let actual = match name {
            "gelu_erf" => x.iter().copied().map(gelu).collect(),
            "silu" => x.iter().copied().map(silu).collect(),
            "tanh" => x.iter().map(|&v| v.tanh()).collect(),
            "layer_scale" => (0..x.len()).map(|i| x[i] * other[(i / t) % c]).collect(),
            "layernorm_first" | "layernorm_last" => {
                layer_norm_channels_first(
                    &AudioTensor::new(n, c, t, x).unwrap(),
                    &other,
                    &third,
                    1e-6,
                )
                .unwrap()
                .data
            }
            "add" => x.iter().zip(&third).map(|(&a, &b)| a + b).collect(),
            "parallel_mean" => x
                .iter()
                .zip(&other)
                .zip(&third)
                .map(|((&a, &b), &c)| (a + b + c) / 3.0)
                .collect(),
            _ => panic!("unexpected fixture {name}"),
        };
        let worst = actual
            .iter()
            .zip(&expected)
            .map(|(&got, &want)| (got - want).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1.0e-5, "{name}: maximum error {worst}");
    }
}
