use fish_speech_core::firefly_cpu::{
    AudioTensor, Conv1dKernel, ConvNextBlock, ResBlock1, TransConv1dKernel,
};

fn values(n: usize, seed: usize, weight: bool) -> Vec<f32> {
    (0..n)
        .map(|i| {
            if weight {
                ((i * 13 + seed) % 37) as f32 / 40.0 - 0.45
            } else {
                ((i * 19 + seed) % 29) as f32 / 11.0 - 1.2
            }
        })
        .collect()
}

fn bias_values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + seed) % 13) as f32 / 30.0 - 0.2)
        .collect()
}

fn input(batch: usize, channels: usize, frames: usize) -> AudioTensor {
    AudioTensor::new(
        batch,
        channels,
        frames,
        values(batch * channels * frames, 3, false),
    )
    .unwrap()
}

fn gold(label: &str) -> AudioTensor {
    let txt = include_str!("../fixtures/generate_firefly_golden.txt");
    let line = txt
        .lines()
        .find(|line| line.split_whitespace().next() == Some(label))
        .expect("missing golden fixture");
    let mut parts = line.split_whitespace();
    assert_eq!(parts.next(), Some(label));
    let dims = parts
        .next()
        .unwrap()
        .split(',')
        .map(|x| x.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let numbers = parts.map(|x| x.parse::<f32>().unwrap()).collect::<Vec<_>>();
    AudioTensor::new(dims[0], dims[1], dims[2], numbers).unwrap()
}

fn assert_gold(label: &str, actual: AudioTensor, atol: f32, rtol: f32) {
    let expected = gold(label);
    assert_eq!(
        (actual.batch, actual.channels, actual.frames),
        (expected.batch, expected.channels, expected.frames),
        "shape mismatch {label}"
    );
    for (i, (&got, &want)) in actual.data.iter().zip(&expected.data).enumerate() {
        assert!(
            got.is_finite() && (got - want).abs() <= atol + rtol * want.abs(),
            "{label} idx {i}: expected {want}, got {got}, delta {}",
            (got - want).abs()
        );
    }
}

fn conv_kernel(
    inc: usize,
    outc: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    seed: usize,
) -> Conv1dKernel {
    Conv1dKernel {
        in_channels: inc,
        out_channels: outc,
        kernel_size: kernel,
        stride,
        dilation,
        groups,
        weights: values(outc * (inc / groups) * kernel, seed, true),
        bias: Some(bias_values(outc, seed + 1)),
    }
}

#[test]
fn conv_regular_vs_original_torch() {
    assert_gold(
        "conv_regular",
        conv_kernel(2, 3, 3, 2, 1, 1, 1)
            .forward(&input(1, 2, 7))
            .unwrap(),
        2e-6,
        1e-5,
    );
}
#[test]
fn conv_depthwise_vs_original_torch() {
    assert_gold(
        "conv_depthwise",
        conv_kernel(3, 3, 7, 1, 1, 3, 2)
            .forward(&input(1, 3, 5))
            .unwrap(),
        2e-6,
        1e-5,
    );
}
#[test]
fn conv_dilated_vs_original_torch() {
    assert_gold(
        "conv_dilated",
        conv_kernel(2, 2, 3, 1, 3, 1, 3)
            .forward(&input(1, 2, 6))
            .unwrap(),
        2e-6,
        1e-5,
    );
}
#[test]
fn conv_short_vs_original_torch() {
    assert_gold(
        "conv_short",
        conv_kernel(1, 1, 7, 1, 1, 1, 4)
            .forward(&input(2, 1, 1))
            .unwrap(),
        2e-6,
        1e-5,
    );
}
#[test]
fn conv_grouped_vs_original_torch() {
    assert_gold(
        "conv_grouped",
        conv_kernel(4, 6, 4, 2, 1, 2, 5)
            .forward(&input(2, 4, 9))
            .unwrap(),
        2e-6,
        1e-5,
    );
}

fn transconv_kernel(
    inc: usize,
    outc: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    seed: usize,
) -> TransConv1dKernel {
    TransConv1dKernel {
        in_channels: inc,
        out_channels: outc,
        kernel_size: kernel,
        stride,
        dilation,
        weights: values(inc * outc * kernel, seed, true),
        bias: Some(bias_values(outc, seed + 1)),
    }
}
#[test]
fn transconv_regular_vs_original_torch() {
    assert_gold(
        "transconv_regular",
        transconv_kernel(3, 2, 4, 2, 1, 6)
            .forward(&input(1, 3, 5))
            .unwrap(),
        2e-6,
        1e-5,
    );
}
#[test]
fn transconv_upsample_vs_original_torch() {
    assert_gold(
        "transconv_upsample",
        transconv_kernel(1, 1, 16, 8, 1, 7)
            .forward(&input(2, 1, 3))
            .unwrap(),
        2e-6,
        1e-5,
    );
}
#[test]
fn transconv_dilated_vs_original_torch() {
    assert_gold(
        "transconv_dilated",
        transconv_kernel(2, 3, 5, 2, 2, 8)
            .forward(&input(1, 2, 4))
            .unwrap(),
        2e-6,
        1e-5,
    );
}

fn convnext(channels: usize, seed: usize) -> ConvNextBlock {
    let hidden = channels * 2;
    ConvNextBlock {
        depthwise: conv_kernel(channels, channels, 7, 1, 1, channels, seed),
        norm_weight: values(channels, seed + 1, true),
        norm_bias: bias_values(channels, seed + 2),
        pointwise1_weight: values(hidden * channels, seed + 3, true),
        pointwise1_bias: bias_values(hidden, seed + 4),
        pointwise2_weight: values(channels * hidden, seed + 5, true),
        pointwise2_bias: bias_values(channels, seed + 6),
        gamma: Some(values(channels, seed + 7, true)),
        epsilon: 1e-6,
    }
}
#[test]
fn convnext_three_channels_vs_original_torch() {
    assert_gold(
        "convnext_c3_t4",
        convnext(3, 10).forward(&input(1, 3, 4)).unwrap(),
        3e-6,
        1e-5,
    );
}
#[test]
fn convnext_short_sequence_vs_original_torch() {
    assert_gold(
        "convnext_c2_t1",
        convnext(2, 11).forward(&input(1, 2, 1)).unwrap(),
        3e-6,
        1e-5,
    );
}
#[test]
fn hifigan_resblock_vs_original_torch() {
    let pairs = [1, 3, 5]
        .into_iter()
        .enumerate()
        .map(|(i, dilation)| {
            (
                conv_kernel(2, 2, 3, 1, dilation, 1, 12 + i * 2),
                conv_kernel(2, 2, 3, 1, dilation, 1, 13 + i * 2),
            )
        })
        .collect();
    assert_gold(
        "resblock_c2_t6",
        ResBlock1 { pairs }.forward(&input(1, 2, 6)).unwrap(),
        4e-6,
        2e-5,
    );
}

#[test]
fn complete_toy_hifigan_decoder_vs_original_torch() {
    use fish_speech_core::firefly_cpu::HifiGanDecoder;
    let pre = conv_kernel(3, 8, 7, 1, 1, 1, 20);
    let upsample = vec![
        transconv_kernel(8, 4, 4, 2, 1, 21),
        transconv_kernel(4, 2, 4, 2, 1, 22),
    ];
    let mut seed = 23;
    let mut residuals = Vec::new();
    for channels in [4, 2] {
        let mut branches = Vec::new();
        for kernel in [3, 7, 11] {
            let mut pairs = Vec::new();
            for (i, dilation) in [1, 3, 5].into_iter().enumerate() {
                let first = conv_kernel(channels, channels, kernel, 1, dilation, 1, seed + i);
                let second = conv_kernel(channels, channels, kernel, 1, dilation, 1, seed + 3 + i);
                pairs.push((first, second));
            }
            branches.push(ResBlock1 { pairs });
            seed += 6;
        }
        residuals.push(branches);
    }
    assert_eq!(seed, 59);
    let post = conv_kernel(2, 1, 7, 1, 1, 1, seed);
    let model = HifiGanDecoder {
        pre,
        upsample,
        residuals,
        post,
    };
    assert_gold(
        "hifigan_toy_full",
        model.forward(&input(1, 3, 3)).unwrap(),
        5e-6,
        2e-5,
    );
}
