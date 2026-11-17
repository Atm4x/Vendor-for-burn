use fish_speech_core::spectrogram_cpu::FishLogMel;
use std::f64::consts::PI;

fn waveform(count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let i = i as f64;
            let idx = i as usize;
            (0.37 * (2.0 * PI * 440.0 * i / 44100.0).sin()
                + 0.21 * (2.0 * PI * 1300.0 * i / 44100.0).cos()
                + 0.03 * (2.0 * PI * 9000.0 * i / 44100.0).sin()
                + 0.02 * (((idx * 17 % 31) as f64) / 31.0 - 0.5)) as f32
        })
        .collect()
}

fn golden(length: usize) -> Vec<f32> {
    let source = include_str!("../fixtures/generate_spectrogram_golden.txt");
    let label = length.to_string();
    let line = source
        .lines()
        .find(|l| l.split_whitespace().next() == Some(label.as_str()))
        .expect("missing reference");
    let mut parts = line.split_whitespace();
    assert_eq!(parts.next(), Some(label.as_str()));
    assert_eq!(
        parts.next().unwrap().parse::<usize>().unwrap(),
        length / 512
    );
    parts.map(|x| x.parse::<f32>().unwrap()).collect()
}

#[test]
fn original_fish_log_mel_cpu_769_samples() {
    check(769);
}
#[test]
fn original_fish_log_mel_cpu_1024_samples() {
    check(1024);
}
#[test]
fn original_fish_log_mel_cpu_3073_samples() {
    check(3073);
}
#[test]
fn original_fish_log_mel_cpu_4096_samples() {
    check(4096);
}

fn check(length: usize) {
    let out = FishLogMel::new().transform_mono(&waveform(length)).unwrap();
    let expected = golden(length);
    assert_eq!(out.batch, 1);
    assert_eq!(out.channels, 160);
    assert_eq!(out.frames, length / 512);
    assert_eq!(out.data.len(), expected.len());
    let mut max_err: f32 = 0.0;
    let mut max_index = 0;
    for (i, (&actual, &want)) in out.data.iter().zip(&expected).enumerate() {
        let err = (actual - want).abs();
        if err > max_err {
            max_err = err;
            max_index = i;
        }
    }
    assert!(
        max_err < 0.001,
        "mel length {length} maximum error {max_err} at {max_index}, actual {} vs original {}",
        out.data[max_index],
        expected[max_index]
    );
}
