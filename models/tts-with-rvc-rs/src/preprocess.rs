const B: [f64; 6] = [
    0.9699606451838447,
    -4.849803225919223,
    9.699606451838447,
    -9.699606451838447,
    4.849803225919223,
    -0.9699606451838447,
];
const A: [f64; 6] = [
    1.0,
    -4.939001819168364,
    9.757863526739543,
    -9.639544849413458,
    4.761506797356209,
    -0.9408236532054606,
];
const ZI: [f64; 5] = [
    -0.9699604796995847,
    3.8798419288925783,
    -5.819762908173043,
    3.879841948472456,
    -0.9699604894923387,
];
const PAD: usize = 18;

pub fn highpass_48hz_zero_phase(input: &[f32]) -> Result<Vec<f32>, String> {
    if input.len() <= PAD || input.iter().any(|value| !value.is_finite()) {
        return Err("zero-phase 48-Hz high-pass requires at least 19 finite 16-kHz samples".into());
    }
    let mut extended = Vec::with_capacity(input.len() + PAD * 2);
    for i in (1..=PAD).rev() {
        extended.push(2.0 * input[0] as f64 - input[i] as f64);
    }
    extended.extend(input.iter().map(|&x| x as f64));
    let last = input.len() - 1;
    for i in 1..=PAD {
        extended.push(2.0 * input[last] as f64 - input[last - i] as f64);
    }
    let mut filtered = lfilter(&extended, extended[0]);
    filtered.reverse();
    let mut reversed = lfilter(&filtered, filtered[0]);
    reversed.reverse();
    let result: Vec<f32> = reversed[PAD..PAD + input.len()]
        .iter()
        .map(|&x| x as f32)
        .collect();
    if result.iter().any(|x| !x.is_finite()) {
        return Err("RVC high-pass filter generated non-finite samples".into());
    }
    Ok(result)
}

fn lfilter(input: &[f64], initial_value: f64) -> Vec<f64> {
    let mut state = ZI.map(|zi| zi * initial_value);
    let mut output = Vec::with_capacity(input.len());
    for &sample in input {
        let value = B[0] * sample + state[0];
        for i in 0..4 {
            state[i] = B[i + 1] * sample - A[i + 1] * value + state[i + 1];
        }
        state[4] = B[5] * sample - A[5] * value;
        output.push(value);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scipy_butterworth_filtfilt_reference() {
        let pcm: Vec<f32> = (0..96)
            .map(|i| ((i as f64 * 0.1).sin() + i as f64 * 0.0003) as f32)
            .collect();
        let expected: [f64; 12] = [
            0.26702303352574175,
            0.8726606149738061,
            1.0453240601730611,
            0.6176667539819587,
            -0.20919321338994135,
            -0.9848784224958691,
            -1.279715075137309,
            -0.9416670969210978,
            -0.18435794796836696,
            0.5472434685795518,
            0.8520304921654913,
            0.6219535399834036,
        ];
        let actual = highpass_48hz_zero_phase(&pcm).unwrap();
        assert_eq!(actual.len(), pcm.len());
        for (i, &want) in expected.iter().enumerate() {
            assert!(
                (actual[i * 8] as f64 - want).abs() < 2e-4,
                "scipy-filtfilt sample {}: {} vs {}",
                i * 8,
                actual[i * 8],
                want
            );
        }
    }

    #[test]
    fn rejects_nonfinite_pcm() {
        assert!(highpass_48hz_zero_phase(&[0.0, f32::NAN]).is_err());
    }
}
