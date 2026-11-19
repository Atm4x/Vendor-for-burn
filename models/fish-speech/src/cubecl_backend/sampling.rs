use cubecl::prelude::*;
use cubecl::server::Handle;
use fish_speech_core::{
    error::{FishError, Result},
    sampling::SamplingParams,
};

#[cube(launch)]
fn greedy_token_kernel(
    logits: &[f32],
    history: &[i32],
    result: &mut [i32],
    #[comptime] offset: usize,
    #[comptime] vocabulary: usize,
    penalty: f32,
    eos_id: i32,
) {
    if ABSOLUTE_POS == 0 {
        let mut best = logits[offset] * 0.0 - 3.402823466e38f32;
        let mut chosen = i32::cast_from(ABSOLUTE_POS);
        for i in 0..vocabulary {
            let mut score = logits[offset + i];
            if score != score || score > 3.402823e38f32 || score < -3.402823e38f32 {
                result[0] = -1;
                result[1] = -1;
                terminate!();
            }
            for j in 0..history.len() {
                if history[j] == i as i32 {
                    if score < 0.0 {
                        score *= penalty;
                    } else {
                        score /= penalty;
                    }
                    break;
                }
            }
            if score > best {
                best = score;
                chosen = i as i32;
            }
        }
        result[0] = chosen;
        result[1] = if chosen == eos_id { 1 } else { 0 };
    }
}

pub(super) fn greedy_device_token(
    client: &Client,
    logits: &Handle,
    total: usize,
    offset: usize,
    vocabulary: usize,
    history: &[usize],
    params: SamplingParams,
    eos_id: u32,
    retained: &mut Vec<Handle>,
) -> Result<(u32, bool)> {
    params.validate()?;
    let end = offset
        .checked_add(vocabulary)
        .ok_or(FishError::InvalidShape("CubeCL sampling offset overflow"))?;
    if vocabulary == 0
        || end > total
        || total > i32::MAX as usize
        || history.iter().any(|&id| id >= vocabulary)
    {
        return Err(FishError::InvalidShape(
            "CubeCL sampling vocabulary or history invalid",
        ));
    }
    let eos = i32::try_from(eos_id)
        .map_err(|_| FishError::InvalidShape("CubeCL EOS exceeds i32 range"))?;
    let mut history = history
        .iter()
        .copied()
        .map(|id| i32::try_from(id).expect("checked vocabulary <= i32"))
        .collect::<Vec<_>>();
    if history.is_empty() {
        history.push(-1);
    }
    let history_handle = client.create_from_slice(bytemuck::cast_slice(&history));
    let result = client.empty(2 * size_of::<i32>());
    greedy_token_kernel::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(logits.clone(), total) },
        unsafe { BufferArg::from_raw_parts(history_handle.clone(), history.len()) },
        unsafe { BufferArg::from_raw_parts(result.clone(), 2) },
        offset,
        vocabulary,
        params.repetition_penalty,
        eos,
    );
    retained.push(history_handle);
    retained.push(result.clone());
    let bytes = client
        .read_one(result)
        .map_err(|_| FishError::InvalidTransition("CubeCL GPU sampled token readback failed"))?;
    if bytes.len() != 8 {
        return Err(FishError::InvalidShape(
            "CubeCL GPU sampler returned invalid token data",
        ));
    }
    let token = i32::from_le_bytes(bytes[0..4].try_into().expect("four bytes"));
    let flag = i32::from_le_bytes(bytes[4..8].try_into().expect("four bytes"));
    if token < 0 || token as usize >= vocabulary || flag != i32::from(token == eos) {
        return Err(FishError::InvalidSampling(
            "CubeCL GPU sampler returned invalid token or EOS",
        ));
    }
    Ok((token as u32, flag != 0))
}

#[cube(launch)]
fn top_p_token_kernel(
    logits: &[f32],
    history: &[i32],
    race_denominators: &[f32],
    scores: &mut [f32],
    weights: &mut [f32],
    result: &mut [i32],
    #[comptime] offset: usize,
    #[comptime] vocabulary: usize,
    top_p: f32,
    temperature: f32,
    penalty: f32,
    eos_id: i32,
) {
    if ABSOLUTE_POS == 0 {
        let mut maximum = logits[offset] * 0.0f32 - 3.402823466e38f32;
        for i in 0..vocabulary {
            let mut score = logits[offset + i];
            if score != score || score > 3.402823e38f32 || score < -3.402823e38f32 {
                result[0] = -1;
                result[1] = -1;
                terminate!();
            }
            for j in 0..history.len() {
                if history[j] == i as i32 {
                    if score < 0.0 {
                        score *= penalty;
                    } else {
                        score /= penalty;
                    }
                    break;
                }
            }
            if score != score || score > 3.402823e38f32 || score < -3.402823e38f32 {
                result[0] = -1;
                result[1] = -1;
                terminate!();
            }
            scores[i] = score;
            if score > maximum {
                maximum = score;
            }
        }
        let mut total = 0.0f32;
        for i in 0..vocabulary {
            let weight = (scores[i] - maximum).exp();
            weights[i] = weight;
            total += weight;
        }
        if total <= 0.0 || total != total {
            result[0] = -1;
            result[1] = -1;
            terminate!();
        }
        let mut threshold = u32::cast_from(ABSOLUTE_POS);
        let mut cutoff_id = i32::cast_from(ABSOLUTE_POS);
        if top_p < 1.0 {
            let target = total * top_p;
            let mut low = 0u64;
            let mut high = 4294967295u64;
            for _pass in 0..32 {
                let midpoint = (low + high + 1u64) / 2u64;
                let mut mass = 0.0f32;
                for i in 0..vocabulary {
                    let bits = u32::reinterpret(scores[i]);
                    let key = if (bits & 2147483648u32) != 0u32 {
                        !bits
                    } else {
                        bits ^ 2147483648u32
                    };
                    if u64::cast_from(key) >= midpoint {
                        mass += weights[i];
                    }
                }
                if mass > target {
                    low = midpoint;
                } else {
                    high = midpoint - 1u64;
                }
            }
            threshold = u32::cast_from(low);
            let mut above_mass = 0.0f32;
            let mut equal_count = 0i32;
            let mut equal_weight = weights[0] * 0.0f32;
            for i in 0..vocabulary {
                let bits = u32::reinterpret(scores[i]);
                let key = if (bits & 2147483648u32) != 0u32 {
                    !bits
                } else {
                    bits ^ 2147483648u32
                };
                if key > threshold {
                    above_mass += weights[i];
                } else if key == threshold {
                    equal_count += 1;
                    equal_weight = weights[i];
                }
            }
            let remaining = if target > above_mass {
                target - above_mass
            } else {
                0.0f32
            };
            let mut allowed = if equal_weight > 0.0f32 {
                i32::cast_from(remaining / equal_weight)
            } else {
                0i32
            };
            if above_mass == 0.0f32 && allowed == 0i32 {
                allowed = 1;
            }
            if allowed > equal_count {
                allowed = equal_count;
            }
            if allowed > 0i32 {
                let mut seen = 0i32;
                for i in 0..vocabulary {
                    let bits = u32::reinterpret(scores[i]);
                    let key = if (bits & 2147483648u32) != 0u32 {
                        !bits
                    } else {
                        bits ^ 2147483648u32
                    };
                    if key == threshold {
                        seen += 1;
                        if seen == allowed {
                            cutoff_id = i as i32;
                            break;
                        }
                    }
                }
            } else {
                cutoff_id = -1;
            }
        }
        let mut best = scores[0] * 0.0f32 - 1.0f32;
        let mut chosen = i32::cast_from(ABSOLUTE_POS) - 1i32;
        for i in 0..vocabulary {
            let bits = u32::reinterpret(scores[i]);
            let key = if (bits & 2147483648u32) != 0u32 {
                !bits
            } else {
                bits ^ 2147483648u32
            };
            if top_p == 1.0f32 || key > threshold || (key == threshold && i as i32 <= cutoff_id) {
                let denominator = race_denominators[i];
                let race = ((scores[i] - maximum) / temperature).exp() / denominator;
                if denominator > 0.0f32 && race > best {
                    best = race;
                    chosen = i as i32;
                }
            }
        }
        result[0] = chosen;
        result[1] = if chosen == eos_id { 1 } else { 0 };
    }
}

pub(super) fn top_p_device_token(
    client: &Client,
    logits: &Handle,
    total: usize,
    offset: usize,
    vocabulary: usize,
    history: &[usize],
    params: SamplingParams,
    eos_id: u32,
    rng: &mut fish_speech_core::sampling::SessionRng,
    retained: &mut Vec<Handle>,
) -> Result<(u32, bool)> {
    params.validate()?;
    let end = offset
        .checked_add(vocabulary)
        .ok_or(FishError::InvalidShape("CubeCL sampling offset overflow"))?;
    if vocabulary == 0
        || end > total
        || total > i32::MAX as usize
        || history.iter().any(|&id| id >= vocabulary)
    {
        return Err(FishError::InvalidShape(
            "CubeCL sampling vocabulary or history invalid",
        ));
    }
    let eos = i32::try_from(eos_id)
        .map_err(|_| FishError::InvalidShape("CubeCL EOS exceeds i32 range"))?;
    let mut history = history
        .iter()
        .copied()
        .map(|id| id as i32)
        .collect::<Vec<_>>();
    if history.is_empty() {
        history.push(-1);
    }
    // Draw one independent exponential per original vocabulary index, even
    // when top-p later excludes the index; this preserves SessionRng ordering.
    let denominators = (0..vocabulary)
        .map(|_| (-rng.uniform_open_01().ln()) as f32)
        .collect::<Vec<_>>();
    let history_handle = client.create_from_slice(bytemuck::cast_slice(&history));
    let draws = client.create_from_slice(bytemuck::cast_slice(&denominators));
    let scores = client.empty(vocabulary * size_of::<f32>());
    let weights = client.empty(vocabulary * size_of::<f32>());
    let result = client.empty(2 * size_of::<i32>());
    top_p_token_kernel::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(logits.clone(), total) },
        unsafe { BufferArg::from_raw_parts(history_handle.clone(), history.len()) },
        unsafe { BufferArg::from_raw_parts(draws.clone(), vocabulary) },
        unsafe { BufferArg::from_raw_parts(scores.clone(), vocabulary) },
        unsafe { BufferArg::from_raw_parts(weights.clone(), vocabulary) },
        unsafe { BufferArg::from_raw_parts(result.clone(), 2) },
        offset,
        vocabulary,
        params.top_p,
        params.temperature,
        params.repetition_penalty,
        eos,
    );
    retained.extend([history_handle, draws, scores, weights, result.clone()]);
    let bytes = client.read_one(result).map_err(|_| {
        FishError::InvalidTransition("CubeCL GPU top-p sampled token readback failed")
    })?;
    if bytes.len() != 8 {
        return Err(FishError::InvalidShape(
            "CubeCL GPU top-p sampler invalid token data",
        ));
    }
    let token = i32::from_le_bytes(bytes[0..4].try_into().expect("four bytes"));
    let flag = i32::from_le_bytes(bytes[4..8].try_into().expect("four bytes"));
    if token < 0 || token as usize >= vocabulary || flag != i32::from(token == eos) {
        return Err(FishError::InvalidSampling(
            "CubeCL GPU top-p sampler invalid token or EOS",
        ));
    }
    Ok((token as u32, flag != 0))
}

#[cfg(all(test, feature = "cubecl-cpu"))]
mod tests {
    use super::*;
    use cubecl::Device;
    use fish_speech_core::sampling::{greedy, logits_to_probs};

    #[test]
    fn cubecl_top_p_matches_host_for_eos_penalty_ties_and_offset() {
        let client = Device::Cpu(Default::default()).client();
        let cases = [
            (vec![2.0f32, 1.0, 0.0], vec![], 0.7f32, 1.0f32, 1.0f32),
            (vec![2.0, 1.0, 0.0], vec![0, 0], 0.6, 0.9, 1.5),
            (vec![0.0, 0.0, 0.0, 0.0], vec![], 0.5, 0.8, 1.0),
            (vec![-2.0, -0.5, -1.0, -1.5], vec![1, 3], 0.9, 1.1, 1.8),
            (vec![1.0, 4.0, -2.0, 0.5, -1.0], vec![1, 4], 1.0, 0.6, 1.2),
            (vec![0.0, 0.0, 0.0], vec![], 0.01, 1.5, 1.0),
        ];
        for (logits, history, top_p, temperature, repetition_penalty) in cases {
            let params = SamplingParams {
                top_p,
                temperature,
                repetition_penalty,
            };
            let padded = [vec![40.0f32, 30.0], logits.clone(), vec![20.0]].concat();
            let device_logits = client.create_from_slice(bytemuck::cast_slice(&padded));
            for seed in [1u64, 42u64] {
                let mut reference_rng = fish_speech_core::sampling::SessionRng::new(seed);
                let expected = reference_rng
                    .sample_exponential_race(&logits_to_probs(&logits, &history, params).unwrap())
                    .unwrap() as u32;
                let reference_next = reference_rng.uniform_open_01();
                let mut rng = fish_speech_core::sampling::SessionRng::new(seed);
                let mut retained = Vec::new();
                let actual = top_p_device_token(
                    &client,
                    &device_logits,
                    padded.len(),
                    2,
                    logits.len(),
                    &history,
                    params,
                    expected,
                    &mut rng,
                    &mut retained,
                )
                .unwrap();
                assert_eq!(
                    actual,
                    (expected, true),
                    "seed={seed}, logits={logits:?}, history={history:?}, top_p={top_p}"
                );
                assert_eq!(rng.uniform_open_01(), reference_next);
            }
        }
    }

    #[test]
    fn cubecl_top_p_matches_reference_across_generated_vocabularies() {
        let client = Device::Cpu(Default::default()).client();
        for vocabulary in [4usize, 16, 64, 257] {
            let values: Vec<f32> = (0..vocabulary)
                .map(|i| ((i * 97 % 181) as f32 - 90.0) / 17.0)
                .collect();
            let logits = client.create_from_slice(bytemuck::cast_slice(&values));
            let history = [1usize, 3, 3];
            for (top_p, seed) in [(0.15, 1), (0.37, 123), (0.7, 19), (0.9, 42), (1.0, 99)] {
                let params = SamplingParams {
                    top_p,
                    temperature: 0.83,
                    repetition_penalty: 1.3,
                };
                let mut reference_rng = fish_speech_core::sampling::SessionRng::new(seed);
                let expected = reference_rng
                    .sample_exponential_race(&logits_to_probs(&values, &history, params).unwrap())
                    .unwrap() as u32;
                let mut rng = fish_speech_core::sampling::SessionRng::new(seed);
                let mut retained = Vec::new();
                let actual = top_p_device_token(
                    &client,
                    &logits,
                    vocabulary,
                    0,
                    vocabulary,
                    &history,
                    params,
                    expected,
                    &mut rng,
                    &mut retained,
                )
                .unwrap()
                .0;
                assert_eq!(
                    actual, expected,
                    "vocab={vocabulary}, p={top_p}, seed={seed}"
                );
            }
        }
    }

    #[test]
    fn cubecl_top_p_rejects_invalid_logits_without_silent_token_selection() {
        let client = Device::Cpu(Default::default()).client();
        let values = [f32::NAN, 0.0, 1.0];
        let logits = client.create_from_slice(bytemuck::cast_slice(&values));
        let mut rng = fish_speech_core::sampling::SessionRng::new(1);
        let mut retained = Vec::new();
        assert!(
            top_p_device_token(
                &client,
                &logits,
                values.len(),
                0,
                values.len(),
                &[],
                SamplingParams::default(),
                1,
                &mut rng,
                &mut retained,
            )
            .is_err()
        );
    }

    #[test]
    fn cubecl_greedy_sampler_matches_host_with_eos_repetition_and_offsets() {
        let client = Device::Cpu(Default::default()).client();
        let samples = [
            (vec![0.5f32, 2.0, 2.0, -1.0], vec![], 1u32),
            (vec![0.5f32, 2.0, 2.0, -1.0], vec![1], 2u32),
            (vec![-1.0f32, -0.5, -0.25, -2.0], vec![2], 1u32),
            (vec![-1.0f32, -0.5, -0.25, -2.0], vec![0, 0, 2], 3u32),
        ];
        let params = SamplingParams {
            temperature: 0.75,
            top_p: 1.0,
            repetition_penalty: 1.5,
        };
        for (logits, history, eos) in samples {
            let mut padded = vec![10.0f32, 20.0];
            padded.extend_from_slice(&logits);
            padded.push(30.0);
            let logits_handle = client.create_from_slice(bytemuck::cast_slice(&padded));
            let mut retained = Vec::new();
            let gpu = greedy_device_token(
                &client,
                &logits_handle,
                padded.len(),
                2,
                logits.len(),
                &history,
                params,
                eos,
                &mut retained,
            )
            .unwrap();
            let cpu = greedy(&logits_to_probs(&logits, &history, params).unwrap()).unwrap() as u32;
            assert_eq!(gpu, (cpu, cpu == eos));
        }
    }
}
