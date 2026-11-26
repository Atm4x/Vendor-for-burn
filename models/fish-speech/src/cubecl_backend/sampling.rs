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

pub(super) fn enqueue_greedy(
    client: &Client,
    logits: &Handle,
    total: usize,
    offset: usize,
    vocabulary: usize,
    history: &[usize],
    params: SamplingParams,
    eos_id: u32,
    retained: &mut Vec<Handle>,
) -> Result<Handle> {
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
    Ok(result)
}

pub(super) fn greedy_device_token_with_handle(
    client: &Client,
    logits: &Handle,
    total: usize,
    offset: usize,
    vocabulary: usize,
    history: &[usize],
    params: SamplingParams,
    eos_id: u32,
    retained: &mut Vec<Handle>,
) -> Result<(u32, bool, Handle)> {
    let handle = enqueue_greedy(
        client, logits, total, offset, vocabulary, history, params, eos_id, retained,
    )?;
    let bytes = client
        .read_one(handle.clone())
        .map_err(|_| FishError::InvalidTransition("CubeCL GPU sampled token readback failed"))?;
    let (token, eos) = decode_sample(&bytes, vocabulary, eos_id)?;
    Ok((token, eos, handle))
}

#[cfg(all(test, feature = "cubecl-cpu"))]
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
    greedy_device_token_with_handle(
        client, logits, total, offset, vocabulary, history, params, eos_id, retained,
    )
    .map(|(token, eos, _)| (token, eos))
}

pub(super) fn decode_sample(bytes: &[u8], vocabulary: usize, eos_id: u32) -> Result<(u32, bool)> {
    if bytes.len() != 8 {
        return Err(FishError::InvalidShape(
            "CubeCL GPU sampler invalid token data",
        ));
    }
    let token = i32::from_le_bytes(bytes[..4].try_into().expect("four bytes"));
    let flag = i32::from_le_bytes(bytes[4..].try_into().expect("four bytes"));
    if token < 0 || token as usize >= vocabulary || flag != i32::from(token as u32 == eos_id) {
        return Err(FishError::InvalidSampling(
            "CubeCL GPU sampler invalid token or EOS",
        ));
    }
    Ok((token as u32, flag != 0))
}

const SAMPLER_THREADS: u32 = 128;

#[cube]
fn score_key(score: f32) -> u32 {
    let bits = u32::reinterpret(score);
    if (bits & 0x8000_0000u32) != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000u32
    }
}

#[cube]
fn reduce_mass(values: &mut Shared<[f32]>, lane: usize, partial: f32) -> f32 {
    values[lane] = partial;
    sync_cube();
    let mut width = CUBE_DIM_X as usize / 2;
    while width > 0 {
        if lane < width {
            values[lane] += values[lane + width];
        }
        sync_cube();
        width /= 2;
    }
    let sum = values[0];
    sync_cube();
    sum
}

#[cube(launch)]
fn semantic_tiles_kernel(
    logits: &[f32],
    history: &[i32],
    scores: &mut [f32],
    summaries: &mut [f32],
    #[comptime] offset: usize,
    #[comptime] vocabulary: usize,
    penalty: f32,
) {
    let lane = UNIT_POS_X as usize;
    let tile = CUBE_POS_X as usize;
    let i = ABSOLUTE_POS;
    let mut score = f32::cast_from(CUBE_POS_X) - 3.402823466e38f32;
    let mut invalid = 0.0f32;
    if i < vocabulary {
        score = logits[offset + i];
        for j in 0..history.len() {
            if history[j] == i as i32 {
                score = if score < 0.0 {
                    score * penalty
                } else {
                    score / penalty
                };
                break;
            }
        }
        if score != score || score > 3.402823e38f32 || score < -3.402823e38f32 {
            invalid = 1.0;
        }
        scores[i] = score;
    }
    let mut shared = Shared::<[f32]>::new_slice(128usize);
    let invalid = reduce_mass(&mut shared, lane, invalid);
    shared[lane] = score;
    sync_cube();
    let mut width = 64usize;
    while width > 0 {
        if lane < width {
            if shared[lane + width] > shared[lane] {
                shared[lane] = shared[lane + width];
            }
        }
        sync_cube();
        width /= 2;
    }
    let maximum = shared[0];
    sync_cube();
    let mass = if i < vocabulary {
        (score - maximum).exp()
    } else {
        0.0f32
    };
    let total = reduce_mass(&mut shared, lane, mass);
    if lane == 0 {
        summaries[tile * 2] = maximum;
        summaries[tile * 2 + 1] = if invalid > 0.0 { -1.0f32 } else { total };
    }
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
    #[comptime] tiled: bool,
    top_p: f32,
    temperature: f32,
    penalty: f32,
    eos_id: i32,
) {
    let lane = UNIT_POS_X as usize;
    let mut values = Shared::<[f32]>::new_slice(128usize);
    let mut ids = Shared::<[i32]>::new_slice(128usize);
    let mut local_max = f32::cast_from(CUBE_POS_X) - 3.402823466e38f32;
    let mut invalid = 0.0f32;
    let mut i = lane;
    let tiles = vocabulary.div_ceil(128);
    if tiled {
        while i < tiles {
            let tile_max = weights[i * 2];
            if tile_max > local_max {
                local_max = tile_max;
            }
            if weights[i * 2 + 1] < 0.0 {
                invalid = 1.0;
            }
            i += CUBE_DIM_X as usize;
        }
    } else {
        while i < vocabulary {
            let mut score = logits[offset + i];
            for j in 0..history.len() {
                if history[j] == i as i32 {
                    score = if score < 0.0 {
                        score * penalty
                    } else {
                        score / penalty
                    };
                    break;
                }
            }
            if score != score || score > 3.402823e38f32 || score < -3.402823e38f32 {
                invalid = 1.0;
            }
            scores[i] = score;
            if score > local_max {
                local_max = score;
            }
            i += CUBE_DIM_X as usize;
        }
    }
    let invalid = reduce_mass(&mut values, lane, invalid);
    if invalid > 0.0 {
        if lane == 0 {
            result[0] = -1;
            result[1] = -1;
        }
        terminate!();
    }
    values[lane] = local_max;
    sync_cube();
    let mut width = CUBE_DIM_X as usize / 2;
    while width > 0 {
        if lane < width {
            let other = values[lane + width];
            if other > values[lane] {
                values[lane] = other;
            }
        }
        sync_cube();
        width /= 2;
    }
    let maximum = values[0];
    sync_cube();
    let mut keys = Shared::<[u32]>::new_slice(2048usize);
    let mut masses = Shared::<[f32]>::new_slice(2048usize);
    let floor = score_key(maximum - 8.0f32);
    let mut local_total = 0.0f32;
    let mut tail_total = 0.0f32;
    let mut entries = lane - lane;
    i = lane;
    if tiled {
        while i < tiles {
            local_total += weights[i * 2 + 1] * (weights[i * 2] - maximum).exp();
            i += CUBE_DIM_X as usize;
        }
        for tile in 0..tiles {
            if top_p < 1.0 && score_key(weights[tile * 2]) >= floor {
                let id = tile * 128 + lane;
                if id < vocabulary {
                    let key = score_key(scores[id]);
                    if key >= floor {
                        let weight = (scores[id] - maximum).exp();
                        if entries < 16 {
                            let slot = entries * 128 + lane;
                            keys[slot] = key;
                            masses[slot] = weight;
                        }
                        entries += 1;
                        tail_total += weight;
                    }
                }
            }
        }
    } else {
        while i < vocabulary {
            let weight = (scores[i] - maximum).exp();
            weights[i] = weight;
            local_total += weight;
            let key = score_key(scores[i]);
            if top_p < 1.0 && key >= floor {
                if entries < 16 {
                    let slot = entries * CUBE_DIM_X as usize + lane;
                    keys[slot] = key;
                    masses[slot] = weight;
                }
                entries += 1;
                tail_total += weight;
            }
            i += CUBE_DIM_X as usize;
        }
    }
    let total = reduce_mass(&mut values, lane, local_total);
    let retained_total = reduce_mass(&mut values, lane, tail_total);
    let overflow = reduce_mass(
        &mut values,
        lane,
        if entries > 16 { 1.0f32 } else { 0.0f32 },
    );
    let mut threshold = 0u32;
    let mut cutoff = i32::cast_from(CUBE_POS_X) - 1;
    if top_p < 1.0 {
        let target = total * top_p;
        let compact = overflow == 0.0 && retained_total > target;
        // Build the cutoff in descending bit order without overflowing U32.
        let mut bit = 0x8000_0000u32;
        while bit != 0 {
            let candidate = threshold | bit;
            let mut local_mass = 0.0f32;
            if compact {
                if candidate <= floor {
                    local_mass = tail_total;
                } else {
                    let mut slot = lane;
                    let stop = entries * CUBE_DIM_X as usize + lane;
                    while slot < stop {
                        if keys[slot] >= candidate {
                            local_mass += masses[slot];
                        }
                        slot += CUBE_DIM_X as usize;
                    }
                }
            } else {
                i = lane;
                while i < vocabulary {
                    if score_key(scores[i]) >= candidate {
                        local_mass += if tiled {
                            (scores[i] - maximum).exp()
                        } else {
                            weights[i]
                        };
                    }
                    i += CUBE_DIM_X as usize;
                }
            }
            if reduce_mass(&mut values, lane, local_mass) > target {
                threshold = candidate;
            }
            bit >>= 1;
        }
        let mut above = 0.0f32;
        i = lane;
        while i < vocabulary {
            if score_key(scores[i]) > threshold {
                above += if tiled {
                    (scores[i] - maximum).exp()
                } else {
                    weights[i]
                };
            }
            i += CUBE_DIM_X as usize;
        }
        let above = reduce_mass(&mut values, lane, above);
        if lane == 0 {
            let remaining = if target > above {
                target - above
            } else {
                0.0f32
            };
            let bits = if (threshold & 0x8000_0000u32) != 0u32 {
                threshold ^ 0x8000_0000u32
            } else {
                !threshold
            };
            let unit = (f32::reinterpret(bits) - maximum).exp();
            if above == 0.0 || remaining >= unit {
                let mut seen = 0i32;
                for token in 0..vocabulary {
                    if score_key(scores[token]) == threshold {
                        seen += 1;
                        let token_mass = if tiled { unit } else { weights[token] };
                        if remaining >= token_mass * seen as f32 || (above == 0.0 && seen == 1) {
                            cutoff = token as i32;
                        } else {
                            break;
                        }
                    }
                }
            }
            ids[0] = cutoff;
        }
        sync_cube();
        cutoff = ids[0];
        sync_cube();
    }
    let mut best = f32::cast_from(CUBE_POS_X) - 1.0f32;
    let mut chosen = i32::cast_from(CUBE_POS_X) - 1;
    i = lane;
    while i < vocabulary {
        let key = score_key(scores[i]);
        if top_p == 1.0 || key > threshold || (key == threshold && i as i32 <= cutoff) {
            let denominator = race_denominators[i];
            let race = ((scores[i] - maximum) / temperature).exp() / denominator;
            if denominator > 0.0 && race > best {
                best = race;
                chosen = i as i32;
            }
        }
        i += CUBE_DIM_X as usize;
    }
    values[lane] = best;
    ids[lane] = chosen;
    sync_cube();
    width = CUBE_DIM_X as usize / 2;
    while width > 0 {
        if lane < width {
            let other = lane + width;
            if values[other] > values[lane]
                || (values[other] == values[lane] && ids[other] < ids[lane])
            {
                values[lane] = values[other];
                ids[lane] = ids[other];
            }
        }
        sync_cube();
        width /= 2;
    }
    if lane == 0 {
        result[0] = ids[0];
        result[1] = if ids[0] == eos_id { 1 } else { 0 };
    }
}

pub(super) fn launch_prepared_top_p(
    client: &Client,
    logits: Handle,
    history: Handle,
    draws: Handle,
    scores: Handle,
    weights: Handle,
    result: Handle,
    vocabulary: usize,
    params: SamplingParams,
) {
    top_p_token_kernel::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(SAMPLER_THREADS),
        unsafe { BufferArg::from_raw_parts(logits, vocabulary) },
        unsafe { BufferArg::from_raw_parts(history, 16) },
        unsafe { BufferArg::from_raw_parts(draws, vocabulary) },
        unsafe { BufferArg::from_raw_parts(scores, vocabulary) },
        unsafe { BufferArg::from_raw_parts(weights, vocabulary) },
        unsafe { BufferArg::from_raw_parts(result, 2) },
        0,
        vocabulary,
        false,
        params.top_p,
        params.temperature,
        params.repetition_penalty,
        i32::MAX,
    );
}

pub(super) fn enqueue_top_p(
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
) -> Result<Handle> {
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
    let tiled = vocabulary >= 8192;
    if tiled {
        semantic_tiles_kernel::launch(
            client,
            CubeCount::Static(vocabulary.div_ceil(128) as u32, 1, 1),
            CubeDim::new_1d(SAMPLER_THREADS),
            unsafe { BufferArg::from_raw_parts(logits.clone(), total) },
            unsafe { BufferArg::from_raw_parts(history_handle.clone(), history.len()) },
            unsafe { BufferArg::from_raw_parts(scores.clone(), vocabulary) },
            unsafe { BufferArg::from_raw_parts(weights.clone(), vocabulary) },
            offset,
            vocabulary,
            params.repetition_penalty,
        );
    }
    let result = client.empty(2 * size_of::<i32>());
    top_p_token_kernel::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(SAMPLER_THREADS),
        unsafe { BufferArg::from_raw_parts(logits.clone(), total) },
        unsafe { BufferArg::from_raw_parts(history_handle.clone(), history.len()) },
        unsafe { BufferArg::from_raw_parts(draws.clone(), vocabulary) },
        unsafe { BufferArg::from_raw_parts(scores.clone(), vocabulary) },
        unsafe { BufferArg::from_raw_parts(weights.clone(), vocabulary) },
        unsafe { BufferArg::from_raw_parts(result.clone(), 2) },
        offset,
        vocabulary,
        tiled,
        params.top_p,
        params.temperature,
        params.repetition_penalty,
        eos,
    );
    retained.extend([history_handle, draws, scores, weights, result.clone()]);
    Ok(result)
}

pub(super) fn top_p_device_token_with_handle(
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
) -> Result<(u32, bool, Handle)> {
    let handle = enqueue_top_p(
        client, logits, total, offset, vocabulary, history, params, eos_id, rng, retained,
    )?;
    let bytes = client
        .read_one(handle.clone())
        .map_err(|_| FishError::InvalidTransition("CubeCL GPU sampled token readback failed"))?;
    let (token, eos) = decode_sample(&bytes, vocabulary, eos_id)?;
    Ok((token, eos, handle))
}

#[cfg(all(test, feature = "cubecl-cpu"))]
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
    top_p_device_token_with_handle(
        client, logits, total, offset, vocabulary, history, params, eos_id, rng, retained,
    )
    .map(|(token, eos, _)| (token, eos))
}

/// Copy the validated selection to the next graph without staging the token through CPU.
/// Each source is the two-element [token, EOS] sampler output.
#[cube(launch)]
pub(super) fn pack_selected_token_kernel(
    sampled: &[i32],
    output: &mut [i32],
    #[comptime] position: usize,
    #[comptime] derive_initial: bool,
    semantic_begin: i32,
) {
    if ABSOLUTE_POS == 0 {
        let word = sampled[0];
        output[position] = if derive_initial {
            if word >= semantic_begin {
                word - semantic_begin
            } else {
                0i32
            }
        } else {
            word
        };
    }
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
    fn cubecl_sampled_tokens_remain_device_resident_across_a_column_pack() {
        let client = Device::Cpu(Default::default()).client();
        let semantic = client.create_from_slice(bytemuck::cast_slice(&[501i32, 0]));
        let acoustic1 = client.create_from_slice(bytemuck::cast_slice(&[6i32, 0]));
        let acoustic2 = client.create_from_slice(bytemuck::cast_slice(&[11i32, 0]));
        let output = client.empty(4 * 4);
        for (index, (handle, derived)) in [
            (&semantic, false),
            (&semantic, true),
            (&acoustic1, false),
            (&acoustic2, false),
        ]
        .iter()
        .enumerate()
        {
            pack_selected_token_kernel::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe { BufferArg::from_raw_parts((*handle).clone(), 2) },
                unsafe { BufferArg::from_raw_parts(output.clone(), 4) },
                index,
                *derived,
                500i32,
            );
        }
        let bytes = client.read_one(output).unwrap();
        let ids = bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(ids, [501, 1, 6, 11]);

        let first_underflow = client.create_from_slice(bytemuck::cast_slice(&[17i32, 1]));
        let out = client.empty(4);
        pack_selected_token_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(first_underflow, 2) },
            unsafe { BufferArg::from_raw_parts(out.clone(), 1) },
            0,
            true,
            500i32,
        );
        assert_eq!(
            i32::from_le_bytes(client.read_one(out).unwrap()[..4].try_into().unwrap()),
            0
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
