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

#[cfg(all(test, feature = "cubecl-cpu"))]
mod tests {
    use super::*;
    use cubecl::Device;
    use fish_speech_core::sampling::{greedy, logits_to_probs};

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
