fn recurrent_weight(hidden: usize, direction: usize, gate: usize, lane: usize, column: usize) -> f32 {
    let index = ((((direction * 3 + gate) * hidden + lane) * hidden) + column) as u64;
    (((index.wrapping_mul(1097).wrapping_add(31) % 8191) as i32 - 4095) as f32)
        / (hidden as f32 * 6000.0)
}

fn initial_gate(frame: usize, direction: usize, gate: usize, lane: usize) -> f32 {
    let id = (frame * 877 + direction * 131 + gate * 59 + lane * 17) % 2003;
    (id as f32 - 1001.0) / 8000.0
}

fn bias(direction: usize, gate: usize, lane: usize) -> f32 {
    ((direction * 173 + gate * 7 + lane * 13) % 307) as f32 / 60000.0
}

fn gemv(hidden: usize, direction: usize, state: &[f32], split: usize) -> Vec<f32> {
    assert_eq!(hidden % split, 0);
    let mut gates = vec![0_f32; 3 * hidden];
    for gate in 0..3 {
        for lane in 0..hidden {
            if split == 1 {
                let mut value = bias(direction, gate, lane);
                for column in 0..hidden {
                    value += state[column] * recurrent_weight(hidden, direction, gate, lane, column);
                }
                gates[gate * hidden + lane] = value;
            } else {
                let mut partial = vec![0_f32; split];
                for shard in 0..split {
                    partial[shard] = if shard == 0 { bias(direction, gate, lane) } else { 0.0 };
                    for inner in 0..hidden / split {
                        let column = shard * (hidden / split) + inner;
                        partial[shard] += state[column] * recurrent_weight(hidden, direction, gate, lane, column);
                    }
                }
                let mut total = partial[0];
                for part in partial.iter().skip(1) {
                    total += part;
                }
                gates[gate * hidden + lane] = total;
            }
        }
    }
    gates
}

fn recurrence(hidden: usize, frames: usize, split: usize) -> Vec<f32> {
    let mut result = vec![0_f32; frames * hidden * 2];
    for direction in 0..2 {
        let mut state = vec![0_f32; hidden];
        for step in 0..frames {
            let frame = if direction == 0 { step } else { frames - step - 1 };
            let recurrent = gemv(hidden, direction, &state, split);
            for lane in 0..hidden {
                let reset = 1.0 / (1.0 + (-(initial_gate(frame, direction, 0, lane) + recurrent[lane])).exp());
                let update = 1.0 / (1.0 + (-(initial_gate(frame, direction, 1, lane) + recurrent[hidden + lane])).exp());
                let candidate = (initial_gate(frame, direction, 2, lane) + reset * recurrent[2 * hidden + lane]).tanh();
                let next = (1.0 - update) * candidate + update * state[lane];
                state[lane] = next;
                result[(frame * 2 + direction) * hidden + lane] = next;
            }
        }
    }
    result
}

#[test]
fn split_shards_cover_each_recurrent_weight_once() {
    for hidden in [4, 8, 16, 32, 256] {
        for split in [2, 4] {
            let mut seen = vec![0_usize; hidden];
            for thread in 0..hidden * split {
                let shard = thread / hidden;
                for inner in 0..hidden / split {
                    if thread % hidden == 0 {
                        seen[shard * (hidden / split) + inner] += 1;
                    }
                }
            }
            assert!(seen.iter().all(|x| *x == 1));
        }
    }
}

#[test]
fn split_recurrent_output_is_numerically_stable() {
    for hidden in [4, 8, 32, 256] {
        let baseline = recurrence(hidden, 8, 1);
        for split in [2, 4] {
            let candidate = recurrence(hidden, 8, split);
            let max_abs = baseline.iter().zip(candidate.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs <= 0.00001, "hidden={hidden}, split={split}, max_abs={max_abs}");
        }
    }
}

#[test]
fn split_coalesced_addressing_matches_original() {
    for hidden in [4, 8, 32, 256] {
        let mut packed = vec![0.0f32; 2 * 3 * hidden * hidden];
        for direction in 0..2 {
            for column in 0..hidden {
                for gate in 0..3 {
                    for lane in 0..hidden {
                        let index = ((direction * hidden + column) * 3 * hidden) + gate * hidden + lane;
                        packed[index] = recurrent_weight(hidden, direction, gate, lane, column);
                    }
                }
            }
        }
        for direction in 0..2 {
            for gate in 0..3 {
                for lane in 0..hidden {
                    for column in 0..hidden {
                        let packed_index = ((direction * hidden + column) * 3 * hidden) + gate * hidden + lane;
                        let original_value = recurrent_weight(hidden, direction, gate, lane, column);
                        assert_eq!(original_value.to_bits(), packed[packed_index].to_bits());
                    }
                }
            }
        }
    }
}

fn gemv_gate_parallel(hidden: usize, direction: usize, state: &[f32]) -> Vec<f32> {
    let mut gates = vec![0_f32; 3 * hidden];
    for thread in 0..3 * hidden {
        let gate = thread / hidden;
        let lane = thread % hidden;
        let mut acc = bias(direction, gate, lane);
        for column in 0..hidden {
            acc += state[column] * recurrent_weight(hidden, direction, gate, lane, column);
        }
        gates[gate * hidden + lane] = acc;
    }
    gates
}

fn recurrence_gate_parallel(hidden: usize, frames: usize) -> Vec<f32> {
    let mut result = vec![0_f32; frames * hidden * 2];
    for direction in 0..2 {
        let mut state = vec![0_f32; hidden];
        for step in 0..frames {
            let frame = if direction == 0 { step } else { frames - step - 1 };
            let recurrent = gemv_gate_parallel(hidden, direction, &state);
            for lane in 0..hidden {
                let reset = 1.0 / (1.0 + (-(initial_gate(frame, direction, 0, lane) + recurrent[lane])).exp());
                let update = 1.0 / (1.0 + (-(initial_gate(frame, direction, 1, lane) + recurrent[hidden + lane])).exp());
                let candidate = (initial_gate(frame, direction, 2, lane) + reset * recurrent[2 * hidden + lane]).tanh();
                let next = (1.0 - update) * candidate + update * state[lane];
                state[lane] = next;
                result[(frame * 2 + direction) * hidden + lane] = next;
            }
        }
    }
    result
}

#[test]
fn gate_parallel_does_not_change_fp32_accumulation_order() {
    for hidden in [4, 8, 32, 256] {
        let state = (0..hidden).map(|x| x as f32 / 1600.0).collect::<Vec<_>>();
        for direction in 0..2 {
            let serial = gemv(hidden, direction, &state, 1);
            let parallel = gemv_gate_parallel(hidden, direction, &state);
            for (lhs, rhs) in serial.iter().zip(parallel.iter()) {
                assert_eq!(lhs.to_bits(), rhs.to_bits());
            }
        }
        let serial = recurrence(hidden, 8, 1);
        let parallel = recurrence_gate_parallel(hidden, 8);
        for (lhs, rhs) in serial.iter().zip(parallel.iter()) {
            assert_eq!(lhs.to_bits(), rhs.to_bits());
        }
    }
}
