use fish_speech_core::{
    ar_rotary::fish_rope_coefficients,
    ar_stream_plan::{ArStreamPhase, ArStreamPlan},
    error::{FishError, Result},
    graph_ir::{Layout, ScalarType, TensorType},
    trtx_blueprint::{Binding, TrtPrimitive, TrtStep},
};

pub const ROPE_COS: &str = "slow_dynamic_rope_cos";
pub const ROPE_SIN: &str = "slow_dynamic_rope_sin";
pub const KV_MASK: &str = "slow_dynamic_kv_mask";

fn input(name: &str, result: Binding, shape: &[usize]) -> TrtStep {
    TrtStep {
        name: name.to_owned(),
        primitive: TrtPrimitive::Input,
        arguments: Vec::new(),
        result,
        output_type: TensorType::new(shape.to_vec(), ScalarType::F32, Layout::Plain),
    }
}

pub fn parameterize_slow_decode(plan: &mut ArStreamPlan, capacity: usize) -> Result<()> {
    plan.validate()?;
    if capacity == 0 {
        return Err(FishError::InvalidShape("empty dynamic Slow decode bucket"));
    }
    let mut next_temp = plan
        .phases
        .iter()
        .filter_map(|phase| match phase {
            ArStreamPhase::Network(network) => Some(
                network
                    .steps
                    .iter()
                    .filter_map(|step| match step.result {
                        Binding::Temp(id) => Some(id),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(0),
            ),
            _ => None,
        })
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or(FishError::InvalidShape("dynamic binding overflow"))?;
    let mut rotations = 0;
    let mut attentions = 0;
    for phase in &mut plan.phases {
        let ArStreamPhase::Network(network) = phase else {
            continue;
        };
        let network = std::sync::Arc::make_mut(network);
        let mut built = Vec::with_capacity(network.steps.len() + 3);
        let mut rotary = None;
        let mut mask = None;
        for mut step in std::mem::take(&mut network.steps) {
            match step.primitive {
                TrtPrimitive::RotaryInterleaved { .. } => {
                    let dim = *step
                        .output_type
                        .shape
                        .get(3)
                        .ok_or(FishError::InvalidShape("RoPE head dimension missing"))?;
                    if dim == 0 || dim % 2 != 0 || step.output_type.shape[2] != 1 {
                        return Err(FishError::InvalidShape(
                            "dynamic RoPE requires single decode token",
                        ));
                    }
                    let (cos, sin) = if let Some((cos, sin, old_dim)) = rotary {
                        if old_dim != dim {
                            return Err(FishError::InvalidShape(
                                "inconsistent RoPE head dimension",
                            ));
                        }
                        (cos, sin)
                    } else {
                        let cos = Binding::Temp(next_temp);
                        next_temp = next_temp
                            .checked_add(1)
                            .ok_or(FishError::InvalidShape("dynamic binding overflow"))?;
                        let sin = Binding::Temp(next_temp);
                        next_temp = next_temp
                            .checked_add(1)
                            .ok_or(FishError::InvalidShape("dynamic binding overflow"))?;
                        built.push(input(ROPE_COS, cos, &[1, 1, 1, dim]));
                        built.push(input(ROPE_SIN, sin, &[1, 1, 1, dim]));
                        rotary = Some((cos, sin, dim));
                        (cos, sin)
                    };
                    step.primitive = TrtPrimitive::RotaryCoefficients;
                    step.arguments.extend([cos, sin]);
                    rotations += 1;
                }
                TrtPrimitive::GqaAttention {
                    q_heads, kv_heads, ..
                } => {
                    let binding = if let Some(binding) = mask {
                        binding
                    } else {
                        let binding = Binding::Temp(next_temp);
                        next_temp = next_temp
                            .checked_add(1)
                            .ok_or(FishError::InvalidShape("dynamic binding overflow"))?;
                        built.push(input(KV_MASK, binding, &[1, 1, 1, 1, capacity]));
                        mask = Some(binding);
                        binding
                    };
                    step.primitive = TrtPrimitive::GqaAttentionMasked { q_heads, kv_heads };
                    step.arguments.push(binding);
                    attentions += 1;
                }
                TrtPrimitive::KvActivePrefix { active_len } if active_len != capacity => {
                    return Err(FishError::InvalidShape(
                        "decode KV prefix does not match bucket",
                    ));
                }
                _ => {}
            }
            built.push(step);
        }
        network.steps = built;
        network.validate()?;
    }
    if rotations == 0 || rotations != attentions * 2 {
        return Err(FishError::InvalidShape(
            "dynamic Slow decode missing RoPE or attention",
        ));
    }
    plan.validate()
}

pub fn positioned_slow_decode(
    template: &ArStreamPlan,
    active_len: usize,
    bucket_end: usize,
) -> Result<ArStreamPlan> {
    if active_len == 0 || active_len > bucket_end || bucket_end == 0 {
        return Err(FishError::InvalidShape(
            "Slow decode position outside compiled KV bucket",
        ));
    }
    let mut masked_networks = 0;
    for phase in &template.phases {
        let ArStreamPhase::Network(network) = phase else {
            continue;
        };
        for step in &network.steps {
            match step.primitive {
                TrtPrimitive::RotaryInterleaved { .. } | TrtPrimitive::GqaAttention { .. } => {
                    return Err(FishError::InvalidShape(
                        "Slow decode contains position-specialized TensorRT operators",
                    ));
                }
                TrtPrimitive::GqaAttentionMasked { .. } => masked_networks += 1,
                _ => {}
            }
        }
    }
    if masked_networks == 0 {
        return Err(FishError::InvalidShape(
            "Slow decode has no runtime attention mask",
        ));
    }
    let position = active_len - 1;
    let mut plan = template.clone();
    let mut barriers = 0;
    for phase in &mut plan.phases {
        if let ArStreamPhase::KvWrites(write) = phase {
            for transfer in [&mut write.key, &mut write.value] {
                let geometry = &mut transfer.geometry;
                if geometry.frames != 1
                    || geometry.row_bytes == 0
                    || geometry.row_bytes != geometry.source_pitch_bytes
                {
                    return Err(FishError::InvalidShape("expected single-token KV write"));
                }
                let offset = geometry
                    .row_bytes
                    .checked_mul(position)
                    .ok_or(FishError::InvalidShape("KV write position overflow"))?;
                if offset
                    .checked_add(geometry.row_bytes)
                    .is_none_or(|end| end > geometry.destination_pitch_bytes)
                {
                    return Err(FishError::InvalidShape(
                        "KV write exceeds physical capacity",
                    ));
                }
                geometry.position = position;
                geometry.destination_offset_bytes = offset;
            }
            write.position = position;
            barriers += 1;
        }
    }
    if barriers == 0 {
        return Err(FishError::InvalidShape("Slow decode has no cache writes"));
    }
    plan.validate()?;
    Ok(plan)
}

pub fn decode_control_bytes(
    active_len: usize,
    bucket_end: usize,
    head_dim: usize,
    rope_base: f32,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    if active_len == 0 || active_len > bucket_end {
        return Err(FishError::InvalidShape(
            "invalid dynamic Slow decode attention mask",
        ));
    }
    let rope = fish_rope_coefficients(active_len - 1, 1, head_dim, rope_base)?;
    let as_bytes = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let mut mask = vec![-1.0e9f32; bucket_end];
    mask[..active_len].fill(0.0);
    Ok((
        as_bytes(&rope.cosine),
        as_bytes(&rope.signed_sine),
        as_bytes(&mask),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fish_speech_core::{
        ar_rotary::fish_rope_coefficients, ar_stream_plan::ArStreamPlan, config::FishModelConfig,
        transformer_ir::TransformerIrFactory,
    };

    #[test]
    fn bucket_graph_is_constant_across_distinct_logical_positions() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let mut plan = ArStreamPlan::from_graph(
            &TransformerIrFactory::new(&config)
                .unwrap()
                .slow_decode(8)
                .unwrap(),
        )
        .unwrap();
        assert!(positioned_slow_decode(&plan, 5, 8).is_err());
        parameterize_slow_decode(&mut plan, 8).unwrap();
        let p5 = positioned_slow_decode(&plan, 5, 8).unwrap();
        let p8 = positioned_slow_decode(&plan, 8, 8).unwrap();
        let networks = |p: &ArStreamPlan| {
            p.phases
                .iter()
                .filter_map(|phase| match phase {
                    ArStreamPhase::Network(net) => Some(format!("{net:?}")),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(networks(&p5), networks(&p8));
        let keys = |p: &ArStreamPlan| -> Vec<String> {
            p.phases
                .iter()
                .filter_map(|phase| match phase {
                    ArStreamPhase::Network(network) => Some(
                        crate::trtx_cache::EngineIdentity::for_blueprint(
                            network,
                            &"0".repeat(64),
                            "sm89",
                            "rtx1.7",
                            "driver",
                            "fp32",
                            "source-1.5-bucketed-slow-v1",
                            "build",
                        )
                        .unwrap()
                        .key()
                        .unwrap(),
                    ),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(keys(&p5), keys(&p8));
        for (pos, candidate) in [(4, p5), (7, p8)] {
            for phase in candidate.phases {
                if let ArStreamPhase::KvWrites(writes) = phase {
                    assert_eq!(writes.position, pos);
                    assert_eq!(
                        writes.key.geometry.destination_offset_bytes,
                        pos * writes.key.geometry.row_bytes
                    );
                }
            }
        }
        assert!(positioned_slow_decode(&plan, 9, 8).is_err());
        assert!(positioned_slow_decode(&plan, 0, 8).is_err());
    }

    #[test]
    fn dynamic_rope_and_mask_match_static_position_and_hide_future_kv() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        for active_len in [1, 5, 8] {
            let (cos, sin, mask) =
                decode_control_bytes(active_len, 8, config.slow.head_dim(), config.rope_base)
                    .unwrap();
            let static_rope =
                fish_rope_coefficients(active_len - 1, 1, config.slow.head_dim(), config.rope_base)
                    .unwrap();
            assert_eq!(
                cos,
                static_rope
                    .cosine
                    .iter()
                    .flat_map(|n| n.to_le_bytes())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                sin,
                static_rope
                    .signed_sine
                    .iter()
                    .flat_map(|n| n.to_le_bytes())
                    .collect::<Vec<_>>()
            );
            let actual = mask
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert!(actual[..active_len].iter().all(|&v| v == 0.0));
            assert!(actual[active_len..].iter().all(|&v| v == -1.0e9));
        }
    }
}
