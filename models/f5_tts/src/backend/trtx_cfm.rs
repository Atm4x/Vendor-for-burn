use trtx::{
    Builder, DataType, ElementWiseOperation as Ew, Logger, MemoryPoolType, NetworkDefinition,
    Tensor,
};

use super::{
    trtx::{DeviceBuffer, TrtBackboneWeights, TrtTimeWeights},
    trtx_input::{TrtInputShape, TrtInputWeights, input_graph},
    trtx_native::{backbone_graph, binary, dims, fail, scalar, time_mlp_graph},
};
use crate::{F5Error, Result};

fn duplicate<'a>(net: &mut NetworkDefinition<'a>, input: Tensor<'a>) -> Result<Tensor<'a>> {
    let mut concat = net.add_concatenation(&[&input, &input]).map_err(fail)?;
    concat.set_axis(net, 0);
    concat.output(net, 0).map_err(fail)
}

fn half<'a>(
    net: &mut NetworkDefinition<'a>,
    input: Tensor<'a>,
    start_batch: usize,
    batch: usize,
    frames: usize,
    channels: usize,
) -> Result<Tensor<'a>> {
    net.add_slice(
        &input,
        &[dims(start_batch), 0, 0],
        &[dims(batch), dims(frames), dims(channels)],
        &[1, 1, 1],
    )
    .map_err(fail)?
    .output(net, 0)
    .map_err(fail)
}

/// Builds a fixed-shape TensorRT graph for one F5 Euler step. All mel state remains on GPU.
/// Both CFG branches share the DiT weights and execute in one batched TensorRT graph.
/// `condition` must already be masked outside the reference prefix; `reference_mask`
/// is 1.0 for original frames and 0.0 for generated frames. For the final step
/// set `restore=1`, otherwise use 0. Inputs `dt`, `cfg_strength`, and `restore`
/// are device-resident scalar tensors to permit reusing the same serialized engine.
pub fn build_serialized_cfm_step(
    logger: &Logger,
    input: &TrtInputWeights,
    time: &TrtTimeWeights,
    backbone: &TrtBackboneWeights,
    workspace_bytes: usize,
) -> Result<Vec<u8>> {
    let s = input.shape;
    s.validate()?;
    if s.backbone.batch % 2 != 0
        || s.backbone != backbone.shape
        || time.batch != s.backbone.batch
        || time.dim != s.backbone.dim
        || workspace_bytes == 0
    {
        return Err(F5Error::InvalidConfiguration(
            "Invalid TensorRT CFM step configuration",
        ));
    }
    let b = s.backbone.batch / 2;
    let f = s.backbone.frames;
    let c = s.backbone.mel_channels;
    let mut builder = Builder::new(logger).map_err(fail)?;
    let mut net = builder
        .create_network(trtx::builder::network_flags::EXPLICIT_BATCH)
        .map_err(fail)?;
    let state = net
        .add_input("state", DataType::kFLOAT, &[dims(b), dims(f), dims(c)])
        .map_err(fail)?;
    let condition = net
        .add_input("condition", DataType::kFLOAT, &[dims(b), dims(f), dims(c)])
        .map_err(fail)?;
    let token_ids = net
        .add_input("tokens", DataType::kINT32, &[dims(2 * b), dims(f)])
        .map_err(fail)?;
    let text_mask = net
        .add_input("text_mask", DataType::kINT32, &[dims(2 * b), dims(f)])
        .map_err(fail)?;
    let sincos = net
        .add_input("sincos", DataType::kFLOAT, &[dims(2 * b), 256])
        .map_err(fail)?;
    let dt = net.add_input("dt", DataType::kFLOAT, &[1]).map_err(fail)?;
    let cfg = net
        .add_input("cfg_strength", DataType::kFLOAT, &[1])
        .map_err(fail)?;
    let restore = net
        .add_input("restore", DataType::kFLOAT, &[1])
        .map_err(fail)?;
    let reference_mask = net
        .add_input("reference_mask", DataType::kFLOAT, &[dims(b), dims(f), 1])
        .map_err(fail)?;

    let state_doubled = duplicate(&mut net, state)?;
    let zero = scalar(&mut net, 0.0)?;
    let uncond = binary(&mut net, condition, zero, Ew::kPROD)?;
    let mut pair = net
        .add_concatenation(&[&condition, &uncond])
        .map_err(fail)?;
    pair.set_axis(&mut net, 0);
    let condition_doubled = pair.output(&mut net, 0).map_err(fail)?;
    let timestep = time_mlp_graph(&mut net, sincos, time)?;
    let hidden = input_graph(
        &mut net,
        state_doubled,
        condition_doubled,
        token_ids,
        text_mask,
        input,
    )?;
    let velocities = backbone_graph(&mut net, hidden, timestep, backbone)?;
    let conditional = half(&mut net, velocities, 0, b, f, c)?;
    let unconditional = half(&mut net, velocities, b, b, f, c)?;
    let delta = binary(&mut net, conditional, unconditional, Ew::kSUB)?;
    let scaled = binary(&mut net, delta, cfg, Ew::kPROD)?;
    let velocity = binary(&mut net, conditional, scaled, Ew::kSUM)?;
    let increment = binary(&mut net, velocity, dt, Ew::kPROD)?;
    let next = binary(&mut net, state, increment, Ew::kSUM)?;

    let restore_mask = binary(&mut net, restore, reference_mask, Ew::kPROD)?;
    let reference_delta = binary(&mut net, condition, next, Ew::kSUB)?;
    let patch = binary(&mut net, restore_mask, reference_delta, Ew::kPROD)?;
    let updated = binary(&mut net, next, patch, Ew::kSUM)?;
    updated.set_name(&mut net, "next_state").map_err(fail)?;
    net.mark_output(&updated);
    let mut config = builder.create_config().map_err(fail)?;
    config.set_memory_pool_limit(MemoryPoolType::kWORKSPACE, workspace_bytes);
    let image = builder
        .build_serialized_network(&mut net, &mut config)
        .map_err(fail)?;
    if image.is_empty() {
        return Err(F5Error::ModelExecution(
            "TensorRT F5 CFM step engine is empty".into(),
        ));
    }
    Ok(image.as_ref().to_vec())
}

#[derive(Debug, Clone, Copy)]
pub struct CfmStepBindings {
    pub state: DeviceBuffer,
    pub condition: DeviceBuffer,
    pub tokens: DeviceBuffer,
    pub text_mask: DeviceBuffer,
    pub sincos: DeviceBuffer,
    pub dt: DeviceBuffer,
    pub cfg_strength: DeviceBuffer,
    pub restore: DeviceBuffer,
    pub reference_mask: DeviceBuffer,
    pub next_state: DeviceBuffer,
}

impl CfmStepBindings {
    pub fn validate(self, shape: TrtInputShape) -> Result<()> {
        shape.validate()?;
        let s = shape.backbone;
        if s.batch % 2 != 0 {
            return Err(F5Error::InvalidConfiguration(
                "CFM requires paired CFG batches",
            ));
        }
        let b = s.batch / 2;
        let frame_bytes = b
            .checked_mul(s.frames)
            .and_then(|v| v.checked_mul(s.mel_channels))
            .and_then(|v| v.checked_mul(4))
            .ok_or(F5Error::InvalidShape("CFM state byte overflow"))?;
        let token_bytes = s
            .batch
            .checked_mul(s.frames)
            .and_then(|v| v.checked_mul(4))
            .ok_or(F5Error::InvalidShape("CFM token byte overflow"))?;
        let time_bytes = s
            .batch
            .checked_mul(256)
            .and_then(|v| v.checked_mul(4))
            .ok_or(F5Error::InvalidShape("CFM time byte overflow"))?;
        let mask_bytes = b
            .checked_mul(s.frames)
            .and_then(|v| v.checked_mul(4))
            .ok_or(F5Error::InvalidShape("CFM reference mask overflow"))?;
        for (buffer, bytes) in [
            (self.state, frame_bytes),
            (self.condition, frame_bytes),
            (self.tokens, token_bytes),
            (self.text_mask, token_bytes),
            (self.sincos, time_bytes),
            (self.dt, 4),
            (self.cfg_strength, 4),
            (self.restore, 4),
            (self.reference_mask, mask_bytes),
            (self.next_state, frame_bytes),
        ] {
            buffer.validate(bytes)?;
        }
        let end = self
            .next_state
            .address
            .checked_add(self.next_state.bytes)
            .ok_or(F5Error::InvalidShape("CFM output address overflow"))?;
        for x in [
            self.state,
            self.condition,
            self.tokens,
            self.text_mask,
            self.sincos,
            self.dt,
            self.cfg_strength,
            self.restore,
            self.reference_mask,
        ] {
            let xend = x
                .address
                .checked_add(x.bytes)
                .ok_or(F5Error::InvalidShape("CFM input address overflow"))?;
            if self.next_state.address < xend && x.address < end {
                return Err(F5Error::InvalidShape("CFM output cannot alias its inputs"));
            }
        }
        Ok(())
    }
}

/// # Safety
/// All buffers must be allocated on the stream's CUDA device. The caller must keep both
/// state allocations live and synchronize the CUDA stream before releasing them, the engine
/// or the context. Reuse the same engine across Euler steps, swapping state/next_state buffers.
pub unsafe fn enqueue_cfm_step(
    context: &mut trtx::ExecutionContext<'_>,
    shape: TrtInputShape,
    bindings: CfmStepBindings,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    bindings.validate(shape)?;
    if stream.is_null() {
        return Err(F5Error::InvalidShape("Null CUDA stream"));
    }
    unsafe {
        for (name, buffer) in [
            ("state", bindings.state),
            ("condition", bindings.condition),
            ("tokens", bindings.tokens),
            ("text_mask", bindings.text_mask),
            ("sincos", bindings.sincos),
            ("dt", bindings.dt),
            ("cfg_strength", bindings.cfg_strength),
            ("restore", bindings.restore),
            ("reference_mask", bindings.reference_mask),
        ] {
            context
                .set_input_tensor_address(name, buffer.address as *const _)
                .map_err(fail)?;
        }
        context
            .set_tensor_address("next_state", bindings.next_state.address as *mut _)
            .map_err(fail)?;
        context.enqueue_v3(stream).map_err(fail)
    }
}

#[cfg(test)]
mod tests {
    use super::super::trtx::BackboneShape;
    use super::*;

    fn shape() -> TrtInputShape {
        TrtInputShape {
            backbone: BackboneShape {
                batch: 2,
                frames: 5,
                dim: 16,
                heads: 4,
                depth: 2,
                ff_mult: 2,
                mel_channels: 2,
            },
            text_dim: 8,
            vocab_size: 8,
            text_blocks: 1,
        }
    }

    #[test]
    fn gpu_step_fixed_tensors_and_out_of_place_state_are_validated() {
        let mut buffers = CfmStepBindings {
            state: DeviceBuffer {
                address: 0x1000,
                bytes: 40,
            },
            condition: DeviceBuffer {
                address: 0x2000,
                bytes: 40,
            },
            tokens: DeviceBuffer {
                address: 0x3000,
                bytes: 40,
            },
            text_mask: DeviceBuffer {
                address: 0x4000,
                bytes: 40,
            },
            sincos: DeviceBuffer {
                address: 0x5000,
                bytes: 2048,
            },
            dt: DeviceBuffer {
                address: 0x6000,
                bytes: 4,
            },
            cfg_strength: DeviceBuffer {
                address: 0x7000,
                bytes: 4,
            },
            restore: DeviceBuffer {
                address: 0x8000,
                bytes: 4,
            },
            reference_mask: DeviceBuffer {
                address: 0x9000,
                bytes: 20,
            },
            next_state: DeviceBuffer {
                address: 0xA000,
                bytes: 40,
            },
        };
        buffers.validate(shape()).unwrap();
        buffers.next_state.address = buffers.state.address;
        assert!(buffers.validate(shape()).is_err());
        buffers.next_state.address = 0xA000;
        buffers.sincos.bytes -= 4;
        assert!(buffers.validate(shape()).is_err());
    }
}
