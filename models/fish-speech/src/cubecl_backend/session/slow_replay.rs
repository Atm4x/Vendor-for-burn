//! Replay one Slow AR decode step with request-owned KV and a dynamic position.
use super::*;
use crate::cubecl_backend::runtime::FastFrameScratch;
use cubecl::bytes::Bytes;

pub(super) struct SlowReplay {
    pub(super) bytes: usize,
    graph: cubecl::client::Graph,
    scratch: FastFrameScratch,
    tokens: Handle,
    logits: CubeClDeviceTensor,
    context: CubeClDeviceTensor,
    capacity: usize,
    channels: usize,
}

impl CubeClRequest<'_> {
    pub(in crate::cubecl_backend) fn replay_slow(
        &mut self,
        active: usize,
        tokens: &[u32],
        lut: &[i32],
        codebook: usize,
    ) -> Result<(CubeClDeviceTensor, CubeClDeviceTensor)> {
        if self.poisoned
            || self.cursor.slow_active_len.checked_add(1) != Some(active)
            || active > 1024
            || tokens.is_empty()
            || tokens[0] as usize >= lut.len()
            || tokens[1..].iter().any(|&id| id as usize >= codebook)
        {
            return Err(FishError::InvalidTransition(
                "invalid captured Slow decode input",
            ));
        }
        self.retire_pending()?;
        let client = self.model.executor.client();
        let position = (active - 1) as u32;
        if self.slow_replay.is_none() {
            let plan = self
                .model
                .prepared
                .graph(&format!("slow_decode_{active}"))?;
            let ir = plan.graph();
            self.cursor.validate_step(ir)?;
            let input = ir
                .values
                .iter()
                .find(|value| value.name == "tokens")
                .ok_or(FishError::InvalidShape("missing Slow decode tokens"))?;
            if input.ty.shape.iter().product::<usize>() != tokens.len() {
                return Err(FishError::InvalidShape(
                    "Slow decode token geometry changed",
                ));
            }
            let bytes = self
                .model
                .executor
                .scratch_bytes(plan)?
                .checked_add(std::mem::size_of_val(lut) + std::mem::size_of_val(tokens) + 4)
                .ok_or(FishError::InvalidShape("Slow replay scratch overflow"))?;
            if bytes
                > self
                    .limits
                    .max_pending_bytes
                    .saturating_sub(self.pending_bytes)
            {
                return Err(FishError::InvalidTransition(
                    "Slow replay exceeds scratch budget",
                ));
            }
            let mut bindings = self.model.weights.clone();
            bindings.extend(
                self.caches
                    .iter()
                    .map(|(name, handle)| (name.clone(), handle.clone())),
            );
            let token_handle = client.create_from_slice(bytemuck::cast_slice(tokens));
            let mut scratch = FastFrameScratch::default();
            scratch.position = Some(client.create_from_slice(bytemuck::bytes_of(&position)));
            bindings.insert("tokens".into(), token_handle.clone());
            bindings.insert(
                "semantic_membership_lut".into(),
                client.create_from_slice(bytemuck::cast_slice(lut)),
            );
            let mut keep = Vec::new();
            let mut dispatch = || {
                scratch.cursor = 0;
                keep.clear();
                self.model.executor.execute_handles_inner(
                    plan,
                    &bindings,
                    &mut keep,
                    true,
                    Some(&mut scratch),
                )
            };
            let err = |e| FishError::Backend(format!("Slow graph capture: {e:?}"));
            client.graph_prepare().map_err(err)?;
            if let Err(error) = dispatch() {
                let _ = client.stop_capture();
                return Err(error);
            }
            if let Err(error) = client.start_capture() {
                let _ = client.stop_capture();
                return Err(err(error));
            }
            let results = dispatch();
            let graph = client.stop_capture().map_err(err)?;
            let outputs = results?;
            let tensor = |name: &str| -> Result<CubeClDeviceTensor> {
                let value = ir
                    .outputs
                    .iter()
                    .map(|id| &ir.values[id.0])
                    .find(|v| v.name == name)
                    .ok_or(FishError::InvalidShape("missing Slow graph output"))?;
                Ok(CubeClDeviceTensor {
                    ty: value.ty.clone(),
                    handle: outputs[name].clone(),
                    request_id: self.request_id,
                    verified_tokens: None,
                })
            };
            self.slow_replay = Some(SlowReplay {
                bytes,
                graph,
                scratch,
                tokens: token_handle,
                logits: tensor("logits")?,
                context: tensor("fast_context")?,
                capacity: ir.kv_capacity.unwrap(),
                channels: tokens.len(),
            });
            self.pending_bytes += bytes;
        }
        let replay = self.slow_replay.as_ref().unwrap();
        if active > replay.capacity || tokens.len() != replay.channels {
            return Err(FishError::InvalidShape("Slow replay geometry changed"));
        }
        let write =
            |handle, bytes: &[u8]| client.write(handle, Bytes::from_bytes_vec(bytes.to_vec()));
        write(&replay.tokens, bytemuck::cast_slice(tokens));
        write(
            replay.scratch.position.as_ref().unwrap(),
            bytemuck::bytes_of(&position),
        );
        for (&(_, frames, dim, base), (cos, sin, _)) in &replay.scratch.rotary {
            let table = fish_speech_core::ar_rotary::fish_rope_coefficients(
                position as usize,
                frames,
                dim,
                f32::from_bits(base),
            )?;
            write(cos, bytemuck::cast_slice(&table.cosine));
            write(sin, bytemuck::cast_slice(&table.signed_sine));
        }
        self.pending_steps += 1;
        if let Err(error) = unsafe { replay.graph.replay() } {
            self.poisoned = true;
            return Err(FishError::Backend(format!("Slow graph replay: {error:?}")));
        }
        self.cursor.slow_active_len = active;
        Ok((replay.logits.clone(), replay.context.clone()))
    }
}
