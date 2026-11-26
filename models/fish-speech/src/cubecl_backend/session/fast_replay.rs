//! Captured Fast AR frame for one CubeCL synthesis request.

use super::*;

fn checked_bytes(bytes: Option<usize>) -> Result<usize> {
    bytes.ok_or(FishError::InvalidShape(
        "CubeCL Fast replay byte count overflow",
    ))
}

#[cube(launch)]
fn copy_fast_context(input: &[f32], out: &mut [f32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    }
}

struct FastInputs {
    stride: usize,
    draw_offset: usize,
    bytes: usize,
}

impl FastInputs {
    fn new(client: &Client, vocabulary: usize, steps: usize) -> Result<Self> {
        let alignment = (client.properties().memory.alignment as usize).max(4);
        let aligned =
            |bytes: usize| checked_bytes(bytes.div_ceil(alignment).checked_mul(alignment));
        let draw_offset = aligned(64)?;
        let stride = checked_bytes(
            draw_offset.checked_add(aligned(checked_bytes(vocabulary.checked_mul(4))?)?),
        )?;
        let bytes = checked_bytes(
            stride
                .checked_mul(steps)
                .and_then(|bytes| bytes.checked_add(draw_offset)),
        )?;
        Ok(Self {
            stride,
            draw_offset,
            bytes,
        })
    }

    fn history_offset(&self, index: usize) -> usize {
        self.draw_offset + index * self.stride
    }

    fn view(&self, handle: &Handle, start: usize, bytes: usize) -> Handle {
        handle
            .clone()
            .offset_start(start as u64)
            .offset_end((self.bytes - start - bytes) as u64)
    }
}

pub(super) struct CapturedFastFrame {
    pub(super) scratch_bytes: usize,
    pub(super) kv_bytes: usize,
    graph: cubecl::client::Graph,
    context: Handle,
    context_size: usize,
    inputs: Handle,
    input_layout: FastInputs,
    vocabulary: usize,
    samples: Vec<Handle>,
}

impl CapturedFastFrame {
    fn new(
        model: &CubeClModel,
        vocabulary: usize,
        steps: usize,
        params: SamplingParams,
        scratch_bytes: usize,
        kv_bytes: usize,
    ) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        use crate::cubecl_backend::runtime::FastFrameScratch;
        let client = model.executor.client();
        let context_type = &model
            .prepared
            .graph("fast_0")?
            .graph()
            .values
            .iter()
            .find(|v| v.name == "fast_context")
            .ok_or("missing Fast context")?
            .ty;
        let context_size = context_type.shape.iter().product();
        let context = model.executor.zeros(context_type)?;
        let input_layout = FastInputs::new(client, vocabulary, steps)?;
        let mut bytes = vec![0u8; input_layout.bytes];
        for index in 0..steps {
            let offset = input_layout.history_offset(index);
            bytes[offset..offset + 64].fill(255);
            let draws = offset + input_layout.draw_offset;
            for draw in bytes[draws..draws + vocabulary * 4].chunks_exact_mut(4) {
                draw.copy_from_slice(&1.0f32.to_le_bytes());
            }
        }
        let inputs = client.create_from_slice(&bytes);
        let initial = input_layout.view(&inputs, 0, 4);
        let mut caches = BTreeMap::new();
        for (name, ty) in &model.prepared.caches {
            if name.starts_with("fast_layers.") {
                caches.insert(name.clone(), model.executor.zeros(ty)?);
            }
        }
        let histories = (0..steps)
            .map(|i| input_layout.view(&inputs, input_layout.history_offset(i), 64))
            .collect::<Vec<_>>();
        let draws = (0..steps)
            .map(|i| {
                input_layout.view(
                    &inputs,
                    input_layout.history_offset(i) + input_layout.draw_offset,
                    vocabulary * 4,
                )
            })
            .collect::<Vec<_>>();
        let scores = (0..steps)
            .map(|_| client.empty(vocabulary * 4))
            .collect::<Vec<_>>();
        let weights = (0..steps)
            .map(|_| client.empty(vocabulary * 4))
            .collect::<Vec<_>>();
        let samples = (0..steps).map(|_| client.empty(8)).collect::<Vec<_>>();
        let mut scratch = FastFrameScratch::default();
        let mut retained = Vec::new();
        let mut dispatch = || -> Result<()> {
            scratch.cursor = 0;
            retained.clear();
            for index in 0..=steps {
                let plan = model.prepared.graph(&format!("fast_{index}"))?;
                let mut bindings = model.weights.clone();
                bindings.extend(caches.iter().map(|(n, h)| (n.clone(), h.clone())));
                if index == 0 {
                    bindings.insert("fast_context".into(), context.clone());
                } else {
                    bindings.insert(
                        "previous_acoustic_code".into(),
                        if index == 1 {
                            initial.clone()
                        } else {
                            samples[index - 2].clone()
                        },
                    );
                }
                let outputs = model.executor.execute_handles_inner(
                    plan,
                    &bindings,
                    &mut retained,
                    true,
                    Some(&mut scratch),
                )?;
                if index > 0 {
                    let i = index - 1;
                    crate::cubecl_backend::sampling::launch_prepared_top_p(
                        client,
                        outputs["fast_logits"].clone(),
                        histories[i].clone(),
                        draws[i].clone(),
                        scores[i].clone(),
                        weights[i].clone(),
                        samples[i].clone(),
                        vocabulary,
                        params,
                    );
                }
            }
            Ok(())
        };
        client.graph_prepare()?;
        if let Err(err) = dispatch() {
            let _ = client.stop_capture();
            return Err(err.into());
        }
        if let Err(err) = client.start_capture() {
            let _ = client.stop_capture();
            return Err(err.into());
        }
        let recording = dispatch();
        let graph = client.stop_capture();
        recording?;
        Ok(Self {
            graph: graph?,
            context,
            context_size,
            inputs,
            input_layout,
            vocabulary,
            samples,
            scratch_bytes,
            kv_bytes,
        })
    }

    fn replay(
        &self,
        client: &Client,
        context: &Handle,
        initial: u32,
        histories: &[Vec<usize>],
        vocabulary: usize,
        rng: &mut fish_speech_core::sampling::SessionRng,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        use cubecl::bytes::Bytes;
        copy_fast_context::launch(
            client,
            CubeCount::Static(self.context_size.div_ceil(128) as u32, 1, 1),
            CubeDim::new_1d(128),
            unsafe { BufferArg::from_raw_parts(context.clone(), self.context_size) },
            unsafe { BufferArg::from_raw_parts(self.context.clone(), self.context_size) },
        );
        let mut bytes = vec![0u8; self.input_layout.bytes];
        bytes[..4].copy_from_slice(&initial.to_le_bytes());
        for (i, history) in histories.iter().enumerate() {
            let offset = self.input_layout.history_offset(i);
            let packed = &mut bytes[offset..offset + 64];
            packed.fill(255);
            for (target, &token) in packed.chunks_exact_mut(4).zip(history) {
                target.copy_from_slice(&i32::try_from(token)?.to_le_bytes());
            }
            let offset = offset + self.input_layout.draw_offset;
            for target in bytes[offset..offset + vocabulary * 4].chunks_exact_mut(4) {
                target.copy_from_slice(&((-rng.uniform_open_01().ln()) as f32).to_le_bytes());
            }
        }
        client.write(&self.inputs, Bytes::from_bytes_vec(bytes));
        unsafe {
            self.graph.replay()?;
        }
        Ok(())
    }
}

impl CubeClRequest<'_> {
    fn fast_capture_scratch(&self, vocabulary: usize, steps: usize) -> Result<usize> {
        let input = self
            .model
            .prepared
            .graph("fast_0")?
            .graph()
            .values
            .iter()
            .find(|v| v.name == "fast_context")
            .ok_or(FishError::InvalidShape("missing Fast context"))?;
        let initial_bytes =
            checked_bytes(self.model.executor.tensor_bytes(&input.ty)?.checked_add(
                FastInputs::new(self.model.executor.client(), vocabulary, steps)?.bytes,
            ))?;
        let sample_bytes = checked_bytes(
            vocabulary
                .checked_mul(8)
                .and_then(|bytes| bytes.checked_add(8))
                .and_then(|bytes| bytes.checked_mul(steps)),
        )?;
        (0..=steps).try_fold(
            checked_bytes(initial_bytes.checked_add(sample_bytes))?,
            |bytes, step| {
                let graph = self.model.prepared.graph(&format!("fast_{step}"))?;
                checked_bytes(bytes.checked_add(self.model.executor.scratch_bytes(graph)?))
            },
        )
    }

    pub(in crate::cubecl_backend) fn defer_fast_context(&mut self, context: CubeClDeviceTensor) {
        self.replay_context = Some(context);
    }

    pub(in crate::cubecl_backend) fn run_captured_fast(
        &mut self,
        initial: u32,
        histories: &[Vec<usize>],
        vocabulary: usize,
        params: SamplingParams,
        rng: &mut fish_speech_core::sampling::SessionRng,
    ) -> Result<Vec<CubeClSampledToken>> {
        if initial as usize >= vocabulary
            || histories
                .iter()
                .any(|history| history.len() > 16 || history.iter().any(|&id| id >= vocabulary))
        {
            return Err(FishError::InvalidSampling(
                "invalid Fast replay token or history",
            ));
        }
        let context = self
            .replay_context
            .take()
            .ok_or(FishError::InvalidTransition("Fast replay context missing"))?;
        if self.fast_replay.is_none() {
            let scratch = self.fast_capture_scratch(vocabulary, histories.len())?;
            let kv = self
                .model
                .prepared
                .caches
                .iter()
                .filter(|(name, _)| name.starts_with("fast_layers."))
                .try_fold(0usize, |bytes, (_, ty)| {
                    checked_bytes(bytes.checked_add(self.model.executor.tensor_bytes(ty)?))
                })?;
            let pending = checked_bytes(self.pending_bytes.checked_add(scratch))?;
            let kv_total = checked_bytes(self.kv_bytes.checked_add(kv))?;
            if pending > self.limits.max_pending_bytes || kv_total > self.limits.max_kv_bytes {
                return Err(FishError::InvalidTransition(
                    "CubeCL Fast replay exceeds request memory budget",
                ));
            }
            self.fast_replay = Some(
                CapturedFastFrame::new(
                    self.model,
                    vocabulary,
                    histories.len(),
                    params,
                    scratch,
                    kv,
                )
                .map_err(|err| FishError::Backend(format!("Fast replay capture: {err:?}")))?,
            );
            self.pending_bytes = pending;
            self.kv_bytes = kv_total;
        }
        let frame = self.fast_replay.as_ref().expect("capture prepared");
        if frame.context_size != context.shape_elements()
            || frame.samples.len() != histories.len()
            || frame.vocabulary != vocabulary
        {
            return Err(FishError::InvalidShape("Fast replay geometry changed"));
        }
        self.pending_steps += 1;
        frame
            .replay(
                self.model.executor.client(),
                &context.handle,
                initial,
                histories,
                vocabulary,
                rng,
            )
            .map_err(|err| FishError::Backend(format!("Fast replay execute: {err:?}")))?;
        let pending = frame
            .samples
            .iter()
            .cloned()
            .map(|handle| PendingFastSample {
                handle,
                request_id: self.request_id,
            })
            .collect();
        self.cursor.fast_active_len = histories.len() + 1;
        self.read_fast_samples(pending, vocabulary)
    }
}
