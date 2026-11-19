use crate::trtx_lowering::LoweringError;
use fish_speech_core::firefly_trtx_fsq::FsqDecodeLookupPlan;
use fish_speech_core::fsq_cpu::FsqGroupProjection;
use safetensors::tensor::Dtype;
use trtx::{DataType, ElementWiseOperation, NetworkDefinition, Tensor};

fn trt(e: trtx::Error) -> LoweringError {
    LoweringError::TensorRt(e.to_string())
}
fn invalid(s: impl Into<String>) -> LoweringError {
    LoweringError::InvalidBlueprint(s.into())
}

/// Fish 1.5 FSQ decode as indexed affine projection lookup rather than 8
/// quantizer iterations + 8 linear layers. One 2 MiB FP32 table for 8 groups,
/// one integer offset, one gather and two shuffles. Exact CPU lookup parity is
/// tested separately. No GPU speedup or engine compatibility claim is made.
pub fn lower_fsq_decode_lut<'n>(
    network: &mut NetworkDefinition<'n>,
    input: &Tensor<'n>,
    plan: &FsqDecodeLookupPlan,
    table: Vec<u8>,
) -> Result<Tensor<'n>, LoweringError> {
    let [batch, groups, _, frames] = plan.input_nchw;
    if groups != 8 || plan.codebook_size != 1000 || plan.features_per_group != 64 {
        return Err(invalid(
            "FSQ lookup only supports Fish 1.5's 8x1000x64 layout",
        ));
    }
    let n = i64::try_from(batch).map_err(|_| invalid("batch exceeds i64"))?;
    let g = i64::try_from(groups).map_err(|_| invalid("groups exceed i64"))?;
    let t = i64::try_from(frames).map_err(|_| invalid("frames exceed i64"))?;
    if input.data_type(network) != DataType::kINT32
        || input.dimensions(network).map_err(trt)? != [n, g, 1, t]
    {
        return Err(invalid(
            "FSQ decode requires [batch,8,1,frames] I32 indices",
        ));
    }
    let expected_bytes = groups * (plan.codebook_size + 1) * plan.features_per_group * 4;
    if table.len() != expected_bytes {
        return Err(invalid("invalid FSQ lookup table byte count"));
    }
    // The execution boundary validates code indices before uploading them.
    // TensorRT assertions on the codes would turn this data input into a shape input.
    let mut offsets = Vec::with_capacity(groups * 4);
    for group in 0..groups {
        offsets.extend_from_slice(
            &(i32::try_from(group * (plan.codebook_size + 1) + 1)
                .map_err(|_| invalid("FSQ offset overflow"))?)
            .to_le_bytes(),
        );
    }
    let group_offsets = network
        .add_constant_owned(
            &[1, g, 1, 1],
            offsets,
            DataType::kINT32,
            Some("fsq.group_offsets"),
        )
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let lookup_indices = network
        .add_elementwise(input, &group_offsets, ElementWiseOperation::kSUM)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let mut flatten = network.add_shuffle(&lookup_indices).map_err(trt)?;
    flatten
        .set_reshape_dimensions(network, &[n, g, t])
        .map_err(trt)?;
    let lookup_indices = flatten.output(network, 0).map_err(trt)?;
    let table = network
        .add_constant_owned(
            &[
                (groups * (plan.codebook_size + 1)) as i64,
                plan.features_per_group as i64,
            ],
            table,
            DataType::kFLOAT,
            Some("fsq.decode_lut"),
        )
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let gathered = network
        .add_gather(&table, &lookup_indices, 0)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let mut pack = network.add_shuffle(&gathered).map_err(trt)?;
    pack.set_first_transpose(network, &[0, 1, 3, 2])
        .map_err(trt)?;
    pack.set_reshape_dimensions(network, &[n, g * (plan.features_per_group as i64), 1, t])
        .map_err(trt)?;
    let output = pack.output(network, 0).map_err(trt)?;
    let expected = plan
        .output_nchw
        .iter()
        .map(|&v| i64::try_from(v).map_err(|_| invalid("FSQ output dimension overflow")))
        .collect::<Result<Vec<_>, _>>()?;
    if output.dimensions(network).map_err(trt)? != expected {
        return Err(LoweringError::TensorRt(format!(
            "FSQ lookup output != {expected:?}"
        )));
    }
    Ok(output)
}

pub fn lower_fsq_decode_from_checkpoint<'n>(
    network: &mut NetworkDefinition<'n>,
    input: &Tensor<'n>,
    plan: &FsqDecodeLookupPlan,
    checkpoint: &crate::weights::LoadedWeights,
) -> Result<Tensor<'n>, LoweringError> {
    let mut projections = Vec::with_capacity(plan.groups);
    for group in 0..plan.groups {
        let key_w = &plan.checkpoint_keys[group * 2];
        let key_b = &plan.checkpoint_keys[group * 2 + 1];
        let w = checkpoint
            .tensor(key_w)
            .map_err(|e| invalid(format!("{key_w}: {e}")))?;
        let b = checkpoint
            .tensor(key_b)
            .map_err(|e| invalid(format!("{key_b}: {e}")))?;
        if w.dtype() != Dtype::F32
            || b.dtype() != Dtype::F32
            || w.shape() != [plan.features_per_group, 4]
            || b.shape() != [plan.features_per_group]
        {
            return Err(invalid(format!(
                "FSQ group {group} project_out shape/dtype mismatch"
            )));
        }
        let to_floats = |bytes: &[u8]| {
            bytes
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        projections.push(FsqGroupProjection {
            weights: to_floats(w.data()),
            bias: to_floats(b.data()),
        });
    }
    let table = plan
        .table_f32_bytes(projections)
        .map_err(|e| invalid(format!("FSQ decode table: {e:?}")))?;
    lower_fsq_decode_lut(network, input, plan, table)
}

fn fsq_constant<'n>(
    network: &mut NetworkDefinition<'n>,
    values: &[f32],
) -> Result<Tensor<'n>, LoweringError> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    network
        .add_constant_owned(
            &[1, values.len() as i64, 1, 1],
            bytes,
            DataType::kFLOAT,
            None,
        )
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)
}

pub fn lower_fsq_encode<'n>(
    network: &mut NetworkDefinition<'n>,
    input: &Tensor<'n>,
    plan: &fish_speech_core::firefly_trtx_fsq::FsqEncodePlan,
    projections: &[FsqGroupProjection],
) -> Result<Tensor<'n>, LoweringError> {
    use trtx::{ActivationType, Axes, ReduceOperation, UnaryOperation};

    let shape = plan.input_nchw.map(|x| x as i64);
    if input.data_type(network) != DataType::kFLOAT
        || input.dimensions(network).map_err(trt)? != shape
    {
        return Err(invalid("FSQ encoder expects FP32 [B,512,1,T]"));
    }
    if plan.groups != 8 || plan.group_width != 64 || plan.output_nchw[1] != 8 {
        return Err(invalid("unsupported Fish 1.5 FSQ encode dimensions"));
    }
    let (kernel, bias) = plan
        .grouped_conv_weights(projections)
        .map_err(|e| invalid(format!("FSQ encoder projections: {e:?}")))?;
    // FSQ rounds the projected features into discrete codes. Keep the small
    // dot products in FP32 instead of using a convolution tactic.
    let width = plan.group_width as i64;
    let mut grouped = network.add_shuffle(input).map_err(trt)?;
    grouped
        .set_reshape_dimensions(network, &[shape[0], 8, 1, width, shape[3]])
        .map_err(trt)?;
    let grouped = grouped.output(network, 0).map_err(trt)?;
    let weights = network
        .add_constant_owned(
            &[1, 8, 4, width, 1],
            kernel,
            DataType::kFLOAT,
            Some("fsq.project_in.grouped.weight"),
        )
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let products = network
        .add_elementwise(&grouped, &weights, ElementWiseOperation::kPROD)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let sums = network
        .add_reduce(&products, ReduceOperation::kSUM, Axes::new([3]), false)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let mut flat = network.add_shuffle(&sums).map_err(trt)?;
    flat.set_reshape_dimensions(network, &[shape[0], 32, 1, shape[3]])
        .map_err(trt)?;
    let flat = flat.output(network, 0).map_err(trt)?;
    let bias = network
        .add_constant_owned(
            &[1, 32, 1, 1],
            bias,
            DataType::kFLOAT,
            Some("fsq.project_in.grouped.bias"),
        )
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let mut value = network
        .add_elementwise(&flat, &bias, ElementWiseOperation::kSUM)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;

    let constants = plan.scalar_constants();
    let shift = fsq_constant(network, &constants.0)?;
    let half = fsq_constant(network, &constants.1)?;
    let offset = fsq_constant(network, &constants.2)?;
    let digit_half = fsq_constant(network, &constants.3)?;
    let basis = fsq_constant(network, &constants.4)?;
    for _ in 0..2 {
        let shifted = network
            .add_elementwise(&value, &shift, ElementWiseOperation::kSUM)
            .map_err(trt)?
            .output(network, 0)
            .map_err(trt)?;
        let bounded = network
            .add_activation(&shifted, ActivationType::kTANH)
            .map_err(trt)?
            .output(network, 0)
            .map_err(trt)?;
        let scaled = network
            .add_elementwise(&bounded, &half, ElementWiseOperation::kPROD)
            .map_err(trt)?
            .output(network, 0)
            .map_err(trt)?;
        value = network
            .add_elementwise(&scaled, &offset, ElementWiseOperation::kSUB)
            .map_err(trt)?
            .output(network, 0)
            .map_err(trt)?;
    }
    let rounded = network
        .add_unary(&value, UnaryOperation::kROUND)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let digits = network
        .add_elementwise(&rounded, &digit_half, ElementWiseOperation::kSUM)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let weighted = network
        .add_elementwise(&digits, &basis, ElementWiseOperation::kPROD)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let mut shuffle = network.add_shuffle(&weighted).map_err(trt)?;
    shuffle
        .set_reshape_dimensions(network, &[shape[0], 8, 4, shape[3]])
        .map_err(trt)?;
    let grouped = shuffle.output(network, 0).map_err(trt)?;
    let sum = network
        .add_reduce(&grouped, ReduceOperation::kSUM, Axes::new([2]), true)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let codes = network
        .add_cast(&sum, DataType::kINT32)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    if codes.dimensions(network).map_err(trt)? != plan.output_nchw.map(|x| x as i64) {
        return Err(invalid("FSQ encode output dimensions mismatch"));
    }
    Ok(codes)
}

pub fn lower_fsq_encode_from_checkpoint<'n>(
    network: &mut NetworkDefinition<'n>,
    input: &Tensor<'n>,
    plan: &fish_speech_core::firefly_trtx_fsq::FsqEncodePlan,
    checkpoint: &crate::weights::LoadedWeights,
) -> Result<Tensor<'n>, LoweringError> {
    let mut projections = Vec::with_capacity(plan.groups);
    for group in 0..plan.groups {
        let weight_key = &plan.checkpoint_keys[group * 2];
        let bias_key = &plan.checkpoint_keys[group * 2 + 1];
        let w = checkpoint
            .tensor(weight_key)
            .map_err(|e| invalid(format!("{weight_key}: {e}")))?;
        let b = checkpoint
            .tensor(bias_key)
            .map_err(|e| invalid(format!("{bias_key}: {e}")))?;
        if w.dtype() != Dtype::F32
            || b.dtype() != Dtype::F32
            || w.shape() != [4, plan.group_width]
            || b.shape() != [4]
        {
            return Err(invalid("FSQ encode requires exact FP32 project_in tensors"));
        }
        let unpack = |bytes: &[u8]| {
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect()
        };
        projections.push(FsqGroupProjection {
            weights: unpack(w.data()),
            bias: unpack(b.data()),
        });
    }
    lower_fsq_encode(network, input, plan, &projections)
}
