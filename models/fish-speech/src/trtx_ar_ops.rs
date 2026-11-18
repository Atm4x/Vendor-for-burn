use fish_speech_core::ar_attention::{GqaShape, right_aligned_causal_bias};
use fish_speech_core::ar_rotary::fish_rope_coefficients;
use fish_speech_core::graph_ir::QkvPart;
use trtx::{Axes, DataType, ElementWiseOperation, MatrixOperation, NetworkDefinition, Tensor};

use crate::trtx_lowering::LoweringError;

fn trt(error: trtx::Error) -> LoweringError {
    LoweringError::TensorRt(error.to_string())
}

fn require_dimensions(
    network: &mut NetworkDefinition<'_>,
    input: &Tensor<'_>,
    rank: usize,
    dtype: DataType,
) -> Result<Vec<i64>, LoweringError> {
    let dims = input.dimensions(network).map_err(trt)?;
    if dims.len() != rank
        || input.data_type(network) != dtype
        || dims.contains(&0)
        || dims.iter().any(|v| *v < 0)
    {
        return Err(LoweringError::InvalidBlueprint(
            "AR TensorRT op expects known static positive dimensions and matching dtype".into(),
        ));
    }
    Ok(dims)
}

pub(crate) fn lower_token_embedding<'network>(
    network: &mut NetworkDefinition<'network>,
    ids: &Tensor<'network>,
    weights: &Tensor<'network>,
) -> Result<Tensor<'network>, LoweringError> {
    let index_shape = require_dimensions(network, ids, 2, DataType::kINT32)?;
    let weight_shape = require_dimensions(network, weights, 2, DataType::kFLOAT)?;
    if index_shape[0] < 1 || index_shape[1] < 1 || weight_shape[0] < 1 {
        return Err(LoweringError::InvalidBlueprint(
            "invalid embedding dimensions".into(),
        ));
    }
    let output = network
        .add_gather(weights, ids, 0)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let expected = [index_shape[0], index_shape[1], weight_shape[1]];
    if output.dimensions(network).map_err(trt)? != expected {
        return Err(LoweringError::TensorRt(
            "TRT embedding Gather output shape mismatch".into(),
        ));
    }
    Ok(output)
}

pub(crate) fn lower_qkv_split<'network>(
    network: &mut NetworkDefinition<'network>,
    packed: &Tensor<'network>,
    part: QkvPart,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor<'network>, LoweringError> {
    let dims = require_dimensions(network, packed, 3, DataType::kFLOAT)?;
    if q_heads == 0 || kv_heads == 0 || head_dim == 0 || q_heads % kv_heads != 0 {
        return Err(LoweringError::InvalidBlueprint(
            "invalid QKV head counts".into(),
        ));
    }
    let to_i64 = |n: usize| {
        i64::try_from(n)
            .map_err(|_| LoweringError::UnsupportedType("head dimension exceeds i64".into()))
    };
    let q = to_i64(q_heads)?;
    let kv = to_i64(kv_heads)?;
    let d = to_i64(head_dim)?;
    if dims[2]
        != (q + 2 * kv)
            .checked_mul(d)
            .ok_or_else(|| LoweringError::InvalidBlueprint("QKV width overflow".into()))?
    {
        return Err(LoweringError::InvalidBlueprint(
            "QKV packed tensor width mismatch".into(),
        ));
    }
    let (heads, offset) = match part {
        QkvPart::Query => (q, 0),
        QkvPart::Key => (kv, q * d),
        QkvPart::Value => (kv, (q + kv) * d),
    };
    let sliced = network
        .add_slice(
            packed,
            &[0, 0, offset],
            &[dims[0], dims[1], heads * d],
            &[1, 1, 1],
        )
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let mut layer = network.add_shuffle(&sliced).map_err(trt)?;
    layer
        .set_reshape_dimensions(network, &[dims[0], dims[1], heads, d])
        .map_err(trt)?;
    layer
        .set_second_transpose(network, &[0, 2, 1, 3])
        .map_err(trt)?;
    let result = layer.output(network, 0).map_err(trt)?;
    if result.dimensions(network).map_err(trt)? != [dims[0], heads, dims[1], d] {
        return Err(LoweringError::TensorRt(
            "TRT split/transpose QKV result shape mismatch".into(),
        ));
    }
    Ok(result)
}

pub(crate) fn lower_rotary_interleaved<'network>(
    network: &mut NetworkDefinition<'network>,
    input: &Tensor<'network>,
    position: usize,
    base: f32,
) -> Result<Tensor<'network>, LoweringError> {
    let dims = require_dimensions(network, input, 4, DataType::kFLOAT)?;
    let frames = usize::try_from(dims[2])
        .map_err(|_| LoweringError::InvalidBlueprint("invalid RoPE sequence length".into()))?;
    let head_dim = usize::try_from(dims[3])
        .map_err(|_| LoweringError::InvalidBlueprint("invalid RoPE head dimension".into()))?;
    let coeff = fish_rope_coefficients(position, frames, head_dim, base).map_err(|e| {
        LoweringError::InvalidBlueprint(format!("invalid RoPE coefficients: {e:?}"))
    })?;
    let indices = coeff
        .pair_indices
        .iter()
        .flat_map(|id| id.to_le_bytes())
        .collect::<Vec<_>>();
    let indices = network
        .add_constant_owned(&[dims[3]], indices, DataType::kINT32, None)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let swapped = network
        .add_gather(input, &indices, 3)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let to_bytes = |scalars: Vec<f32>| {
        scalars
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let cos = network
        .add_constant_owned(
            &[1, 1, dims[2], dims[3]],
            to_bytes(coeff.cosine),
            DataType::kFLOAT,
            None,
        )
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let sin = network
        .add_constant_owned(
            &[1, 1, dims[2], dims[3]],
            to_bytes(coeff.signed_sine),
            DataType::kFLOAT,
            None,
        )
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let real = network
        .add_elementwise(input, &cos, ElementWiseOperation::kPROD)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let imag = network
        .add_elementwise(&swapped, &sin, ElementWiseOperation::kPROD)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let output = network
        .add_elementwise(&real, &imag, ElementWiseOperation::kSUM)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    if output.dimensions(network).map_err(trt)? != dims {
        return Err(LoweringError::TensorRt(
            "TRT RoPE result shape mismatch".into(),
        ));
    }
    Ok(output)
}

pub(crate) fn lower_fish_embedding<'network>(
    network: &mut NetworkDefinition<'network>,
    tokens: &Tensor<'network>,
    words: &Tensor<'network>,
    codebook_weights: &Tensor<'network>,
    semantic_membership_lut: &Tensor<'network>,
    num_codebooks: usize,
    codebook_size: usize,
    share_codebook_embeddings: bool,
) -> Result<Tensor<'network>, LoweringError> {
    let ts = require_dimensions(network, tokens, 3, DataType::kINT32)?;
    let ws = require_dimensions(network, words, 2, DataType::kFLOAT)?;
    let cs = require_dimensions(network, codebook_weights, 2, DataType::kFLOAT)?;
    let ls = require_dimensions(network, semantic_membership_lut, 1, DataType::kINT32)?;
    let channels = i64::try_from(num_codebooks + 1)
        .map_err(|_| LoweringError::InvalidBlueprint("codebook count overflow".into()))?;
    let codebook_rows = i64::try_from(num_codebooks)
        .ok()
        .and_then(|n| n.checked_mul(codebook_size as i64))
        .ok_or_else(|| LoweringError::InvalidBlueprint("codebook rows overflow".into()))?;
    if num_codebooks == 0
        || codebook_size == 0
        || ts[1] != channels
        || ws[0] != ls[0]
        || ws[1] != cs[1]
        || cs[0] != codebook_rows
    {
        return Err(LoweringError::InvalidBlueprint(
            "FishEmbedding incompatible shapes".into(),
        ));
    }
    let (batch, frames, dim) = (ts[0], ts[2], ws[1]);
    let reshaped_slice = |network: &mut NetworkDefinition<'network>,
                          index: i64|
     -> Result<Tensor<'network>, LoweringError> {
        let sliced = network
            .add_slice(tokens, &[0, index, 0], &[batch, 1, frames], &[1, 1, 1])
            .map_err(trt)?
            .output(network, 0)
            .map_err(trt)?;
        let mut shuffle = network.add_shuffle(&sliced).map_err(trt)?;
        shuffle
            .set_reshape_dimensions(network, &[batch, frames])
            .map_err(trt)?;
        shuffle.output(network, 0).map_err(trt)
    };
    let word_ids = reshaped_slice(network, 0)?;
    let word_emb = lower_token_embedding(network, &word_ids, words)?;
    let mask_ids = network
        .add_gather(semantic_membership_lut, &word_ids, 0)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let mask_float = network
        .add_cast(&mask_ids, DataType::kFLOAT)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let mut mask_expand = network.add_shuffle(&mask_float).map_err(trt)?;
    mask_expand
        .set_reshape_dimensions(network, &[batch, frames, 1])
        .map_err(trt)?;
    let mask = mask_expand.output(network, 0).map_err(trt)?;
    let mut cb_sum: Option<Tensor<'network>> = None;
    for book in 0..num_codebooks {
        let ids = reshaped_slice(network, (book + 1) as i64)?;
        let ids_with_offset = if share_codebook_embeddings {
            let offset = i32::try_from(book.checked_mul(codebook_size).ok_or_else(|| {
                LoweringError::InvalidBlueprint("codebook offset overflow".into())
            })?)
            .map_err(|_| LoweringError::InvalidBlueprint("codebook offset exceeds INT32".into()))?;
            let constant = network
                .add_small_constant_copied(&[1, 1], &offset.to_le_bytes(), DataType::kINT32, None)
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?;
            network
                .add_elementwise(&ids, &constant, ElementWiseOperation::kSUM)
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?
        } else {
            ids
        };
        let embed = lower_token_embedding(network, &ids_with_offset, codebook_weights)?;
        cb_sum = Some(match cb_sum {
            None => embed,
            Some(acc) => network
                .add_elementwise(&acc, &embed, ElementWiseOperation::kSUM)
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?,
        });
    }
    let masked = network
        .add_elementwise(
            &cb_sum
                .ok_or_else(|| LoweringError::InvalidBlueprint("no acoustic codebooks".into()))?,
            &mask,
            ElementWiseOperation::kPROD,
        )
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let output = network
        .add_elementwise(&word_emb, &masked, ElementWiseOperation::kSUM)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    if output.dimensions(network).map_err(trt)? != [batch, frames, dim] {
        return Err(LoweringError::TensorRt(
            "FishEmbedding produced incorrect dimensions".into(),
        ));
    }
    Ok(output)
}

/// Native static-profile GQA: Q [B,Hq,T,D], K/V [B,Hkv,S,D].
/// Matmul broadcasts the singleton group dimension instead of explicitly
/// allocating repeated K/V heads. No physical KV capacity is included here.
pub(crate) fn lower_gqa_attention<'network>(
    network: &mut NetworkDefinition<'network>,
    query: &Tensor<'network>,
    keys: &Tensor<'network>,
    values: &Tensor<'network>,
    position: usize,
    query_heads: usize,
    kv_heads: usize,
) -> Result<Tensor<'network>, LoweringError> {
    let q = require_dimensions(network, query, 4, DataType::kFLOAT)?;
    let k = require_dimensions(network, keys, 4, DataType::kBF16)?;
    let v = require_dimensions(network, values, 4, DataType::kBF16)?;
    if k != v
        || q[0] != k[0]
        || q[3] != k[3]
        || query_heads == 0
        || kv_heads == 0
        || query_heads % kv_heads != 0
        || q[1] != query_heads as i64
        || k[1] != kv_heads as i64
        || position.checked_add(q[2] as usize) != Some(k[2] as usize)
    {
        return Err(LoweringError::InvalidBlueprint(
            "GQA expects right-aligned FP32 Q and BF16 active-prefix K/V".into(),
        ));
    }
    let shape = GqaShape {
        batch: q[0] as usize,
        query_heads,
        kv_heads,
        query_len: q[2] as usize,
        kv_len: k[2] as usize,
        head_dim: q[3] as usize,
        first_position: position,
    };
    let mask = right_aligned_causal_bias(shape)
        .map_err(|e| LoweringError::InvalidBlueprint(format!("invalid GQA mask: {e:?}")))?;
    let groups = (query_heads / kv_heads) as i64;
    let reshape = |network: &mut NetworkDefinition<'network>,
                   input: &Tensor<'network>,
                   dims: &[i64]|
     -> Result<Tensor<'network>, LoweringError> {
        let mut layer = network.add_shuffle(input).map_err(trt)?;
        layer.set_reshape_dimensions(network, dims).map_err(trt)?;
        layer.output(network, 0).map_err(trt)
    };
    let qg = reshape(network, query, &[q[0], k[1], groups, q[2], q[3]])?;
    let k_f32 = network
        .add_cast(keys, DataType::kFLOAT)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let v_f32 = network
        .add_cast(values, DataType::kFLOAT)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let kg = reshape(network, &k_f32, &[k[0], k[1], 1, k[2], k[3]])?;
    let vg = reshape(network, &v_f32, &[v[0], v[1], 1, v[2], v[3]])?;
    let logits = network
        .add_matrix_multiply(
            &qg,
            MatrixOperation::kNONE,
            &kg,
            MatrixOperation::kTRANSPOSE,
        )
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let scalar = (q[3] as f32).sqrt().recip();
    let scale = network
        .add_small_constant_copied(
            &[1, 1, 1, 1, 1],
            &scalar.to_le_bytes(),
            DataType::kFLOAT,
            None,
        )
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let scaled = network
        .add_elementwise(&logits, &scale, ElementWiseOperation::kPROD)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let masked = if q[2] == 1 {
        scaled
    } else {
        let bias = network
            .add_constant_owned(
                &[1, 1, 1, q[2], k[2]],
                mask.into_iter().flat_map(|val| val.to_le_bytes()).collect(),
                DataType::kFLOAT,
                None,
            )
            .map_err(trt)?
            .output(network, 0)
            .map_err(trt)?;
        network
            .add_elementwise(&scaled, &bias, ElementWiseOperation::kSUM)
            .map_err(trt)?
            .output(network, 0)
            .map_err(trt)?
    };
    let probs = network
        .add_softmax(&masked, Axes::new([4]))
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let attended = network
        .add_matrix_multiply(&probs, MatrixOperation::kNONE, &vg, MatrixOperation::kNONE)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let mut restore = network.add_shuffle(&attended).map_err(trt)?;
    restore
        .set_first_transpose(network, &[0, 3, 1, 2, 4])
        .map_err(trt)?;
    restore
        .set_reshape_dimensions(network, &[q[0], q[2], q[1] * q[3]])
        .map_err(trt)?;
    let result = restore.output(network, 0).map_err(trt)?;
    if result.dimensions(network).map_err(trt)? != [q[0], q[2], q[1] * q[3]] {
        return Err(LoweringError::TensorRt(
            "GQA native grouped attention result shape mismatch".into(),
        ));
    }
    Ok(result)
}
