use anyhow::{Result, ensure};
use std::collections::HashMap;
use crate::{DType, PickleValue, Tensor, dictionary, require_tensor};

fn validate_block(tensors: &HashMap<String, Tensor>, prefix: &str, width: usize, feed_forward: usize) -> Result<()> {
    for (suffix, dims) in [
        ("self_attn.in_proj_weight", vec![3 * width, width]),
        ("self_attn.in_proj_bias", vec![3 * width]),
        ("self_attn.out_proj.weight", vec![width, width]),
        ("self_attn.out_proj.bias", vec![width]),
        ("conv1.weight", vec![feed_forward, width, 9]),
        ("conv1.bias", vec![feed_forward]),
        ("conv2.weight", vec![width, feed_forward, 1]),
        ("conv2.bias", vec![width]),
        ("norm1.weight", vec![width]), ("norm1.bias", vec![width]),
        ("norm2.weight", vec![width]), ("norm2.bias", vec![width]),
    ] { require_tensor(tensors, &format!("{prefix}.{suffix}"), DType::F32, &dims)?; }
    Ok(())
}

fn validate_position(tensors: &HashMap<String, Tensor>, prefix: &str, width: usize) -> Result<()> {
    for (suffix, dims) in [("pos_encoder.pe", vec![5000, 1, width]), ("pos_encoder.scale", vec![1]),
        ("norm.weight", vec![width]), ("norm.bias", vec![width])] {
        require_tensor(tensors, &format!("{prefix}.{suffix}"), DType::F32, &dims)?;
    }
    Ok(())
}

pub(crate) fn validate(tensors: &HashMap<String, Tensor>, metadata: &PickleValue, symbols: usize, speakers: usize) -> Result<()> {
    ensure!(tensors.len() == 363, "Unsupported V3 tensor count: {}", tensors.len());
    let required = |name: &str, dims: &[usize]| require_tensor(tensors, &format!("tts_model.{name}"), DType::F32, dims);
    for prefix in ["dur_predictor.dur_pred", "pitch_predictor.pitch_pred"] {
        for (suffix, dims) in [("embedding.weight", vec![symbols, 64]), ("speaker_embedding.weight", vec![speakers, 64]),
            ("lin.weight", vec![1, 64]), ("lin.bias", vec![1])] {
            required(&format!("{prefix}.{suffix}"), &dims)?;
        }
        let stack = format!("tts_model.{prefix}.transformer");
        validate_position(tensors, &stack, 64)?;
        for i in 0..4 { validate_block(tensors, &format!("{stack}.layers.{i}"), 64, 64)?; }
    }
    for (name, dims) in [
        ("tacotron.embedding.weight", vec![symbols, 128]), ("tacotron.speaker_embedding.weight", vec![speakers, 128]),
        ("tacotron.pitch_proj.weight", vec![128, 1, 3]), ("tacotron.pitch_proj.bias", vec![128]),
        ("tacotron.lin.weight", vec![128, 128]), ("tacotron.lin.bias", vec![128]),
        ("tacotron.decoder.upsample.proj.weight", vec![384, 128]), ("tacotron.decoder.upsample.proj.bias", vec![384]),
        ("pitch_predictor.pitch_proj.weight", vec![256, 1, 3]), ("pitch_predictor.pitch_proj.bias", vec![256]),
        ("vocoder.pqmf_layer.H", vec![4, 1, 63]), ("vocoder.pqmf_layer.G", vec![1, 4, 63]),
        ("vocoder.pqmf_layer.updown_filter", vec![4, 4, 4]),
    ] { required(name, &dims)?; }
    for prefix in ["tts_model.tacotron.encoder", "tts_model.tacotron.decoder"] { validate_position(tensors, prefix, 128)?; }
    for i in 0..4 { validate_block(tensors, &format!("tts_model.tacotron.encoder.layers.{i}"), 128, 512)?; }
    for layer in ["pre_vanilla_layers.0", "shorten_layers.0", "shorten_layers.1", "post_vanilla_layers.0"] {
        validate_block(tensors, &format!("tts_model.tacotron.decoder.{layer}"), 128, 512)?;
    }
    for (name, input, output) in [("conv_pre.0", 128, 256), ("conv_pre.1", 128, 512), ("conv_post", 64, 4), ("conv_post_u", 64, 4)] {
        required(&format!("vocoder.{name}.weight"), &[output, input, 7])?;
        required(&format!("vocoder.{name}.bias"), &[output])?;
    }
    for (suffix, channels, strides) in [("_8", &[256,128,64][..], &[5,5][..]), ("", &[512,256,128,64][..], &[3,5,5][..]), ("_u", &[512,256,128,64,64][..], &[3,5,5,2][..])] {
        for (i, &stride) in strides.iter().enumerate() {
            let (input, output) = (channels[i], channels[i+1]);
            required(&format!("vocoder.ups{suffix}.{i}.weight"), &[input, output, stride*2])?;
            required(&format!("vocoder.ups{suffix}.{i}.bias"), &[output])?;
            for (j, kernel) in [3, 7, 11].into_iter().enumerate() {
                for k in 0..2 {
                    let name = format!("vocoder.resblocks{suffix}.{}.convs.{k}", 3*i+j);
                    required(&format!("{name}.weight"), &[output, output, kernel])?;
                    required(&format!("{name}.bias"), &[output])?;
                }
            }
        }
    }
    let script = dictionary(metadata, "script metadata")?;
    let get_dict = |key: &str| -> Result<&HashMap<String, PickleValue>> {
        let mut node = script;
        for segment in key.split('.') {
            let value = node.get(segment).ok_or_else(|| anyhow::anyhow!("Missing V3 metadata: {key}"))?;
            node = dictionary(value, key)?;
        }
        Ok(node)
    };
    let tacotron = get_dict("tts_model.tacotron")?;
    let decoder = get_dict("tts_model.tacotron.decoder")?;
    let vocoder = get_dict("tts_model.vocoder")?;
    let pqmf = get_dict("tts_model.vocoder.pqmf_layer")?;
    ensure!(matches!(tacotron.get("pitch_strength"), Some(PickleValue::Float(v)) if *v == 1.0), "Unsupported V3 pitch strength");
    ensure!(matches!(decoder.get("shorten_factor"), Some(PickleValue::Int(3))), "Unsupported V3 HourGlass factor");
    ensure!(matches!(vocoder.get("num_kernels"), Some(PickleValue::Int(3))) && matches!(vocoder.get("multiband"), Some(PickleValue::Bool(true))) && matches!(vocoder.get("resample"), Some(PickleValue::Int(1))), "Unsupported V3 vocoder metadata");
    ensure!(matches!(pqmf.get("N"), Some(PickleValue::Int(4))) && matches!(pqmf.get("taps"), Some(PickleValue::Int(62))), "Unsupported V3 PQMF metadata");
    Ok(())
}
