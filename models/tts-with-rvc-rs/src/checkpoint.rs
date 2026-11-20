use burn::tensor::{Device, Tensor, TensorData};
use half::{bf16, f16};
use pytorch_reader::{DType, PytorchReader};
use std::{convert::TryInto, path::Path};

pub struct RvcCheckpoint {
    reader: PytorchReader,
}

impl RvcCheckpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let reader = PytorchReader::with_top_level_key(path, "weight")
            .map_err(|error| format!("RVC checkpoint: {error}"))?;
        if reader.is_empty() {
            return Err("RVC checkpoint contains no weight tensors".into());
        }
        Ok(Self { reader })
    }

    pub fn open_root(path: impl AsRef<Path>) -> Result<Self, String> {
        Self::from_reader(PytorchReader::new(path).map_err(|error| format!("checkpoint: {error}"))?)
    }

    pub fn open_model(path: impl AsRef<Path>) -> Result<Self, String> {
        Self::from_reader(
            PytorchReader::with_top_level_key(path, "model")
                .map_err(|error| format!("model checkpoint: {error}"))?,
        )
    }

    fn from_reader(reader: PytorchReader) -> Result<Self, String> {
        if reader.is_empty() {
            return Err("checkpoint contains no tensors".into());
        }
        Ok(Self { reader })
    }

    pub fn tensor_count(&self) -> usize {
        self.reader.len()
    }

    pub fn keys(&self) -> Vec<String> {
        let mut keys = self.reader.keys();
        keys.sort();
        keys
    }

    pub fn contains(&self, name: &str) -> bool {
        self.reader.get(name).is_some()
    }

    pub fn tensor_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        let tensor = self.reader.get(name).ok_or_else(|| format!("tensor missing: {name}"))?;
        let bytes = tensor.read().map_err(|error| format!("cannot read {name}: {error}"))?;
        let data = match tensor.dtype() {
            DType::F32 => bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect(),
            DType::F64 => bytes.chunks_exact(8).map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()) as f32).collect(),
            DType::F16 => bytes.chunks_exact(2).map(|chunk| f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32()).collect(),
            DType::BF16 => bytes.chunks_exact(2).map(|chunk| bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32()).collect(),
            dtype => return Err(format!("unsupported weight dtype for {name}: {dtype:?}")),
        };
        Ok((tensor.shape().to_vec(), data))
    }

    pub fn tensor<const D: usize>(&self, name: &str, device: &Device) -> Result<Tensor<D>, String> {
        let (shape, data) = self.tensor_f32(name)?;
        let dims: [usize; D] = shape.try_into().map_err(|shape: Vec<usize>| {
            format!("{name}: expected rank {D}, found shape {shape:?}")
        })?;
        Ok(Tensor::from_data(TensorData::new(data, dims), device))
    }

    pub fn weight_norm_conv1d_values(&self, prefix: &str) -> Result<([usize; 3], Vec<f32>), String> {
        let (shape, mut weights) = self.tensor_f32(&format!("{prefix}.weight_v"))?;
        let [out_channels, input_channels, kernel]: [usize; 3] = shape
            .try_into().map_err(|shape: Vec<usize>| format!("{prefix}: expected Conv1d rank 3, got {shape:?}"))?;
        let channel_elements = input_channels * kernel;
        let (g_shape, gains) = self.tensor_f32(&format!("{prefix}.weight_g"))?;
        if (g_shape != [out_channels, 1, 1] && g_shape != [out_channels]) || gains.len() != out_channels {
            return Err(format!("{prefix}: invalid weight_g shape {g_shape:?}"));
        }
        for (gain, row) in gains.into_iter().zip(weights.chunks_exact_mut(channel_elements)) {
            let norm = row.iter().map(|weight| weight * weight).sum::<f32>().sqrt();
            if norm == 0.0 || !norm.is_finite() {
                return Err(format!("{prefix}: invalid weight_norm norm"));
            }
            for weight in row { *weight *= gain / norm; }
        }
        Ok(([out_channels, input_channels, kernel], weights))
    }

    pub fn weight_norm_conv1d(&self, prefix: &str, device: &Device) -> Result<Tensor<3>, String> {
        let (shape, weights) = self.weight_norm_conv1d_values(prefix)?;
        Ok(Tensor::from_data(TensorData::new(weights, shape), device))
    }

    pub fn weight_norm_conv1d_dim2_values(&self, prefix: &str) -> Result<([usize; 3], Vec<f32>), String> {
        let (shape, mut weights) = self.tensor_f32(&format!("{prefix}.weight_v"))?;
        if shape.len() != 3 || shape.contains(&0) {
            return Err(format!("{prefix}: expected nonempty rank-3 positional Conv1d, found {shape:?}"));
        }
        let (out_channels, input_per_group, kernel) = (shape[0], shape[1], shape[2]);
        let (g_shape, gains) = self.tensor_f32(&format!("{prefix}.weight_g"))?;
        if g_shape != [1, 1, kernel] && g_shape != [kernel] {
            return Err(format!("{prefix}: expected kernel-axis weight_g [1,1,{kernel}], found {g_shape:?}"));
        }
        let mut squared_norms = vec![0.0f32; kernel];
        for row in weights.chunks_exact(kernel) {
            for (index, value) in row.iter().enumerate() {
                squared_norms[index] += value * value;
            }
        }
        let factors: Result<Vec<_>, String> = squared_norms.iter().zip(gains.iter()).map(|(&sq, &gain)| {
            if !sq.is_finite() || sq <= 0.0 || !gain.is_finite() {
                return Err(format!("{prefix}: invalid kernel-axis weight_norm values"));
            }
            Ok(gain / sq.sqrt())
        }).collect();
        let factors = factors?;
        for row in weights.chunks_exact_mut(kernel) {
            for (value, gain) in row.iter_mut().zip(factors.iter()) {
                *value *= gain;
            }
        }
        Ok(([out_channels, input_per_group, kernel], weights))
    }

    pub fn weight_norm_conv1d_dim2(&self, prefix: &str, device: &Device) -> Result<Tensor<3>, String> {
        let (shape, weights) = self.weight_norm_conv1d_dim2_values(prefix)?;
        Ok(Tensor::from_data(TensorData::new(weights, shape), device))
    }

}
