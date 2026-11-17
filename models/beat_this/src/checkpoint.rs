use burn::tensor::{Device, Tensor, TensorData};
use pytorch_reader::{DType, PytorchReader};
use std::{convert::TryInto, path::Path};

pub struct Checkpoint {
    reader: PytorchReader,
}

impl Checkpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let reader = PytorchReader::with_top_level_key(path, "state_dict")
            .map_err(|e| format!("cannot open Lightning checkpoint: {e}"))?;
        if reader.get("model.frontend.stem.conv2d.weight").is_none() {
            return Err("checkpoint does not contain Beat This! state_dict".into());
        }
        Ok(Self { reader })
    }

    pub fn count(&self) -> usize {
        self.reader.len()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.reader.get(&format!("model.{name}")).is_some()
    }

    pub fn values(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        let full = format!("model.{name}");
        let tensor = self
            .reader
            .get(&full)
            .ok_or_else(|| format!("missing tensor: {full}"))?;
        let shape = tensor.shape().to_vec();
        let bytes = tensor.read().map_err(|e| format!("{full}: {e}"))?;
        let values = match tensor.dtype() {
            DType::F32 => bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect(),
            DType::F64 => bytes
                .chunks_exact(8)
                .map(|v| f64::from_le_bytes(v.try_into().unwrap()) as f32)
                .collect(),
            d => return Err(format!("{full}: unsupported dtype {d:?}")),
        };
        Ok((shape, values))
    }

    pub fn tensor<const N: usize>(&self, name: &str, device: &Device) -> Result<Tensor<N>, String> {
        let (shape, data) = self.values(name)?;
        let shape: [usize; N] = shape
            .try_into()
            .map_err(|s: Vec<usize>| format!("{name}: expected {N} dimensions, found {s:?}"))?;
        Ok(Tensor::from_data(TensorData::new(data, shape), device))
    }

    pub fn dimension(&self, name: &str) -> Result<usize, String> {
        self.values(name)?
            .0
            .first()
            .copied()
            .ok_or_else(|| format!("{name} is scalar"))
    }
}
