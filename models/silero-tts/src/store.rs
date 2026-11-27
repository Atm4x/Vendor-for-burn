use anyhow::{Context, Result, bail};
use burn::tensor::{Device, Tensor, TensorData};
use safetensors::{SafeTensors, tensor::Dtype};
use std::{fs, path::Path};

/// Self-contained weights accessor: data is held locally, and individual
/// tensors are uploaded to the selected CubeCL backend only on demand.
pub enum TensorStore {
    Prepared(Vec<u8>),
    Native(silero_package::WeightSet),
}

impl TensorStore {
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        SafeTensors::deserialize(&bytes).context("invalid safetensors")?;
        Ok(Self::Prepared(bytes))
    }

    pub fn native(weights: silero_package::WeightSet) -> Self { Self::Native(weights) }

    pub fn names(&self) -> Result<Vec<String>> {
        match self {
            Self::Prepared(bytes) => Ok(SafeTensors::deserialize(bytes)?.names().into_iter().map(str::to_owned).collect()),
            Self::Native(weights) => Ok(weights.tensors().keys().cloned().collect()),
        }
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        match self {
            Self::Prepared(_) => Ok(self.tensor(name)?.shape().to_vec()),
            Self::Native(weights) => Ok(weights.shape(name)?.to_vec()),
        }
    }

    fn tensor(&self, name: &str) -> Result<safetensors::tensor::TensorView<'_>> {
        let Self::Prepared(bytes) = self else { bail!("not a prepared tensor store") };
        Ok(SafeTensors::deserialize(bytes)?.tensor(name)?)
    }

    pub fn float<const D: usize>(&self, name: &str, device: &Device) -> Result<Tensor<D>> {
        let shape = self.shape(name)?;
        let dims: [usize; D] = shape.as_slice().try_into().map_err(|_| {
            anyhow::anyhow!("Incorrect rank for {name}: expected {D}")
        })?;
        Ok(Tensor::from_data(TensorData::new(self.float_data(name)?, dims), device))
    }

    pub fn float_data(&self, name: &str) -> Result<Vec<f32>> {
        if let Self::Native(weights) = self { return weights.read_f32(name); }
        let tensor = self.tensor(name)?;
        if tensor.dtype() != Dtype::F32 {
            bail!("Expected float32 {name}")
        }
        Ok(tensor.data().chunks_exact(4).map(|ch| f32::from_le_bytes(ch.try_into().unwrap())).collect())
    }
}
