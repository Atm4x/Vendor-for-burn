use anyhow::{Context, Result, bail};
use burn::tensor::{Device, Tensor, TensorData};
use safetensors::{SafeTensors, tensor::Dtype};
use std::{fs, path::Path};

/// Self-contained weights accessor: data is held locally, and individual
/// tensors are uploaded to the selected CubeCL backend only on demand.
pub struct TensorStore {
    bytes: Vec<u8>,
}

impl TensorStore {
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        SafeTensors::deserialize(&bytes).context("invalid safetensors")?;
        Ok(Self { bytes })
    }

    pub fn names(&self) -> Result<Vec<String>> {
        Ok(SafeTensors::deserialize(&self.bytes)?.names().into_iter().map(str::to_owned).collect())
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.tensor(name)?.shape().to_vec())
    }

    fn tensor(&self, name: &str) -> Result<safetensors::tensor::TensorView<'_>> {
        Ok(SafeTensors::deserialize(&self.bytes)?.tensor(name)?)
    }

    pub fn float<const D: usize>(&self, name: &str, device: &Device) -> Result<Tensor<D>> {
        let tensor = self.tensor(name)?;
        let dims: [usize; D] = tensor.shape().try_into().map_err(|_| {
            anyhow::anyhow!("Incorrect rank for {name}: expected {D}")
        })?;
        let data: Vec<f32> = match tensor.dtype() {
            Dtype::F32 => tensor.data().chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect(),
            Dtype::F16 => bail!("Half precision weight {name} requires conversion to f32 during export"),
            dtype => bail!("Tensor {name} has dtype {dtype:?}, expected f32"),
        };
        Ok(Tensor::from_data(TensorData::new(data, dims), device))
    }

    pub fn float_data(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self.tensor(name)?;
        if tensor.dtype() != Dtype::F32 {
            bail!("Expected float32 {name}")
        }
        Ok(tensor.data().chunks_exact(4).map(|ch| f32::from_le_bytes(ch.try_into().unwrap())).collect())
    }
}
