use anyhow::{Context, Result, bail, ensure};
use burn::tensor::{Device, Tensor, TensorData};
use burn_store::{ModuleStore, PytorchStore};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use std::path::Path;

pub enum Weights {
    Safe(Vec<u8>),
    Torch(PytorchStore),
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        match path.extension().and_then(|v| v.to_str()) {
            Some("safetensors") => Ok(Self::Safe(
                std::fs::read(path).with_context(|| format!("reading {}", path.display()))?,
            )),
            Some("ckpt" | "pth" | "pt") => Ok(Self::Torch(
                PytorchStore::from_file(path)
                    .with_top_level_key("state_dict")
                    .map_indices_contiguous(false),
            )),
            _ => bail!("expected .safetensors, .ckpt, .pth or .pt file"),
        }
    }

    pub fn values(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        match self {
            Self::Safe(bytes) => {
                let store = SafeTensors::deserialize(bytes)?;
                let tensor = store
                    .tensor(name)
                    .with_context(|| format!("missing weight: {name}"))?;
                let shape = tensor.shape().to_vec();
                Ok((shape, decode_bytes(tensor.data(), tensor.dtype())?))
            }
            Self::Torch(store) => {
                let tensor = store
                    .get_tensor(name)?
                    .with_context(|| format!("missing weight: {name}"))?;
                let shape = tensor.shape.as_slice().to_vec();
                let raw = tensor.to_bytes()?;
                let dtype = format!("{:?}", tensor.dtype);
                let dtype = match dtype.as_str() {
                    "F32" => Dtype::F32,
                    "F16" => Dtype::F16,
                    "BF16" => Dtype::BF16,
                    "F64" => Dtype::F64,
                    _ => bail!("unsupported PyTorch dtype {dtype} for {name}"),
                };
                Ok((shape, decode_bytes(raw.as_ref(), dtype)?))
            }
        }
    }

    pub fn tensor<const D: usize>(&mut self, name: &str, device: &Device) -> Result<Tensor<D>> {
        let (shape, values) = self.values(name)?;
        ensure!(
            shape.len() == D,
            "{name}: expected rank {D}, got {:?}",
            shape
        );
        let dims: [usize; D] = shape
            .try_into()
            .map_err(|_| anyhow::anyhow!("invalid tensor rank"))?;
        Ok(Tensor::<D>::from_data(
            TensorData::new(values, dims),
            device,
        ))
    }
}

fn decode_bytes(bytes: &[u8], dtype: Dtype) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            ensure!(bytes.len() % 4 == 0, "invalid float32 byte length");
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        Dtype::F16 => {
            ensure!(bytes.len() % 2 == 0, "invalid float16 byte length");
            Ok(bytes
                .chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32())
                .collect())
        }
        Dtype::BF16 => {
            ensure!(bytes.len() % 2 == 0, "invalid bfloat16 byte length");
            Ok(bytes
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32())
                .collect())
        }
        Dtype::F64 => {
            ensure!(bytes.len() % 8 == 0, "invalid float64 byte length");
            Ok(bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect())
        }
        other => bail!("unsupported checkpoint tensor dtype: {other:?}"),
    }
}
