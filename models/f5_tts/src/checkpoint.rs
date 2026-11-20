use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use crate::{F5Error, Result};

const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub source_key: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub first_byte: u64,
    pub byte_len: u64,
}

#[derive(Debug)]
pub struct Checkpoint {
    path: PathBuf,
    payload_offset: u64,
    tensors: BTreeMap<String, TensorInfo>,
}

fn invalid(message: impl Into<String>) -> F5Error {
    F5Error::InvalidCheckpoint(message.into())
}

fn bytes_per_element(dtype: &str) -> Option<u64> {
    match dtype {
        "F16" | "BF16" | "I16" | "U16" => Some(2),
        "F32" | "I32" | "U32" => Some(4),
        "F64" | "I64" | "U64" => Some(8),
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" => Some(1),
        _ => None,
    }
}

impl Checkpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path).map_err(|error| invalid(error.to_string()))?;
        let total_bytes = file
            .metadata()
            .map_err(|error| invalid(error.to_string()))?
            .len();
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|error| invalid(error.to_string()))?;
        let header_bytes = u64::from_le_bytes(prefix);
        if header_bytes == 0
            || header_bytes > MAX_HEADER_BYTES
            || header_bytes > total_bytes.saturating_sub(8)
        {
            return Err(invalid("Invalid safetensors header length"));
        }
        let mut header = vec![0; header_bytes as usize];
        file.read_exact(&mut header)
            .map_err(|error| invalid(error.to_string()))?;
        let json: serde_json::Value =
            serde_json::from_slice(&header).map_err(|error| invalid(error.to_string()))?;
        let root = json
            .as_object()
            .ok_or_else(|| invalid("Safetensors header must be an object"))?;
        let payload_offset = header_bytes + 8;
        let payload_len = total_bytes - payload_offset;
        let mut tensors = BTreeMap::new();
        for (key, value) in root {
            if key == "__metadata__" {
                continue;
            }
            let shape = value
                .get("shape")
                .and_then(|value| value.as_array())
                .ok_or_else(|| invalid(format!("Missing shape for {key}")))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(|| invalid("Invalid tensor dimension"))
                })
                .collect::<Result<Vec<_>>>()?;
            let dtype = value
                .get("dtype")
                .and_then(|value| value.as_str())
                .ok_or_else(|| invalid(format!("Missing dtype for {key}")))?;
            let width = bytes_per_element(dtype)
                .ok_or_else(|| invalid(format!("Unsupported tensor dtype {dtype}")))?;
            let offsets = value
                .get("data_offsets")
                .and_then(|value| value.as_array())
                .filter(|items| items.len() == 2)
                .ok_or_else(|| invalid(format!("Missing data offsets for {key}")))?;
            let start = offsets[0]
                .as_u64()
                .ok_or_else(|| invalid("Invalid data start"))?;
            let end = offsets[1]
                .as_u64()
                .ok_or_else(|| invalid("Invalid data end"))?;
            let expected = shape
                .iter()
                .try_fold(width, |acc, &dim| acc.checked_mul(dim as u64))
                .ok_or_else(|| invalid("Tensor byte count overflow"))?;
            if start > end || end > payload_len || end - start != expected {
                return Err(invalid(format!(
                    "Tensor {key} has invalid byte offsets or shape"
                )));
            }
            let normalized = key.strip_prefix("ema_model.").unwrap_or(key).to_owned();
            let info = TensorInfo {
                source_key: key.to_owned(),
                dtype: dtype.to_owned(),
                shape,
                first_byte: start,
                byte_len: end - start,
            };
            if tensors.insert(normalized.clone(), info).is_some() {
                return Err(invalid(format!("Duplicate normalized weight {normalized}")));
            }
        }
        if tensors.is_empty() {
            return Err(invalid("Checkpoint has no tensors"));
        }
        Ok(Self {
            path,
            payload_offset,
            tensors,
        })
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }

    pub fn require_prefixes(&self, prefixes: &[&str]) -> Result<()> {
        for prefix in prefixes {
            if !self.tensors.keys().any(|key| key.starts_with(prefix)) {
                return Err(invalid(format!(
                    "Required model weights not found: {prefix}"
                )));
            }
        }
        Ok(())
    }

    pub fn validate_dit_v1(&self) -> Result<()> {
        self.require_prefixes(&[
            "transformer.time_embed.",
            "transformer.text_embed.",
            "transformer.input_embed.",
            "transformer.transformer_blocks.0.",
            "transformer.transformer_blocks.21.",
            "transformer.norm_out.",
            "transformer.proj_out.",
        ])
    }

    pub fn read_tensor(&self, normalized_name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(normalized_name)
            .ok_or_else(|| invalid(format!("Weight not found: {normalized_name}")))?;
        let mut file = File::open(&self.path).map_err(|error| invalid(error.to_string()))?;
        file.seek(SeekFrom::Start(self.payload_offset + info.first_byte))
            .map_err(|error| invalid(error.to_string()))?;
        let length =
            usize::try_from(info.byte_len).map_err(|_| invalid("Tensor exceeds address space"))?;
        let mut result = vec![0u8; length];
        file.read_exact(&mut result)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn inspect_and_load_ema_without_loading_entire_checkpoint() {
        let path =
            std::env::temp_dir().join(format!("f5-tts-test-{}.safetensors", std::process::id()));
        let header = br#"{"ema_model.transformer.proj_out.bias":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut file = File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&1.25_f32.to_le_bytes()).unwrap();
        file.write_all(&3.0_f32.to_le_bytes()).unwrap();
        file.flush().unwrap();
        let model = Checkpoint::open(&path).unwrap();
        assert_eq!(model.tensors()["transformer.proj_out.bias"].shape, [2]);
        assert_eq!(
            model
                .read_tensor("transformer.proj_out.bias")
                .unwrap()
                .len(),
            8
        );
        drop(file);
        std::fs::remove_file(path).unwrap();
    }
}
