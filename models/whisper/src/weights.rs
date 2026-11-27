use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::Path;

pub struct RawTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct Weights {
    map: HashMap<String, RawTensor>,
    prefix: String,
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    let v = match exp {
        0 => man * 2f32.powi(-24),
        31 => {
            if man == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    };
    if sign == 1 { -v } else { v }
}

impl Weights {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            bail!("not a safetensors file (too short)");
        }
        let n = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
        let end = n
            .checked_add(8)
            .context("corrupt safetensors header length")?;
        let header = bytes.get(8..end).context("truncated safetensors header")?;
        let header: serde_json::Value = serde_json::from_slice(header)?;
        let data = &bytes[end..];

        let mut map = HashMap::new();
        for (name, info) in header.as_object().context("bad header")? {
            if name == "__metadata__" {
                continue;
            }
            let dtype = info["dtype"].as_str().context("no dtype")?;
            let shape: Vec<usize> = info["shape"]
                .as_array()
                .context("no shape")?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = info["data_offsets"].as_array().context("no offsets")?;
            let (a, b) = (
                off[0].as_u64().context("offset")? as usize,
                off[1].as_u64().context("offset")? as usize,
            );
            let raw = data
                .get(a..b)
                .with_context(|| format!("tensor {name} out of range"))?;
            let elem = match dtype {
                "F32" => 4,
                "F16" | "BF16" => 2,
                _ => 0,
            };
            if elem != 0 && raw.len() != shape.iter().product::<usize>() * elem {
                bail!(
                    "tensor {name}: {} bytes do not match shape {shape:?} ({dtype})",
                    raw.len()
                );
            }
            let values: Vec<f32> = match dtype {
                "F32" => raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                "F16" => raw
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
                "BF16" => raw
                    .chunks_exact(2)
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect(),
                _ => continue,
            };
            map.insert(
                name.clone(),
                RawTensor {
                    shape,
                    data: values,
                },
            );
        }

        let prefix = if map.contains_key("model.encoder.conv1.weight") {
            "model."
        } else {
            ""
        };
        Ok(Self {
            map,
            prefix: prefix.to_string(),
        })
    }

    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(&format!("{}{}", self.prefix, name))
    }

    pub fn take(&mut self, name: &str) -> Result<RawTensor> {
        let key = format!("{}{}", self.prefix, name);
        self.map
            .remove(&key)
            .with_context(|| format!("missing tensor `{key}` in safetensors"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_f32_tensor() {
        let header =
            br#"{"model.encoder.conv1.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header);
        file.extend_from_slice(&1.5f32.to_le_bytes());
        file.extend_from_slice(&(-2.0f32).to_le_bytes());
        let mut w = Weights::from_bytes(&file).unwrap();
        assert!(w.has("encoder.conv1.weight"));
        let t = w.take("encoder.conv1.weight").unwrap();
        assert_eq!(t.shape, vec![2]);
        assert_eq!(t.data, vec![1.5, -2.0]);
    }

    #[test]
    fn rejects_size_mismatch_and_bad_header() {
        let header = br#"{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header);
        file.extend_from_slice(&[0u8; 8]);
        assert!(Weights::from_bytes(&file).is_err());
        let mut bad = u64::MAX.to_le_bytes().to_vec();
        bad.extend_from_slice(b"{}");
        assert!(Weights::from_bytes(&bad).is_err());
    }

    #[test]
    fn f16_conversion() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333251953125);
    }
}
