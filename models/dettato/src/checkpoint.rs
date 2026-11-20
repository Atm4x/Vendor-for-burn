use burn::tensor::{Device, Tensor, TensorData};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use std::{convert::TryInto, fs, path::{Path, PathBuf}};

pub const STAGES: [(&str, &str); 4] = [
    ("transcriber", "dettato"),
    ("evidence", "dettato-evidence"),
    ("grouper", "dettato-grouper"),
    ("drums", "dettato-drums"),
];

pub struct ModelPaths {
    pub config: PathBuf,
    pub weights: PathBuf,
}

impl ModelPaths {
    /// Resolve either the upstream Hugging Face directory layout or a flat
    /// directory of the exact artifacts supplied by the user (including the
    /// historical `detatto-evidence` typo).
    pub fn in_directory(root: impl AsRef<Path>, stage: &str) -> Result<Self, String> {
        let name = STAGES.iter().find(|(id, _)| *id == stage)
            .map(|(_, name)| *name)
            .ok_or_else(|| format!("unknown Dettato stage: {stage}"))?;
        let root = root.as_ref();
        let mut bases = vec![root.to_path_buf()];
        if !root.ends_with("dettato") {
            bases.push(root.join("dettato"));
        }
        for base in &bases {
            let canonical = base.join(name);
            let paths = Self {
                config: canonical.join("config.json"),
                weights: canonical.join("model.safetensors"),
            };
            if paths.config.is_file() && paths.weights.is_file() { return Ok(paths); }
            let aliases = if stage == "evidence" {
                vec![name.to_owned(), "detatto-evidence".to_owned()]
            } else { vec![name.to_owned()] };
            for alias in aliases {
                let paths = Self {
                    config: base.join(format!("{alias}-config.json")),
                    weights: base.join(format!("{alias}.safetensors")),
                };
                if paths.config.is_file() && paths.weights.is_file() { return Ok(paths); }
            }
        }
        // Retain a deterministic canonical path when files are not yet present;
        // actual IO will report the missing file and its location.
        let base = root.join(name);
        Ok(Self { config: base.join("config.json"), weights: base.join("model.safetensors") })
    }

    pub fn load_config(&self) -> Result<Value, String> {
        let json: Value = serde_json::from_slice(&fs::read(&self.config).map_err(|e| e.to_string())?)
            .map_err(|e| format!("{}: {e}", self.config.display()))?;
        Ok(json)
    }
}

pub struct Checkpoint {
    raw: Vec<u8>,
}

impl Checkpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let raw = fs::read(path.as_ref()).map_err(|e| e.to_string())?;
        SafeTensors::deserialize(&raw).map_err(|e| e.to_string())?;
        Ok(Self { raw })
    }

    fn reader(&self) -> Result<SafeTensors<'_>, String> {
        SafeTensors::deserialize(&self.raw).map_err(|e| e.to_string())
    }

    pub fn names(&self) -> Result<Vec<String>, String> {
        Ok(self.reader()?.names().iter().map(|s| (*s).to_owned()).collect())
    }

    pub fn has(&self, key: &str) -> bool {
        self.reader().is_ok_and(|r| r.tensor(key).is_ok())
    }

    pub fn values(&self, key: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        let reader = self.reader()?;
        let tensor = reader.tensor(key).map_err(|e| format!("{key}: {e}"))?;
        let shape = tensor.shape().to_vec();
        let bytes = tensor.data();
        let floats = match tensor.dtype() {
            Dtype::F32 => {
                if bytes.len() % 4 != 0 { return Err(format!("{key}: invalid f32 bytes")); }
                bytes.chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>()
            }
            Dtype::F64 => {
                if bytes.len() % 8 != 0 { return Err(format!("{key}: invalid f64 bytes")); }
                bytes.chunks_exact(8)
                    .map(|v| f64::from_le_bytes(v.try_into().unwrap()) as f32).collect()
            }
            Dtype::F16 => {
                if bytes.len() % 2 != 0 { return Err(format!("{key}: invalid f16 bytes")); }
                bytes.chunks_exact(2)
                    .map(|v| f16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f32()).collect()
            }
            Dtype::BF16 => {
                if bytes.len() % 2 != 0 { return Err(format!("{key}: invalid bf16 bytes")); }
                bytes.chunks_exact(2)
                    .map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f32()).collect()
            }
            dtype => return Err(format!("{key}: unsupported dtype {dtype:?}")),
        };
        let expected = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| format!("{key}: shape overflow"))?;
        if floats.len() != expected { return Err(format!("{key}: shape/data mismatch")); }
        Ok((shape, floats))
    }

    pub fn tensor<const N: usize>(&self, key: &str, device: &Device) -> Result<Tensor<N>, String> {
        let (shape, values) = self.values(key)?;
        let shape: [usize; N] = shape.try_into()
            .map_err(|dimensions: Vec<usize>| format!("{key}: expected {N} dimensions; got {dimensions:?}"))?;
        Ok(Tensor::from_data(TensorData::new(values, shape), device))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stage_layout_matches_upstream() {
        let paths = ModelPaths::in_directory("weights", "grouper").unwrap();
        assert_eq!(paths.config, PathBuf::from("weights/dettato-grouper/config.json"));
        assert_eq!(paths.weights, PathBuf::from("weights/dettato-grouper/model.safetensors"));
        assert!(ModelPaths::in_directory("weights", "unknown").is_err());
    }

    #[test]
    fn flat_archive_names_and_evidence_typo_resolve() {
        let root = std::env::temp_dir().join(format!("dettato-layout-{}", std::process::id()));
        let flat = root.join("dettato");
        std::fs::create_dir_all(&flat).unwrap();
        for alias in ["dettato", "detatto-evidence", "dettato-grouper", "dettato-drums"] {
            std::fs::write(flat.join(format!("{alias}-config.json")), b"{}").unwrap();
            std::fs::write(flat.join(format!("{alias}.safetensors")), b"").unwrap();
        }
        for (stage, alias) in [
            ("transcriber", "dettato"), ("evidence", "detatto-evidence"),
            ("grouper", "dettato-grouper"), ("drums", "dettato-drums"),
        ] {
            let found = ModelPaths::in_directory(&root, stage).unwrap();
            assert_eq!(found.config, flat.join(format!("{alias}-config.json")));
            assert_eq!(found.weights, flat.join(format!("{alias}.safetensors")));
        }
        std::fs::remove_dir_all(root).unwrap();
    }

}
