use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path};

use fish_speech_core::config::FishModelConfig;
use fish_speech_core::cpu_ir::{NamedTensors, TensorData};
use fish_speech_core::graph_ir::{Graph, ScalarType, TensorType};
use fish_speech_core::transformer_ir::TransformerIrFactory;
use half::{bf16, f16};
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const WEIGHT_MANIFEST_SCHEMA: u32 = 1;
pub const MODEL_FAMILY: &str = "fish-speech-1.5";

#[derive(Debug, Error)]
pub enum WeightError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("manifest JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("safetensors: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("invalid model weights: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, WeightError>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorEntry {
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightManifest {
    pub schema_version: u32,
    pub model_family: String,
    pub component: String,
    pub source_checkpoint_sha256: String,
    pub source_license: String,
    pub config_sha256: String,
    pub shards: BTreeMap<String, String>,
    pub tensors: BTreeMap<String, TensorEntry>,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_hash(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn safe_shard(name: &str) -> bool {
    name.ends_with(".safetensors")
        && !name.is_empty()
        && !name.contains(['\\', ':'])
        && Path::new(name)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

impl WeightManifest {
    pub fn validate_schema(&self) -> Result<()> {
        if self.schema_version != WEIGHT_MANIFEST_SCHEMA
            || self.model_family != MODEL_FAMILY
            || !matches!(self.component.as_str(), "semantic" | "firefly")
            || self.source_license.trim().is_empty()
            || !valid_hash(&self.source_checkpoint_sha256)
            || !valid_hash(&self.config_sha256)
            || self.shards.is_empty()
            || self.tensors.is_empty()
        {
            return Err(WeightError::Invalid(
                "invalid model manifest schema or provenance".into(),
            ));
        }
        for (path, hash) in &self.shards {
            if !safe_shard(path) || !valid_hash(hash) {
                return Err(WeightError::Invalid(format!(
                    "invalid shard path or hash: {path}"
                )));
            }
        }
        for (key, meta) in &self.tensors {
            if key.is_empty()
                || !self.shards.contains_key(&meta.shard)
                || meta.shape.is_empty()
                || meta.shape.contains(&0)
                || meta
                    .shape
                    .iter()
                    .try_fold(1usize, |n, &dim| n.checked_mul(dim))
                    .is_none()
                || !valid_hash(&meta.sha256)
                || !matches!(meta.dtype.as_str(), "F32" | "F16" | "BF16" | "I32" | "I64")
            {
                return Err(WeightError::Invalid(format!(
                    "invalid tensor metadata: {key}"
                )));
            }
        }
        Ok(())
    }

    pub fn cache_fingerprint(
        &self,
        backend_abi: &str,
        device: &str,
        precision: &str,
        profile: &str,
    ) -> Result<String> {
        self.validate_schema()?;
        if [backend_abi, device, precision, profile]
            .iter()
            .any(|x| x.is_empty())
        {
            return Err(WeightError::Invalid(
                "incomplete engine cache identity".into(),
            ));
        }
        let mut hasher = Sha256::new();
        for fragment in [
            b"fish-speech/weights/engine-cache/v1\0".as_slice(),
            &serde_json::to_vec(self)?,
            backend_abi.as_bytes(),
            device.as_bytes(),
            precision.as_bytes(),
            profile.as_bytes(),
        ] {
            hasher.update((fragment.len() as u64).to_le_bytes());
            hasher.update(fragment);
        }
        Ok(format!("{:x}", hasher.finalize()))
    }
}

#[derive(Debug)]
pub struct LoadedWeights {
    manifest: WeightManifest,
    shards: BTreeMap<String, Vec<u8>>,
}

impl LoadedWeights {
    pub fn open(dir: &Path, manifest_file: &Path) -> Result<Self> {
        let metadata = fs::metadata(manifest_file)?;
        if metadata.len() > 16 * 1024 * 1024 {
            return Err(WeightError::Invalid("manifest exceeds 16 MiB limit".into()));
        }
        let manifest: WeightManifest = serde_json::from_slice(&fs::read(manifest_file)?)?;
        Self::load(dir, manifest)
    }

    pub fn load(dir: &Path, manifest: WeightManifest) -> Result<Self> {
        manifest.validate_schema()?;
        let root = fs::canonicalize(dir)?;
        let mut shards = BTreeMap::new();
        let mut seen = BTreeSet::new();
        for (name, expected_hash) in &manifest.shards {
            let resolved = fs::canonicalize(root.join(name))?;
            if !resolved.starts_with(&root) || !resolved.is_file() {
                return Err(WeightError::Invalid(format!(
                    "shard escapes model directory: {name}"
                )));
            }
            let bytes = fs::read(resolved)?;
            if sha256(&bytes) != *expected_hash {
                return Err(WeightError::Invalid(format!("shard hash mismatch: {name}")));
            }
            let safes = SafeTensors::deserialize(&bytes)?;
            for key in safes.names() {
                if !seen.insert(key.to_string()) {
                    return Err(WeightError::Invalid(format!(
                        "duplicate tensor across shards: {key}"
                    )));
                }
                let meta = manifest
                    .tensors
                    .get(key)
                    .ok_or_else(|| WeightError::Invalid(format!("unmanifested tensor: {key}")))?;
                let view = safes.tensor(key)?;
                if meta.shard != *name
                    || meta.dtype != format!("{:?}", view.dtype())
                    || meta.shape != view.shape()
                    || meta.sha256 != sha256(view.data())
                {
                    return Err(WeightError::Invalid(format!(
                        "tensor metadata/hash mismatch: {key}"
                    )));
                }
            }
            shards.insert(name.clone(), bytes);
        }
        for key in manifest.tensors.keys() {
            if !seen.contains(key) {
                return Err(WeightError::Invalid(format!(
                    "missing tensor in shards: {key}"
                )));
            }
        }
        Ok(Self { manifest, shards })
    }

    pub fn manifest(&self) -> &WeightManifest {
        &self.manifest
    }

    pub fn tensor(&self, key: &str) -> Result<TensorView<'_>> {
        let meta = self
            .manifest
            .tensors
            .get(key)
            .ok_or_else(|| WeightError::Invalid(format!("unknown checkpoint key: {key}")))?;
        let data = self
            .shards
            .get(&meta.shard)
            .ok_or_else(|| WeightError::Invalid(format!("missing shard: {}", meta.shard)))?;
        Ok(SafeTensors::deserialize(data)?.tensor(key)?)
    }

    pub(crate) fn semantic_upload_bytes(
        &self,
        key: &str,
        expected: &TensorType,
    ) -> Result<Vec<u8>> {
        if expected.dtype != ScalarType::BF16 {
            return self.semantic_f32_upload_bytes(key, expected);
        }
        let mut fp32 = expected.clone();
        fp32.dtype = ScalarType::F32;
        let data = self.semantic_f32_upload_bytes(key, &fp32)?;
        let mut packed = Vec::with_capacity(data.len() / 2);
        for chunk in data.chunks_exact(4) {
            let value = bf16::from_f32(f32::from_le_bytes(chunk.try_into().unwrap()));
            if !value.is_finite() {
                return Err(WeightError::Invalid(format!(
                    "weight exceeds BF16 range: {key}"
                )));
            }
            packed.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        Ok(packed)
    }

    pub fn semantic_f32_upload_bytes(&self, key: &str, expected: &TensorType) -> Result<Vec<u8>> {
        if self.manifest.component != "semantic" || expected.dtype != ScalarType::F32 {
            return Err(WeightError::Invalid(format!(
                "not a floating semantic TensorRT weight: {key}"
            )));
        }
        let view = self.tensor(key)?;
        if view.shape() != expected.shape {
            return Err(WeightError::Invalid(format!(
                "TensorRT weight shape mismatch: {key}"
            )));
        }
        let count = expected
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| {
                WeightError::Invalid(format!("TensorRT weight extent overflow: {key}"))
            })?;
        let mut output = Vec::with_capacity(count.checked_mul(4).ok_or_else(|| {
            WeightError::Invalid(format!("TensorRT weight byte count overflow: {key}"))
        })?);
        let bytes = view.data();
        match view.dtype() {
            Dtype::F32 if bytes.len() == count * 4 => {
                for sample in bytes.chunks_exact(4) {
                    if !f32::from_le_bytes(sample.try_into().unwrap()).is_finite() {
                        return Err(WeightError::Invalid(format!(
                            "non-finite TensorRT weight: {key}"
                        )));
                    }
                }
                output.extend_from_slice(bytes);
            }
            Dtype::F16 | Dtype::BF16 if bytes.len() == count * 2 => {
                for sample in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes(sample.try_into().unwrap());
                    let value = if view.dtype() == Dtype::F16 {
                        f16::from_bits(bits).to_f32()
                    } else {
                        bf16::from_bits(bits).to_f32()
                    };
                    if !value.is_finite() {
                        return Err(WeightError::Invalid(format!(
                            "non-finite TensorRT weight: {key}"
                        )));
                    }
                    output.extend_from_slice(&value.to_le_bytes());
                }
            }
            _ => {
                return Err(WeightError::Invalid(format!(
                    "unsupported TensorRT weight dtype: {key}"
                )));
            }
        }
        Ok(output)
    }

    fn cpu_f32_weights(
        &self,
        requirements: &BTreeMap<String, TensorType>,
        component: &str,
    ) -> Result<BTreeMap<String, Vec<f32>>> {
        if self.manifest.component != component {
            return Err(WeightError::Invalid(format!(
                "expected {component} model weights, got {}",
                self.manifest.component
            )));
        }
        let mut result = BTreeMap::new();
        for (key, expected) in requirements {
            if expected.dtype != ScalarType::F32 {
                return Err(WeightError::Invalid(format!(
                    "CPU graph requires floating-point weight: {key}"
                )));
            }
            let view = self.tensor(key)?;
            if view.shape() != expected.shape {
                return Err(WeightError::Invalid(format!(
                    "CPU graph checkpoint shape mismatch: {key}"
                )));
            }
            let bytes = view.data();
            let (width, values): (usize, Vec<f32>) = match view.dtype() {
                Dtype::F32 => (
                    4,
                    bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                        .collect(),
                ),
                Dtype::F16 => (
                    2,
                    bytes
                        .chunks_exact(2)
                        .map(|chunk| {
                            f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32()
                        })
                        .collect(),
                ),
                Dtype::BF16 => (
                    2,
                    bytes
                        .chunks_exact(2)
                        .map(|chunk| {
                            bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32()
                        })
                        .collect(),
                ),
                other => {
                    return Err(WeightError::Invalid(format!(
                        "unsupported floating checkpoint weight dtype {other:?}: {key}"
                    )));
                }
            };
            let count: usize = expected.shape.iter().product();
            if count.checked_mul(width) != Some(bytes.len())
                || values.len() != count
                || values.iter().any(|x| !x.is_finite())
            {
                return Err(WeightError::Invalid(format!(
                    "invalid or non-finite CPU tensor payload: {key}"
                )));
            }
            result.insert(key.clone(), values);
        }
        Ok(result)
    }

    pub fn to_cpu_named_tensors(
        &self,
        requirements: &BTreeMap<String, TensorType>,
    ) -> Result<NamedTensors> {
        Ok(self
            .cpu_f32_weights(requirements, "semantic")?
            .into_iter()
            .map(|(name, tensor)| (name, TensorData::F32(tensor)))
            .collect())
    }

    pub fn firefly_cpu_weights(
        &self,
        graph: &fish_speech_core::firefly_ir::FireflyGraph,
    ) -> Result<BTreeMap<String, Vec<f32>>> {
        graph
            .validate()
            .map_err(|e| WeightError::Invalid(format!("Firefly graph: {e}")))?;
        self.cpu_f32_weights(&graph.weights, "firefly")
    }

    pub fn semantic_cpu_weights(&self, config: &FishModelConfig) -> Result<NamedTensors> {
        let factory = TransformerIrFactory::new(config)
            .map_err(|e| WeightError::Invalid(format!("invalid semantic model config: {e}")))?;
        let slow = factory
            .slow_prefill(1)
            .map_err(|e| WeightError::Invalid(format!("Slow graph: {e}")))?;
        let mut requirements = slow.weight_requirements();
        if config.fast.is_some() {
            for step in [0, config.num_codebooks.saturating_sub(1)] {
                let fast = factory
                    .fast_decode(step)
                    .map_err(|e| WeightError::Invalid(format!("Fast graph: {e}")))?;
                for (key, ty) in fast.weight_requirements() {
                    if let Some(old) = requirements.insert(key.clone(), ty.clone()) {
                        if old != ty {
                            return Err(WeightError::Invalid(format!(
                                "conflicting CPU AR weight shape: {key}"
                            )));
                        }
                    }
                }
            }
        }
        self.to_cpu_named_tensors(&requirements)
    }

    pub fn validate_graph(&self, graph: &Graph) -> Result<usize> {
        graph
            .validate()
            .map_err(|error| WeightError::Invalid(format!("graph validation: {error:?}")))?;
        self.validate_requirements(&graph.weight_requirements())
    }

    pub fn validate_exact_requirements(
        &self,
        requirements: &BTreeMap<String, TensorType>,
    ) -> Result<usize> {
        let validated = self.validate_requirements(requirements)?;
        let unexpected = self
            .manifest
            .tensors
            .keys()
            .filter(|key| !requirements.contains_key(*key))
            .cloned()
            .collect::<Vec<_>>();
        if !unexpected.is_empty() {
            return Err(WeightError::Invalid(format!(
                "unexpected checkpoint tensors: {}",
                unexpected.join(", ")
            )));
        }
        Ok(validated)
    }

    pub fn validate_requirements(
        &self,
        requirements: &BTreeMap<String, TensorType>,
    ) -> Result<usize> {
        for (key, expected) in requirements {
            let view = self.tensor(key)?;
            let dtype = match expected.dtype {
                ScalarType::F32 => Dtype::F32,
                ScalarType::BF16 => Dtype::BF16,
                ScalarType::I32 => Dtype::I32,
            };
            if view.dtype() != dtype || view.shape() != expected.shape {
                return Err(WeightError::Invalid(format!(
                    "graph weight mismatch: {key}, expected {:?} {:?}, got {:?} {:?}",
                    dtype,
                    expected.shape,
                    view.dtype(),
                    view.shape()
                )));
            }
        }
        Ok(requirements.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fish_speech_core::graph::Stage;
    use fish_speech_core::graph_ir::{Layout, Op};
    use safetensors::tensor::{TensorView, serialize};

    fn fixture() -> (Vec<u8>, WeightManifest) {
        let values = [1.0_f32, -2.0, 3.0, 4.0];
        let data = values
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<_>>();
        let view = TensorView::new(Dtype::F32, vec![2, 2], &data).unwrap();
        let encoded = serialize([("layers.0.weight", view)], None).unwrap();
        let manifest = WeightManifest {
            schema_version: 1,
            model_family: MODEL_FAMILY.into(),
            component: "semantic".into(),
            source_checkpoint_sha256: sha256(b"trusted-source"),
            source_license: "fixture-only".into(),
            config_sha256: sha256(b"config"),
            shards: BTreeMap::from([("semantic.safetensors".into(), sha256(&encoded))]),
            tensors: BTreeMap::from([(
                "layers.0.weight".into(),
                TensorEntry {
                    shard: "semantic.safetensors".into(),
                    dtype: "F32".into(),
                    shape: vec![2, 2],
                    sha256: sha256(&data),
                },
            )]),
        };
        (encoded, manifest)
    }

    fn temp_path(suffix: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("fish-weights-{}-{suffix}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn exact_weights_can_bind_to_typed_graph() {
        let (data, manifest) = fixture();
        let path = temp_path("exact");
        fs::write(path.join("semantic.safetensors"), data).unwrap();
        let weights = LoadedWeights::load(&path, manifest).unwrap();
        let mut graph = Graph::new(Stage::SlowPrefill, None, None);
        let input = graph
            .input(
                "input",
                TensorType::new(vec![2, 2], ScalarType::F32, Layout::Plain),
            )
            .unwrap();
        let weight = graph
            .weight(
                "layers.0.weight",
                TensorType::new(vec![2, 2], ScalarType::F32, Layout::Plain),
            )
            .unwrap();
        let output = graph.emit("copy", Op::Identity, &[input]).unwrap();
        graph.output(output).unwrap();
        assert_eq!(weights.validate_graph(&graph).unwrap(), 1);
        assert_eq!(
            weights
                .validate_exact_requirements(&graph.weight_requirements())
                .unwrap(),
            1
        );
        let no_weights = BTreeMap::new();
        assert!(weights.validate_exact_requirements(&no_weights).is_err());
        assert_eq!(weights.tensor("layers.0.weight").unwrap().data().len(), 16);
        let mut invalid = BTreeMap::new();
        invalid.insert(
            "layers.0.weight".into(),
            TensorType::new(vec![4], ScalarType::F32, Layout::Plain),
        );
        assert!(weights.validate_requirements(&invalid).is_err());
        assert!(weight.0 > 0);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn cpu_weight_bridge_accepts_f32_and_checks_shapes() {
        let (data, manifest) = fixture();
        let path = temp_path("cpu-bridge");
        fs::write(path.join("semantic.safetensors"), data).unwrap();
        let weights = LoadedWeights::load(&path, manifest).unwrap();
        let required = BTreeMap::from([(
            "layers.0.weight".to_string(),
            TensorType::new(vec![2, 2], ScalarType::F32, Layout::Plain),
        )]);
        assert_eq!(
            weights.to_cpu_named_tensors(&required).unwrap()["layers.0.weight"],
            TensorData::F32(vec![1.0, -2.0, 3.0, 4.0]),
        );
        let invalid = BTreeMap::from([(
            "layers.0.weight".to_string(),
            TensorType::new(vec![4], ScalarType::F32, Layout::Plain),
        )]);
        assert!(weights.to_cpu_named_tensors(&invalid).is_err());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn cpu_weight_bridge_converts_bf16_and_f16_exactly() {
        let value = [1.0_f32, -0.5_f32, 2.0_f32];
        let f16_bytes = value
            .iter()
            .flat_map(|f| f16::from_f32(*f).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let bf16_bytes = value
            .iter()
            .flat_map(|f| bf16::from_f32(*f).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let encoded = serialize(
            [
                (
                    "f16",
                    TensorView::new(Dtype::F16, vec![3], &f16_bytes).unwrap(),
                ),
                (
                    "bf16",
                    TensorView::new(Dtype::BF16, vec![3], &bf16_bytes).unwrap(),
                ),
            ],
            None,
        )
        .unwrap();
        let manifest = WeightManifest {
            schema_version: 1,
            model_family: MODEL_FAMILY.into(),
            component: "semantic".into(),
            source_checkpoint_sha256: sha256(b"trusted-source"),
            source_license: "fixture-only".into(),
            config_sha256: sha256(b"config"),
            shards: BTreeMap::from([("semantic.safetensors".into(), sha256(&encoded))]),
            tensors: BTreeMap::from([
                (
                    "f16".into(),
                    TensorEntry {
                        shard: "semantic.safetensors".into(),
                        dtype: "F16".into(),
                        shape: vec![3],
                        sha256: sha256(&f16_bytes),
                    },
                ),
                (
                    "bf16".into(),
                    TensorEntry {
                        shard: "semantic.safetensors".into(),
                        dtype: "BF16".into(),
                        shape: vec![3],
                        sha256: sha256(&bf16_bytes),
                    },
                ),
            ]),
        };
        let path = temp_path("cpu-mixed-dtypes");
        fs::write(path.join("semantic.safetensors"), encoded).unwrap();
        let weights = LoadedWeights::load(&path, manifest).unwrap();
        let req = BTreeMap::from([
            (
                "f16".into(),
                TensorType::new([3], ScalarType::F32, Layout::Plain),
            ),
            (
                "bf16".into(),
                TensorType::new([3], ScalarType::F32, Layout::Plain),
            ),
        ]);
        let converted = weights.to_cpu_named_tensors(&req).unwrap();
        for key in ["bf16", "f16"] {
            let packed = weights.semantic_f32_upload_bytes(key, &req[key]).unwrap();
            assert_eq!(
                packed,
                value
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()
            );
        }
        assert!(
            weights
                .semantic_f32_upload_bytes(
                    "f16",
                    &TensorType::new([4], ScalarType::F32, Layout::Plain)
                )
                .is_err()
        );
        assert_eq!(converted["bf16"], TensorData::F32(value.to_vec()));
        assert_eq!(converted["f16"], TensorData::F32(value.to_vec()));
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn changed_payload_or_manifest_is_rejected() {
        let (data, manifest) = fixture();
        let path = temp_path("tamper");
        fs::write(path.join("semantic.safetensors"), data).unwrap();
        let mut wrong = manifest.clone();
        wrong.tensors.get_mut("layers.0.weight").unwrap().shape = vec![4];
        assert!(LoadedWeights::load(&path, wrong).is_err());
        let mut wrong = manifest.clone();
        wrong.tensors.get_mut("layers.0.weight").unwrap().sha256 = sha256(b"tampered");
        assert!(LoadedWeights::load(&path, wrong).is_err());
        let mut wrong = manifest.clone();
        wrong.tensors.get_mut("layers.0.weight").unwrap().shard = "../unsafe.safetensors".into();
        assert!(LoadedWeights::load(&path, wrong).is_err());
        fs::write(
            path.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        assert!(LoadedWeights::open(&path, &path.join("manifest.json")).is_ok());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn engine_cache_fingerprint_rejects_missing_identity_and_tracks_profiles() {
        let (_, manifest) = fixture();
        let a = manifest
            .cache_fingerprint("trt-rtx-1.7", "sm89", "bf16", "kv-128")
            .unwrap();
        let b = manifest
            .cache_fingerprint("trt-rtx-1.7", "sm89", "bf16", "kv-256")
            .unwrap();
        assert_ne!(a, b);
        assert_eq!(a.len(), 64);
        assert!(
            manifest
                .cache_fingerprint("", "sm89", "bf16", "kv-128")
                .is_err()
        );
    }
}
