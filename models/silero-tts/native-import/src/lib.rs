use anyhow::{Context, Result, bail, ensure};
pub use pytorch_reader::{DType, PickleValue, Tensor};
use pytorch_reader::TorchPackageReader;
use std::{collections::{BTreeMap, HashMap, HashSet}, path::Path};

mod russian;
mod layout;
pub use layout::ArchiveLayout;
mod v3_schema;
mod v5_plain;
mod v4;
mod v2;
pub use v2::{V2Checkpoint, encode_v2};
mod ascii_accent;
pub use ascii_accent::AsciiAccentCheckpoint;
pub use v4::V4Checkpoint;
pub use v5_plain::V5PlainCheckpoint;
pub use russian::{RussianCheckpoint, RussianConfig, RussianTokenizer, RussianControls, RussianProsody};

pub enum OfficialCheckpoint { V2(V2Checkpoint), V4(V4Checkpoint), V3(V3Checkpoint), Russian(RussianCheckpoint), V5Plain(V5PlainCheckpoint) }
impl OfficialCheckpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        match ArchiveLayout::inspect(path)? {
            ArchiveLayout::TtsPackage => (),
            ArchiveLayout::TorchScript => bail!("standalone TorchScript detected; native graph lowering is not implemented for this layout"),
        }
        let package = TorchPackageReader::open(path)?;
        let metadata = dictionary(package.metadata(), "model metadata")?;
        if let Some(PickleValue::List(parts)) = metadata.get("packages") {
            let part = parts.first().context("empty Silero V5 package")?;
            let fields = dictionary(part, "V5 package metadata")?;
            if matches!(fields.get("accentor"), Some(PickleValue::None)) {
                Ok(Self::V5Plain(V5PlainCheckpoint::from_package(package)?))
            } else {
                Ok(Self::Russian(RussianCheckpoint::from_package(package)?))
            }
        }
        else if metadata.contains_key("speaker_to_ids") { Ok(Self::V4(V4Checkpoint::from_package(package)?)) }
        else if matches!(metadata.get("symbols"), Some(PickleValue::String(s)) if s.starts_with("_~") && !s.starts_with("_~|")) { Ok(Self::V2(V2Checkpoint::from_package(package)?)) }
        else if metadata.contains_key("speaker_to_id") { Ok(Self::V3(V3Checkpoint::from_package(package)?)) }
        else { bail!("unsupported Silero package architecture") }
    }
}

pub struct WeightSet {
    tensors: HashMap<String, Tensor>,
    quantization: HashMap<String, (f32, f32)>,
}
impl WeightSet {
    pub fn new(tensors: HashMap<String, Tensor>) -> Result<Self> {
        let total = tensors.values().try_fold(0usize, |sum, t| sum.checked_add(t.byte_len())).context("weight byte count overflow")?;
        ensure!(total <= 256 * 1024 * 1024, "weight set exceeds byte budget");
        Ok(Self { tensors, quantization: HashMap::new() })
    }
    pub fn tensors(&self) -> &HashMap<String, Tensor> { &self.tensors }
    pub fn shape(&self, name: &str) -> Result<&[usize]> { Ok(self.tensors.get(name).with_context(|| format!("missing tensor {name}"))?.shape()) }
    pub fn set_quantization(&mut self, name: &str, scale: f32, zero: f32) -> Result<()> {
        ensure!(self.tensors.get(name).is_some_and(|t| t.dtype()==DType::I8), "expected quantized I8 tensor {name}");
        ensure!(scale.is_finite() && scale > 0.0 && zero.is_finite(), "invalid quantization parameters");
        self.quantization.insert(name.to_owned(), (scale, zero)); Ok(())
    }
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
        let values: Vec<f32> = match t.dtype() {
            DType::F32 => t.read()?.chunks_exact(4).map(|b| f32::from_ne_bytes(b.try_into().unwrap())).collect(),
            DType::I8 => { let (scale, zero)=self.quantization.get(name).context("missing I8 dequantization parameters")?; t.read()?.into_iter().map(|x| scale*(x as i8 as f32-zero)).collect() },
            dtype => bail!("unexpected tensor type {dtype:?} for {name}"),
        };
        ensure!(values.len()==t.num_elements() && values.iter().all(|v| v.is_finite()), "invalid numeric data for {name}");
        Ok(values)
    }
}
fn require_tensor(tensors: &HashMap<String, Tensor>, name: &str, dtype: DType, shape: &[usize]) -> Result<()> {
    let t = tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
    ensure!(t.dtype()==dtype && t.shape()==shape, "unexpected tensor {name}: {:?} {:?}", t.dtype(), t.shape()); Ok(())
}

pub struct V3Checkpoint {
    pub symbols: String,
    pub speakers: BTreeMap<String, usize>,
    pub ascii_symbols: BTreeMap<char, String>,
    pub module_metadata: PickleValue,
    pub encoded_symbols: String,
    tensors: HashMap<String, Tensor>,
    pub accent: Option<AsciiAccentCheckpoint>,
}

impl V3Checkpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let package = TorchPackageReader::open(path.as_ref()).context("reading Silero TorchPackage")?;
        Self::from_package(package)
    }

    pub(crate) fn from_package(package: TorchPackageReader) -> Result<Self> {
        let metadata = dictionary(package.metadata(), "model metadata")?;
        let symbols = string(metadata.get("symbols"), "symbols")?.to_owned();
        let speakers = ids(metadata.get("speaker_to_id"), "speaker_to_id")?;
        let symbol_count = symbols.chars().count();
        ensure!((4..=4096).contains(&symbol_count) && symbols.starts_with("_~|"), "unsupported V3 input symbols");
        ensure!(symbols.chars().collect::<HashSet<_>>().len() == symbol_count, "duplicate V3 input symbols");
        ensure!(!speakers.is_empty() && speakers.len() <= 4096, "invalid V3 speaker table");
        ensure!(speakers.values().copied().collect::<HashSet<_>>().len() == speakers.len() && speakers.values().all(|&id| id < speakers.len()), "noncontiguous V3 speaker IDs");
        let mut ascii_symbols = BTreeMap::new();
        if let Some(value) = metadata.get("symb_ascii_dict") {
            for (key, value) in dictionary(value, "symb_ascii_dict")? {
                let mut chars = key.chars();
                let ch = chars.next().context("empty symbol mapping key")?;
                ensure!(chars.next().is_none(), "non-character symbol mapping key");
                ascii_symbols.insert(ch, string(Some(value), "symbol mapping")?.to_owned());
            }
        }
        let Some(PickleValue::Unsupported(reference)) = metadata.get("model") else { bail!("missing script module reference"); };
        let id = reference.strip_prefix("torchpackage.script:").context("unsupported model reference")?;
        ensure!(package.module_ids().len() == 1, "V3 requires exactly one script module");
        let (module_metadata, mut tensors) = package.module(id).context("reading V3 script tensors")?;
        let script = dictionary(&module_metadata, "script metadata")?;
        let accent = match script.get("set_accent") {
            Some(PickleValue::Bool(true)) => {
                let accent = ascii_accent::load_ascii_accent(script, &tensors)?;
                tensors.retain(|name, _| !name.starts_with("accentor."));
                Some(accent)
            }
            Some(PickleValue::Bool(false)) => None,
            _ => bail!("unsupported V3 accentor topology"),
        };
        v3_schema::validate(&tensors, &module_metadata, symbol_count, speakers.len())?;
        let total = tensors.values().try_fold(0usize, |sum, tensor| sum.checked_add(tensor.byte_len())).context("tensor byte count overflow")?;
        ensure!(total <= 256 * 1024 * 1024, "V3 tensor byte budget exceeded");
        let script = dictionary(&module_metadata, "script metadata")?;
        let symbol_to_id = ids(script.get("symbol_to_id"), "symbol_to_id")?;
        let encoded_symbols: String = symbols.chars().map(|ch| ascii_symbols.get(&ch).cloned().unwrap_or_else(|| ch.to_string())).collect();
        ensure!(encoded_symbols.chars().count() == symbol_count && symbol_to_id.len() == symbol_count, "invalid V3 encoded alphabet size");
        for (idx, ch) in encoded_symbols.chars().enumerate() {
            ensure!(symbol_to_id.get(&ch.to_string()) == Some(&idx), "inconsistent encoded symbol id for {ch:?}");
        }
        Ok(Self { symbols, speakers, ascii_symbols, module_metadata, encoded_symbols, tensors, accent })
    }

    pub fn tensors(&self) -> &HashMap<String, Tensor> { &self.tensors }
    pub fn into_weights(self) -> Result<WeightSet> {
        let mut weights = HashMap::new();
        for (name, tensor) in self.tensors {
            let name = name.strip_prefix("tts_model.").unwrap_or(&name).to_owned();
            ensure!(weights.insert(name.clone(), tensor).is_none(), "duplicate tensor name {name}");
        }
        WeightSet::new(weights)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let sentence = normalize_v3(&self.symbols, &self.ascii_symbols, text)?;
        encode_v3(&self.encoded_symbols, &sentence)
    }
}

pub fn normalize_v3(symbols: &str, ascii_symbols: &BTreeMap<char, String>, text: &str) -> Result<String> {
    let text = text.to_lowercase().replace('—', "–").replace('‑', "-");
    let allowed: HashSet<char> = symbols.chars().skip(3).collect();
    let filtered: String = text.chars().filter(|c| allowed.contains(c)).collect();
    let normalized = filtered.split_whitespace().collect::<Vec<_>>().join(" ");
    let sentence: String = normalized.chars().map(|ch| ascii_symbols.get(&ch).cloned().unwrap_or_else(|| ch.to_string())).collect();
    ensure!(sentence.chars().any(|ch| ch.is_ascii_lowercase() || ('1'..='9').contains(&ch) || ch == '-'), "text empty after V3 normalization");
    Ok(sentence)
}

pub fn encode_v3(symbols: &str, sentence: &str) -> Result<Vec<i64>> {
    std::iter::once('|').chain(sentence.chars()).chain(std::iter::once('~')).map(|ch| {
        symbols.chars().position(|symbol| symbol == ch).map(|i| i as i64).with_context(|| format!("unknown V3 symbol {ch:?}"))
    }).collect()
}

fn dictionary<'a>(value: &'a PickleValue, field: &str) -> Result<&'a HashMap<String, PickleValue>> {
    match value { PickleValue::Dict(dict) => Ok(dict), _ => bail!("invalid {field}: expected dictionary") }
}
fn string<'a>(value: Option<&'a PickleValue>, field: &str) -> Result<&'a str> {
    match value { Some(PickleValue::String(s)) => Ok(s), _ => bail!("invalid {field}: expected string") }
}
fn ids(value: Option<&PickleValue>, field: &str) -> Result<BTreeMap<String, usize>> {
    dictionary(value.with_context(|| format!("missing {field}"))?, field)?.iter().map(|(key, value)| {
        let PickleValue::Int(id) = value else { bail!("invalid {field} id for {key}"); };
        Ok((key.clone(), usize::try_from(*id).context("negative or oversized id")?))
    }).collect()
}
