use anyhow::{Context, Result, bail, ensure};
pub use pytorch_reader::{DType, PickleValue, Tensor};
use pytorch_reader::TorchPackageReader;
use std::{collections::{BTreeMap, HashMap, HashSet}, path::Path};

mod russian;
mod english_schema;
pub use russian::{RussianCheckpoint, RussianConfig, RussianTokenizer, RussianControls};

pub enum OfficialCheckpoint { English(EnglishCheckpoint), Russian(RussianCheckpoint) }
impl OfficialCheckpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let package = TorchPackageReader::open(path)?;
        let metadata = dictionary(package.metadata(), "model metadata")?;
        if metadata.contains_key("speaker_to_id") { Ok(Self::English(EnglishCheckpoint::from_package(package)?)) }
        else if metadata.contains_key("packages") { Ok(Self::Russian(RussianCheckpoint::from_package(package)?)) }
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

pub struct EnglishCheckpoint {
    pub symbols: String,
    pub speakers: BTreeMap<String, usize>,
    pub ascii_symbols: BTreeMap<char, String>,
    pub module_metadata: PickleValue,
    tensors: HashMap<String, Tensor>,
}

impl EnglishCheckpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let package = TorchPackageReader::open(path.as_ref()).context("reading Silero TorchPackage")?;
        Self::from_package(package)
    }

    pub(crate) fn from_package(package: TorchPackageReader) -> Result<Self> {
        let metadata = dictionary(package.metadata(), "model metadata")?;
        let symbols = string(metadata.get("symbols"), "symbols")?.to_owned();
        let speakers = ids(metadata.get("speaker_to_id"), "speaker_to_id")?;
        ensure!(symbols.chars().count() == 39 && symbols.starts_with("_~|"), "unsupported v3_en symbols");
        ensure!(symbols.chars().collect::<HashSet<_>>().len() == 39, "duplicate v3_en symbols");
        ensure!(speakers.get("en_0") == Some(&0), "missing v3_en speaker en_0");
        ensure!(speakers.values().all(|&id| id < 119), "speaker id outside embedding table");
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
        ensure!(package.module_ids().len() == 1, "v3_en requires exactly one script module");
        let (module_metadata, tensors) = package.module(id).context("reading v3_en script tensors")?;
        english_schema::validate(&tensors, &module_metadata)?;
        let total = tensors.values().try_fold(0usize, |sum, tensor| sum.checked_add(tensor.byte_len())).context("tensor byte count overflow")?;
        ensure!(total <= 256 * 1024 * 1024, "v3_en tensor byte budget exceeded");
        let script = dictionary(&module_metadata, "script metadata")?;
        let symbol_to_id = ids(script.get("symbol_to_id"), "symbol_to_id")?;
        for (idx, ch) in symbols.chars().enumerate() {
            ensure!(symbol_to_id.get(&ch.to_string()) == Some(&idx), "inconsistent symbol id for {ch:?}");
        }
        Ok(Self { symbols, speakers, ascii_symbols, module_metadata, tensors })
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
        let sentence = normalize_english(&self.symbols, &self.ascii_symbols, text)?;
        encode_english(&self.symbols, &sentence)
    }
}

pub fn normalize_english(symbols: &str, ascii_symbols: &BTreeMap<char, String>, text: &str) -> Result<String> {
    let text = text.to_lowercase().replace('—', "–").replace('‑', "-");
    let allowed: HashSet<char> = symbols.chars().skip(3).collect();
    let filtered: String = text.chars().filter(|c| allowed.contains(c)).collect();
    let normalized = filtered.split_whitespace().collect::<Vec<_>>().join(" ");
    let sentence: String = normalized.chars().map(|ch| ascii_symbols.get(&ch).cloned().unwrap_or_else(|| ch.to_string())).collect();
    ensure!(sentence.chars().any(|ch| ch.is_ascii_lowercase() || ('1'..='9').contains(&ch) || ch == '-'), "text empty after English normalization");
    Ok(sentence)
}

pub fn encode_english(symbols: &str, sentence: &str) -> Result<Vec<i64>> {
    std::iter::once('|').chain(sentence.chars()).chain(std::iter::once('~')).map(|ch| {
        symbols.chars().position(|symbol| symbol == ch).map(|i| i as i64).with_context(|| format!("unknown English symbol {ch:?}"))
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
