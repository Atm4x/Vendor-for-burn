use anyhow::{Context, Result, ensure, bail};
use crate::{dictionary, ids, string, require_tensor, WeightSet, DType, PickleValue as V, TorchPackageReader};
use std::collections::{BTreeMap, HashMap, HashSet};

/// One V5 ConvNeXt package with an explicit text frontend and no neural accentor.
pub struct V5PlainCheckpoint {
    pub input_symbols: String,
    pub symbol_ids: BTreeMap<String, usize>,
    pub alphabet: HashSet<char>,
    pub sos: String,
    pub eos: String,
    pub speakers: BTreeMap<String, usize>,
    pub transliteration: BTreeMap<String, BTreeMap<char, String>>,
    pub acoustic: WeightSet,
    pub pqmf: WeightSet,
}

fn field<'a>(value: &'a HashMap<String, V>, key: &str) -> Result<&'a V> {
    value.get(key).with_context(|| format!("missing V5 metadata {key}"))
}
fn one<'a>(value: &'a V, key: &str) -> Result<&'a V> {
    let V::List(items) = value else { bail!("expected list for {key}") };
    ensure!(items.len() == 1, "multiple {key} require graph routing support");
    Ok(&items[0])
}
fn char_key(s: &str) -> Result<char> {
    let mut chars = s.chars();
    let first = chars.next().context("empty alphabet key")?;
    ensure!(chars.next().is_none(), "multi-character alphabet key");
    Ok(first)
}

impl V5PlainCheckpoint {
    pub(crate) fn from_package(package: TorchPackageReader) -> Result<Self> {
        let root = dictionary(package.metadata(), "V5 root")?;
        let part = dictionary(one(field(root, "packages")?, "packages")?, "V5 model package")?;
        ensure!(matches!(part.get("accentor"), Some(V::None)), "V5 plain graph requires no accentor");
        ensure!(matches!(part.get("phons"), Some(V::Bool(false))), "V5 phoneme frontend is unsupported");
        let input_symbols = string(part.get("symbols"), "symbols")?.to_owned();
        let symbol_ids = ids(part.get("symbol_to_id"), "symbol_to_id")?;
        let sos = string(part.get("sos_token"), "sos_token")?.to_owned();
        let eos = string(part.get("eos_token"), "eos_token")?.to_owned();
        let V::List(alphabet) = field(part, "alphabet")? else { bail!("invalid V5 alphabet") };
        let alphabet = alphabet.iter().map(|v| char_key(string(Some(v), "alphabet")?)).collect::<Result<HashSet<_>>>()?;
        let speakers = ids(Some(one(field(part, "speaker_to_ids")?, "speaker_to_ids")?), "speaker_to_ids")?;
        ensure!(!speakers.is_empty() && speakers.len() <= 4096 && speakers.values().copied().collect::<HashSet<_>>().len() == speakers.len() && speakers.values().all(|&id| id < speakers.len()), "invalid V5 speaker IDs");
        for (node, name) in [(root, "speaker_to_package"), (part, "speaker_to_model")] {
            let routes = ids(node.get(name), name)?;
            ensure!(routes.len() == speakers.len() && speakers.keys().all(|key| routes.get(key) == Some(&0)), "invalid single-graph V5 routing: {name}");
        }
        let mut transliteration = BTreeMap::new();
        match part.get("ext_alph") {
            None | Some(V::None) => (),
            Some(V::Dict(languages)) => {
                for (lang, mapping) in languages {
                    let mut entries = BTreeMap::new();
                    for (key, value) in dictionary(mapping, "transliteration")? {
                        // Upstream converts one Unicode character at a time, even when a mapping
                        // contains digraph keys. Preserve that behavior rather than inventing it.
                        if key.chars().count() == 1 {
                            entries.insert(char_key(key)?, string(Some(value), "transliteration value")?.to_owned());
                        }
                    }
                    transliteration.insert(lang.clone(), entries);
                }
            }
            _ => bail!("invalid V5 transliteration metadata"),
        }
        let V::Unsupported(reference) = one(field(part, "models")?, "models")? else { bail!("missing V5 model reference") };
        let id = reference.strip_prefix("torchpackage.script:").context("invalid V5 script reference")?;
        ensure!(package.module_ids().len() == 1, "unsupported V5 script topology");
        let (script, mut tensors) = package.module(id)?;
        ensure!(tensors.len() == 317, "unsupported plain V5 graph: expected 317 tensors, got {}", tensors.len());
        let script = dictionary(&script, "V5 acoustic metadata")?;
        ensure!(matches!(script.get("mean_std_coef"), Some(V::List(v)) if v.len() == speakers.len()), "invalid V5 pitch metadata");
        let symbol_count = tensors.get("tacotron.embedding.weight").context("missing V5 embedding")?.shape().first().copied().context("invalid V5 embedding")?;
        ensure!((4..=4096).contains(&symbol_count) && symbol_ids.values().all(|&id| id < symbol_count), "invalid V5 symbol IDs");
        ensure!(symbol_ids.get(&sos) == Some(&2) && symbol_ids.get(&eos) == Some(&1), "unsupported V5 boundary IDs");
        ensure!(input_symbols.chars().skip(3).all(|ch| symbol_ids.contains_key(&ch.to_string())), "V5 input alphabet does not match script IDs");
        for prefix in ["dur_predictor.dur_pred", "pitch_predictor.pitch_pred", "tacotron"] {
            let width = if prefix == "tacotron" {128} else {64};
            require_tensor(&tensors, &format!("{prefix}.embedding.weight"), DType::F32, &[symbol_count, width])?;
            require_tensor(&tensors, &format!("{prefix}.speaker_embedding.weight"), DType::F32, &[speakers.len(), width])?;
        }
        for (name, shape) in [("tacotron.lin.weight", &[192,128][..]), ("tacotron.energy_proj.weight", &[128,1,3]), ("vocoder.head.istft.window", &[2400])] {
            require_tensor(&tensors, name, DType::F32, shape)?;
        }
        // New V5 packages hold PQMF buffers in the script graph rather than wrapper state.
        let mut pqmf = HashMap::new();
        for bands in [2,6] {
            let name = format!("vocoder.pqmf_{bands}.H");
            require_tensor(&tensors, &name, DType::F32, &[bands,1,63])?;
            pqmf.insert(format!("analysis_{bands}"), tensors.remove(&name).unwrap());
        }
        Ok(Self { input_symbols, symbol_ids, alphabet, sos, eos, speakers, transliteration,
            acoustic: WeightSet::new(tensors)?, pqmf: WeightSet::new(pqmf)? })
    }
}
