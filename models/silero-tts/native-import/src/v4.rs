use anyhow::{Context, Result, bail, ensure};
use crate::{dictionary, ids, string, require_tensor, DType, PickleValue as V, TorchPackageReader, WeightSet};
use std::collections::{BTreeMap, HashMap, HashSet};

pub struct V4Checkpoint {
    pub symbols: String,
    pub symbol_ids: BTreeMap<String, usize>,
    pub speakers: BTreeMap<String, usize>,
    pub ascii_symbols: BTreeMap<char, String>,
    pub weights: WeightSet,
    pub accent: Option<V4AccentCheckpoint>,
}

pub struct V4AccentCheckpoint {
    pub weights: WeightSet,
    pub ngrams: HashMap<String, usize>,
    pub exceptions: HashMap<String, (i64, i64)>,
    pub vowels: String,
    pub stop_words: HashSet<String>,
}

impl V4Checkpoint {
    pub(crate) fn from_package(package: TorchPackageReader) -> Result<Self> {
        let root = dictionary(package.metadata(), "V4 wrapper")?;
        ensure!(matches!(root.get("phons"), Some(V::Bool(false))), "unsupported V4 phoneme frontend");
        let symbols = string(root.get("symbols"), "symbols")?.to_owned();
        ensure!(symbols.starts_with("_~|") && (4..=4096).contains(&symbols.chars().count()), "invalid V4 symbols");
        let Some(V::List(tables)) = root.get("speaker_to_ids") else { bail!("invalid V4 speaker tables") };
        ensure!(tables.len() == 1, "unsupported V4 multi-graph routing");
        let speakers = ids(tables.first(), "speaker_to_ids")?;
        let routes = ids(root.get("speaker_to_model"), "speaker_to_model")?;
        ensure!(!speakers.is_empty() && speakers.len() <= 4096 && routes.len() == speakers.len() && speakers.keys().all(|k| routes.get(k) == Some(&0)), "invalid V4 routing");
        let Some(V::List(models)) = root.get("models") else { bail!("invalid V4 model references") };
        ensure!(models.len() == 1 && package.module_ids().len() == 1, "unsupported V4 script topology");
        let V::Unsupported(reference) = &models[0] else { bail!("invalid V4 script reference") };
        let module_id = reference.strip_prefix("torchpackage.script:").context("invalid V4 module id")?;
        let (script, tensors) = package.module(module_id)?;
        let script = dictionary(&script, "V4 graph")?;
        let has_accent = match script.get("set_accent") { Some(V::Bool(v)) => *v, _ => bail!("invalid V4 accentor flag") };
        let symbol_ids = ids(script.get("symbol_to_id"), "V4 script symbol IDs")?;
        let num_symbols = tensors.get("tts_model.tacotron.embedding.weight").context("missing V4 symbol embedding")?.shape().first().copied().context("invalid V4 embedding shape")?;
        let num_speakers = tensors.get("tts_model.tacotron.speaker_embedding.weight").context("missing V4 speaker embedding")?.shape().first().copied().context("invalid V4 embedding shape")?;
        ensure!((4..=4096).contains(&num_symbols) && (1..=4096).contains(&num_speakers), "invalid V4 embedding rows");
        ensure!(symbol_ids.values().all(|&v| v < num_symbols) && speakers.values().all(|&v| v < num_speakers), "V4 ids out of bounds");
        ensure!(symbol_ids.get("|") == Some(&2) && symbol_ids.get("~") == Some(&1), "invalid V4 boundary IDs");
        let mut ascii_symbols = BTreeMap::new();
        for (k, v) in dictionary(root.get("symb_ascii_dict").context("missing V4 ASCII mapping")?, "V4 ASCII mapping")? {
            let mut chars = k.chars();
            let ch = chars.next().context("empty V4 symbol")?;
            ensure!(chars.next().is_none(), "invalid V4 symbol key");
            ascii_symbols.insert(ch, string(Some(v), "V4 encoded symbol")?.to_owned());
        }
        ensure!(symbols.chars().all(|ch| ascii_symbols.get(&ch).cloned().unwrap_or_else(|| ch.to_string()).chars().all(|encoded| symbol_ids.contains_key(&encoded.to_string()))), "V4 alphabet does not match graph IDs");
        ensure!(speakers.values().copied().collect::<HashSet<_>>().len() == speakers.len(), "duplicate V4 speaker ID");
        crate::v3_schema::validate_acoustic(&tensors, num_symbols, num_speakers)?;
        let model = dictionary(script.get("tts_model").context("missing V4 TTS graph")?, "V4 TTS graph")?;
        let vocoder = dictionary(model.get("vocoder").context("missing V4 vocoder")?, "V4 vocoder")?;
        ensure!(matches!(vocoder.get("average_24_8"), Some(V::Bool(true))) && matches!(vocoder.get("num_kernels"), Some(V::Int(3))) && matches!(vocoder.get("multiband"), Some(V::Bool(true))) && matches!(vocoder.get("resample"), Some(V::Int(1))), "unsupported V4 vocoder topology");
        let tacotron = dictionary(model.get("tacotron").context("missing V4 tacotron")?, "V4 tacotron")?;
        ensure!(matches!(tacotron.get("pitch_strength"), Some(V::Float(x)) if *x == 1.0), "unsupported V4 pitch strength");
        let decoder = dictionary(tacotron.get("decoder").context("missing V4 decoder")?, "V4 decoder")?;
        ensure!(matches!(decoder.get("shorten_factor"), Some(V::Int(3))), "unsupported V4 decoder");
        let required = |name: &str, shape: &[usize]| require_tensor(&tensors, &format!("tts_model.vocoder.{name}"), DType::F32, shape);
        for (name, shape) in [("pqmf_layer.G", &[1,4,63][..]), ("pqmf_layer.updown_filter", &[4,4,4]), ("pqmf_3.H", &[3,1,63]), ("conv_pre.1.weight", &[128,128,7]), ("conv_post.weight", &[4,16,7]), ("conv_post_u.weight", &[4,16,7])] { required(name, shape)?; }
        for (name, width) in [("activation_ups.0", 128), ("activation_ups.1", 64), ("activation_ups.2", 32), ("activation_ups.3", 16), ("activation_post", 16)] {
            for suffix in ["alpha", "beta"] { required(&format!("{name}.{suffix}"), &[width])?; }
        }
        let accent = if has_accent {
            let node = dictionary(script.get("accentor").context("missing V4 accentor")?, "V4 accentor")?;
            let emb = dictionary(node.get("embedding").context("missing V4 accentor embedding")?, "V4 accentor embedding")?;
            let ngrams: HashMap<_, _> = ids(emb.get("ngram_dict"), "V4 ngrams")?.into_iter().collect();
            ensure!(ngrams.len() <= 1_000_000 && ngrams.contains_key("UNK") && ngrams.values().all(|&id| id < ngrams.len()), "invalid V4 ngram IDs");
            let embedding = tensors.get("accentor.embedding.weight").context("missing V4 accentor embedding weight")?;
            ensure!(embedding.shape() == [ngrams.len(),16] && matches!(embedding.dtype(), DType::F32 | DType::I8), "invalid V4 embedding shape or type");
            for (name, output) in [("stress_clf",10),("yo_clf",7)] {
                for (index, input, output) in [(0,16,64),(2,64,128),(4,128,64),(6,64,output)] {
                    require_tensor(&tensors, &format!("accentor.{name}.{index}.weight"), DType::F32, &[output,input])?;
                    require_tensor(&tensors, &format!("accentor.{name}.{index}.bias"), DType::F32, &[output])?;
                }
            }
            let exceptions = dictionary(script.get("exceptions").context("missing V4 exceptions")?, "V4 exceptions")?.iter().map(|(k,v)| {
                let V::List(items) = v else { bail!("invalid V4 exception") };
                let [V::Int(stress),V::Int(yo)] = items.as_slice() else { bail!("invalid V4 exception positions") };
                Ok((k.clone(), (*stress,*yo)))
            }).collect::<Result<_>>()?;
            let vowels = string(script.get("vowels"), "V4 vowels")?.to_owned();
            ensure!(string(script.get("stress_token"), "stress token")? == "+" && string(script.get("yo_token"), "yo token")? == "1" && string(script.get("e_token"), "e token")? == "e", "unsupported V4 accentor alphabet");
            ensure!(matches!(script.get("stress_trs"), Some(V::Float(x)) if *x == 0.5) && matches!(script.get("yo_trs"), Some(V::Float(x)) if *x == 0.5), "unsupported V4 accent thresholds");
            let Some(V::List(stop)) = script.get("stop_words") else { bail!("invalid V4 accent stop words") };
            let stop_words = stop.iter().map(|v| Ok(string(Some(v), "stop word")?.to_owned())).collect::<Result<_>>()?;
            let mut weights = WeightSet::new(tensors.iter().filter_map(|(k,v)| k.strip_prefix("accentor.").map(|s| (s.to_owned(),v.clone()))).collect())?;
            if embedding.dtype() == DType::I8 {
                require_tensor(&tensors, "accentor.scale", DType::F32, &[])?;
                require_tensor(&tensors, "accentor.zero_point", DType::F32, &[])?;
                let scale = weights.read_f32("scale")?[0];
                let zero = weights.read_f32("zero_point")?[0];
                weights.set_quantization("embedding.weight", scale, zero)?;
            }
            Some(V4AccentCheckpoint { weights, ngrams, exceptions, vowels, stop_words })
        } else { None };
        let weights = WeightSet::new(tensors.into_iter().filter_map(|(k,v)| k.strip_prefix("tts_model.").map(|s| (s.to_owned(),v))).collect())?;
        Ok(Self { symbols, symbol_ids, speakers, ascii_symbols, weights, accent })
    }
}
