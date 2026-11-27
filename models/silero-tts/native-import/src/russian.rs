use crate::{dictionary, ids, string, WeightSet, require_tensor};
use anyhow::{Context, Result, bail, ensure};
use pytorch_reader::{DType, PickleValue as V, TorchPackageReader};
use std::{collections::{BTreeMap, HashMap}, path::Path};

pub struct RussianConfig {
    pub symbols: String,
    pub alphabet: String,
    pub sos: String,
    pub eos: String,
    pub speakers: BTreeMap<String, usize>,
}

pub struct RussianTokenizer {
    pub unk_token: String,
    pub cls_token: String,
    pub sep_token: String,
    pub pad_token: String,
    pub homo_start_id: usize,
    pub homo_end_id: usize,
    pub never_split: Vec<String>,
    pub tokenize_chinese_chars: bool,
    pub model_max_length: usize,
    pub vocab: HashMap<String, usize>,
}

pub struct RussianControls {
    pub point_id: i64,
    pub comma_ids: Vec<i64>,
    pub punctuation_ids: Vec<i64>,
    pub pitch_mean_shift: Vec<f32>,
    pub sil_value: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RussianProsody { Legacy, PitchConditioned, SentenceConditioned }

pub struct RussianCheckpoint {
    pub prosody: RussianProsody,
    pub config: RussianConfig,
    pub tokenizer: RussianTokenizer,
    pub controls: RussianControls,
    pub ngrams: HashMap<String, usize>,
    pub exceptions: HashMap<String, (i64, i64)>,
    pub homographs: HashMap<String, Vec<String>>,
    pub acoustic: WeightSet,
    pub accentor: WeightSet,
    pub homosolver: WeightSet,
    pub pqmf: WeightSet,
}

impl RussianCheckpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_package(TorchPackageReader::open(path.as_ref()).context("reading Silero TorchPackage")?)
    }

    pub(crate) fn from_package(package: TorchPackageReader) -> Result<Self> {
        let root = dictionary(package.metadata(), "v5_ru root")?;
        let part = dictionary(only(field(root, "packages")?, "packages")?, "Russian model package")?;
        let symbols = string(part.get("symbols"), "symbols")?.to_owned();
        let symbol_ids = ids(part.get("symbol_to_id"), "symbol_to_id")?;
        ensure!(symbols.chars().count() == 47 && symbols.starts_with("_~|"), "unsupported v5_ru symbols");
        for (idx, ch) in symbols.chars().enumerate() {
            ensure!(symbol_ids.get(&ch.to_string()) == Some(&idx), "inconsistent Russian symbol mapping");
        }
        let speakers = ids(Some(only(field(part, "speaker_to_ids")?, "speaker_to_ids")?), "speaker_to_ids")?;
        ensure!((1..=4096).contains(&speakers.len()) && speakers.values().copied().collect::<std::collections::HashSet<_>>().len() == speakers.len() && speakers.values().all(|&n| n < speakers.len()), "invalid Russian speaker table");
        let config = RussianConfig {
            symbols,
            alphabet: strings(field(part, "alphabet")?, "alphabet")?.concat(),
            sos: string(part.get("sos_token"), "sos_token")?.to_owned(),
            eos: string(part.get("eos_token"), "eos_token")?.to_owned(),
            speakers,
        };
        let accent = dictionary(field(part, "accentor")?, "accentor")?;
        let accentor = dictionary(field(accent, "accentor")?, "accentor model")?;
        let homo = dictionary(field(accent, "homosolver")?, "homosolver")?;
        let accent_id = script_ref(field(accentor, "model")?)?;
        let homo_id = script_ref(field(homo, "model")?)?;
        let tts_id = script_ref(only(field(part, "models")?, "models")?)?;
        ensure!(accent_id != homo_id && accent_id != tts_id && homo_id != tts_id && package.module_ids().len() == 3, "unexpected Russian script topology");
        let (accent_meta, accent_tensors) = package.module(accent_id)?;
        let (homo_meta, homo_tensors) = package.module(homo_id)?;
        let (tts_meta, mut tts_tensors) = package.module(tts_id)?;
        let tts = dictionary(&tts_meta, "acoustic metadata")?;
        let embedded_pqmf = tts_tensors.contains_key("vocoder.pqmf_2.H");
        let pred_flags = |outer: &str, inner: &str| -> Result<&HashMap<String,V>> { dictionary(field(dictionary(field(tts,outer)?,outer)?,inner)?,inner) };
        let prosody = match tts_tensors.len() {
            311 | 317 => RussianProsody::Legacy,
            321 if matches!(pred_flags("dur_predictor","dur_pred")?.get("utt_type_emb"), Some(V::Bool(false))) => {
                for (prefix, type_rows, enabled) in [("dur_predictor.dur_pred", 1, false), ("pitch_predictor.pitch_pred", 6, true)] {
                    let (outer, inner) = prefix.split_once('.').unwrap();
                    let pred = pred_flags(outer,inner)?;
                    ensure!(matches!(pred.get("utt_type_emb"), Some(V::Bool(b)) if *b == enabled) && matches!(pred.get("focus_emb"), Some(V::Bool(b)) if *b == enabled), "unsupported V5 pitch conditioning flags");
                    for (name,rows) in [("type_embedding",type_rows),("focus_embedding",4),("emotion_embedding",15)] {
                        require_tensor(&tts_tensors,&format!("{prefix}.{name}.weight"),DType::F32,&[rows,64])?;
                    }
                }
                RussianProsody::PitchConditioned
            }
            321 => {
                for (prefix, type_rows, emotion_rows, emphasis) in [("dur_predictor.dur_pred", 12, 0, false), ("pitch_predictor.pitch_pred", 6, 15, true)] {
                    let (outer, inner) = prefix.split_once('.').unwrap();
                    let pred = dictionary(field(dictionary(field(tts,outer)?,outer)?,inner)?,inner)?;
                    ensure!(matches!(pred.get("utt_type_emb"), Some(V::Bool(true))) && matches!(pred.get("focus_emb"), Some(V::Bool(true))) && matches!(pred.get("emphasis_emb"), Some(V::Bool(b)) if *b == emphasis), "unsupported V5 prosody predictor flags");
                    for (name,rows) in [("type_embedding",type_rows),("focus_embedding",4),("emotion_embedding",emotion_rows),("emphasis_embedding",2)] {
                        require_tensor(&tts_tensors,&format!("{prefix}.{name}.weight"),DType::F32,&[rows,64])?;
                    }
                }
                RussianProsody::SentenceConditioned
            }
            count => bail!("unsupported neural-stress V5 graph: got {count} tensors"),
        };
        let _ = homo_meta;
        require_tensor(&accent_tensors, "embedding.weight", DType::F32, &[126523,16])?;
        require_tensor(&homo_tensors, "bert.embeddings.word_embeddings.weight", DType::I8, &[83830,312])?;
        require_tensor(&homo_tensors, "bert.scale", DType::F32, &[])?;
        require_tensor(&homo_tensors, "bert.zero_point", DType::F32, &[])?;
        require_tensor(&tts_tensors, "tacotron.embedding.weight", DType::F32, &[47,128])?;
        require_tensor(&tts_tensors, "tacotron.speaker_embedding.weight", DType::F32, &[config.speakers.len(),128])?;
        require_tensor(&tts_tensors, "vocoder.head.istft.window", DType::F32, &[2400])?;
        let embedding = dictionary(field(dictionary(&accent_meta, "accent script")?, "embedding")?, "embedding")?;
        let ngrams: HashMap<_, _> = ids(embedding.get("ngram_dict"), "ngram_dict")?.into_iter().collect();
        ensure!(ngrams.len() == 126523 && ngrams.contains_key("UNK") && ngrams.values().all(|&v| v < 126523), "invalid Accentor ngram ids");
        let exceptions = dictionary(field(accentor, "exceptions")?, "exceptions")?.iter().map(|(k,v)| {
            let items = list(v, "exception")?;
            ensure!(items.len() == 2, "invalid Accentor exception {k}");
            Ok((k.clone(), (integer(&items[0], "stress")?, integer(&items[1], "yo")?)))
        }).collect::<Result<_>>()?;
        let homographs = dictionary(field(homo,"homodict")?, "homodict")?.iter().map(|(k,v)| {
            let variants = strings(v, "homograph variants")?;
            ensure!(variants.len() == 2, "invalid homograph {k}");
            Ok((k.clone(),variants))
        }).collect::<Result<_>>()?;
        let tok = dictionary(field(homo,"tokenizer")?, "tokenizer")?;
        let tokenizer = RussianTokenizer {
            unk_token: string(tok.get("unk_token"), "unk_token")?.into(),
            cls_token: string(tok.get("cls_token"), "cls_token")?.into(),
            sep_token: string(tok.get("sep_token"), "sep_token")?.into(),
            pad_token: string(tok.get("pad_token"), "pad_token")?.into(),
            homo_start_id: index(field(tok,"homo_start_id")?,"homo_start_id")?,
            homo_end_id: index(field(tok,"homo_end_id")?,"homo_end_id")?,
            never_split: strings(field(tok,"never_split")?, "never_split")?,
            tokenize_chinese_chars: match field(tok,"tokenize_chinese_chars")? { V::Bool(b) => *b, _ => bail!("invalid tokenize_chinese_chars") },
            model_max_length: index(field(tok,"model_max_length")?, "model_max_length")?,
            vocab: ids(tok.get("vocab"), "vocab")?.into_iter().collect(),
        };
        ensure!(tokenizer.vocab.len() == 83830 && tokenizer.vocab.values().all(|&v| v < 83830) && tokenizer.homo_start_id < 83830 && tokenizer.homo_end_id < 83830 && tokenizer.model_max_length <= 2048, "invalid Russian tokenizer vocabulary");
        let controls = RussianControls {
            point_id: integer(field(tts,"point_id")?, "point_id")?,
            comma_ids: integers(field(tts,"comma_ids")?, "comma_ids")?,
            punctuation_ids: integers(field(tts,"punkt_symbols")?, "punkt_symbols")?,
            pitch_mean_shift: list(field(tts,"mean_std_coef")?, "mean_std_coef")?.iter().map(number).collect::<Result<_>>()?,
            sil_value: number(field(tts,"sil_value")?)?,
        };
        ensure!(controls.pitch_mean_shift.len() == config.speakers.len(), "invalid Russian prosody metadata");
        let mut filters = HashMap::new();
        for bands in [2,6] {
            let alias = format!("analysis_{bands}");
            let shape = [bands,1,63];
            match embedded_pqmf {
                false => {
                    let path = format!("packages.0.wrapped_jit_v.pqmf_{bands}._buffers.H");
                    require_tensor(package.tensors(), &path, DType::F32, &shape)?;
                    filters.insert(alias, package.tensors()[&path].clone());
                }
                true => {
                    let path = format!("vocoder.pqmf_{bands}.H");
                    require_tensor(&tts_tensors, &path, DType::F32, &shape)?;
                    filters.insert(alias, tts_tensors.remove(&path).unwrap());
                }
            }
        }
        let mut homosolver = WeightSet::new(homo_tensors)?;
        let scale = homosolver.read_f32("bert.scale")?[0];
        let zero_point = homosolver.read_f32("bert.zero_point")?[0];
        homosolver.set_quantization("bert.embeddings.word_embeddings.weight", scale, zero_point)?;
        Ok(Self { prosody, config, tokenizer, controls, ngrams, exceptions, homographs,
            acoustic: WeightSet::new(tts_tensors)?, accentor: WeightSet::new(accent_tensors)?,
            homosolver, pqmf: WeightSet::new(filters)? })
    }
}

fn field<'a>(m: &'a HashMap<String,V>, key: &str) -> Result<&'a V> { m.get(key).with_context(|| format!("missing Russian metadata {key}")) }
fn list<'a>(v: &'a V, what: &str) -> Result<&'a [V]> { match v { V::List(x) => Ok(x), _ => bail!("expected list for {what}") } }
fn only<'a>(v: &'a V, what: &str) -> Result<&'a V> { let x=list(v,what)?; ensure!(x.len()==1,"expected one {what}"); Ok(&x[0]) }
fn integer(v:&V, what:&str)->Result<i64>{match v{V::Int(n)=>Ok(*n),_=>bail!("expected integer for {what}")}}
fn index(v:&V, what:&str)->Result<usize>{usize::try_from(integer(v,what)?).context("invalid index")}
fn integers(v:&V,what:&str)->Result<Vec<i64>>{list(v,what)?.iter().map(|x|integer(x,what)).collect()}
fn strings(v:&V,what:&str)->Result<Vec<String>>{list(v,what)?.iter().map(|x|Ok(string(Some(x),what)?.to_owned())).collect()}
fn number(v:&V)->Result<f32>{let x=match v{V::Float(n)=>*n as f32,V::Int(n)=>*n as f32,_=>bail!("expected numeric metadata")};ensure!(x.is_finite(),"nonfinite metadata");Ok(x)}
fn script_ref(v:&V)->Result<&str>{match v{V::Unsupported(s)=>s.strip_prefix("torchpackage.script:").context("invalid script reference"),_=>bail!("expected script reference")}}
