use crate::{DType, PickleValue as V, WeightSet, dictionary, ids, require_tensor, string};
use anyhow::{Context, Result, bail, ensure};
use pytorch_reader::TorchPackageReader;
use std::{
    collections::{BTreeMap, HashSet},
    path::Path,
};

pub struct V2Checkpoint {
    pub standalone: bool,
    pub encoded_symbols: String,
    pub ascii_symbols: BTreeMap<char, String>,
    pub recurrent_accent: Option<crate::RecurrentAccentCheckpoint>,
    pub symbols: String,
    pub speakers: BTreeMap<String, usize>,
    pub multi: bool,
    pub weights: WeightSet,
    pub max_steps: usize,
    pub gate_threshold: f64,
    pub sample_rates: Vec<u32>,
}

impl V2Checkpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_package(TorchPackageReader::open(path)?)
    }

    pub(crate) fn from_package(package: TorchPackageReader) -> Result<Self> {
        let meta = dictionary(package.metadata(), "V2 package")?;
        let symbols = string(meta.get("symbols"), "V2 symbols")?.to_owned();
        let n = symbols.chars().count();
        ensure!(
            symbols.starts_with("_~")
                && (4..=4096).contains(&n)
                && symbols.chars().collect::<HashSet<_>>().len() == n,
            "invalid V2 alphabet"
        );
        let mut ascii_symbols = BTreeMap::new();
        if let Some(value) = meta.get("ru_ascii_dict") {
            for (k,v) in dictionary(value,"V2 ASCII map")? {
                let mut chars = k.chars(); let ch = chars.next().context("empty V2 mapping key")?;
                ensure!(chars.next().is_none(), "invalid V2 mapping key");
                ascii_symbols.insert(ch,string(Some(v),"V2 ASCII value")?.to_owned());
            }
        }
        let encoded_symbols: String = symbols.chars().map(|c| ascii_symbols.get(&c).cloned().unwrap_or_else(||c.to_string())).collect();
        ensure!(encoded_symbols.chars().count()==n && encoded_symbols.chars().collect::<HashSet<_>>().len()==n,"invalid V2 encoded symbols");
        let multi = meta.contains_key("speaker_to_id");
        let speakers = if multi {
            ids(meta.get("speaker_to_id"), "V2 speakers")?
        } else {
            BTreeMap::from([("default".to_owned(), 0)])
        };
        let Some(V::Unsupported(reference)) = meta.get("model") else {
            bail!("missing V2 script reference")
        };
        let id = reference
            .strip_prefix("torchpackage.script:")
            .context("invalid V2 script reference")?;
        ensure!(
            package.module_ids().len() == 1,
            "unexpected V2 script topology"
        );
        let (root, tensors) = package.module(id)?;
        let root = dictionary(&root, "V2 script")?;
        let recurrent_accent = if root.contains_key("accentor") { Some(crate::recurrent_accent::load_recurrent_accent(root,&tensors)?) } else { None };
        if recurrent_accent.is_some() {
            let table = ids(root.get("symbol_to_id"),"V2 script symbols")?;
            ensure!(encoded_symbols.chars().enumerate().all(|(i,c)| table.get(&c.to_string())==Some(&i)),"inconsistent V2 encoded alphabet");
        }
        Self::from_state(root, tensors, symbols, encoded_symbols, ascii_symbols, recurrent_accent, speakers, multi, None)
    }

    /// V1 stores the graph in a standalone JIT archive and its alphabet in the upstream catalogue.
    pub fn open_standalone(path: &Path, symbols: &str, sample_rate: u32) -> Result<Self> {
        ensure!(symbols.starts_with("_~") && (4..=4096).contains(&symbols.chars().count()) && symbols.chars().collect::<HashSet<_>>().len()==symbols.chars().count(), "invalid standalone V1 alphabet");
        let (root, tensors) = pytorch_reader::read_script_state(path)?;
        let root = dictionary(&root, "standalone V1 script")?;
        Self::from_state(root, tensors, symbols.to_owned(), symbols.to_owned(), BTreeMap::new(), None, BTreeMap::from([("default".into(),0)]), false, Some(sample_rate))
    }

    fn from_state(root: &std::collections::HashMap<String, V>, mut tensors: std::collections::HashMap<String, crate::Tensor>, symbols: String, encoded_symbols: String, ascii_symbols: BTreeMap<char,String>, recurrent_accent: Option<crate::RecurrentAccentCheckpoint>, speakers: BTreeMap<String,usize>, multi: bool, standalone: Option<u32>) -> Result<Self> {
        let n = symbols.chars().count();
        let tac = dictionary(
            root.get("tacotron").context("missing V2 Tacotron2")?,
            "Tacotron2",
        )?;
        let decoder = dictionary(
            tac.get("decoder").context("missing V2 decoder")?,
            "V2 decoder",
        )?;
        for (name, value) in [
            ("n_mel_channels", 128),
            ("n_frames_per_step", 1),
            ("encoder_embedding_dim", 512),
            ("attention_rnn_dim", 1024),
            ("decoder_rnn_dim", 1024),
            ("prenet_dim", 256),
        ] {
            ensure!(
                matches!(decoder.get(name), Some(V::Int(v)) if *v==value),
                "unsupported V2 decoder {name}"
            );
        }
        let Some(V::Int(max_steps)) = decoder.get("max_decoder_steps") else {
            bail!("missing V2 max steps")
        };
        let Some(V::Float(gate_threshold)) = decoder.get("gate_threshold") else {
            bail!("missing V2 gate threshold")
        };
        ensure!(
            (1..=2000).contains(max_steps) && (0.0..1.0).contains(gate_threshold),
            "invalid V2 decoding limits"
        );
        require_tensor(
            &tensors,
            "tacotron.embedding.weight",
            DType::F32,
            &[n, if multi { 384 } else { 512 }],
        )?;
        if multi {
            let t = tensors
                .get("tacotron.speaker_embedding.weight")
                .context("missing V2 speaker embedding")?;
            ensure!(
                t.dtype() == DType::F32
                    && t.shape().len() == 2
                    && t.shape()[1] == 128
                    && !speakers.is_empty()
                    && speakers.values().all(|&v| v < t.shape()[0]),
                "invalid V2 speaker embedding"
            );
        }
        let voc = dictionary(
            root.get("vocoder").context("missing V2 vocoder")?,
            "V2 vocoder",
        )?;
        let rates = if let Some(rate) = standalone {
            ensure!(matches!(rate, 8000 | 16000), "invalid V1 sample rate");
            ensure!(matches!(voc.get("num_kernels"),Some(V::Int(3))) && matches!(voc.get("num_upsamples"),Some(V::Int(3))), "unsupported standalone V1 vocoder topology");
            require_tensor(&tensors, "vocoder.conv_pre_layer.weight", DType::F32, &[512,128,7])?;
            require_tensor(&tensors, "vocoder.conv_post.weight", DType::F32, &[1,64,7])?;
            for (index, (input, output, stride)) in [(512,256,8),(256,128,5),(128,64,5)].into_iter().enumerate() {
                require_tensor(&tensors, &format!("vocoder.ups.{index}.weight"), DType::F32, &[input,output,2*stride])?;
            }
            vec![rate]
        } else {
            ensure!(matches!(voc.get("multiband"), Some(V::Bool(true))) && matches!(voc.get("resample"), Some(V::Int(1))), "unsupported V2 vocoder mode");
            let up = tensors.get("vocoder.ups.0.weight").context("missing V2 upsample weights")?;
            match up.shape() {
                [512, 256, 4] if !voc.contains_key("ups_u") => vec![8000, 16000],
                [512, 256, 6] if voc.contains_key("ups_u") => vec![8000, 24000, 48000],
                _ => bail!("unsupported V2 vocoder topology"),
            }
        };
        tensors.retain(|name, _| {
            !name.contains("._flat_weights.")
                && (name.starts_with("tacotron.embedding.")
                    || name.starts_with("tacotron.speaker_embedding.")
                    || name.starts_with("tacotron.encoder.")
                    || name.starts_with("tacotron.postnet.")
                    || name.starts_with("vocoder.")
                    || name.starts_with("tacotron.decoder.")
                        && !matches!(
                            name.strip_prefix("tacotron.decoder.").unwrap_or(""),
                            "attention_hidden"
                                | "attention_cell"
                                | "decoder_hidden"
                                | "decoder_cell"
                                | "attention_weights"
                                | "attention_weights_cum"
                                | "attention_context"
                                | "memory"
                                | "processed_memory"
                        ))
        });
        Ok(Self {
            standalone: standalone.is_some(),
            encoded_symbols, ascii_symbols, recurrent_accent,
            symbols,
            speakers,
            multi,
            weights: WeightSet::new(tensors)?,
            max_steps: *max_steps as usize,
            gate_threshold: *gate_threshold,
            sample_rates: rates,
        })
    }
}

pub fn encode_v2(symbols: &str, text: &str) -> Result<Vec<i64>> {
    let ids: BTreeMap<char, usize> = symbols.chars().enumerate().map(|(i, c)| (c, i)).collect();
    let lower = text.to_lowercase();
    let filtered: String = lower
        .chars()
        .filter(|c| ids.get(c).is_some_and(|&i| i >= 2))
        .collect();
    let mut normalized = filtered.split_whitespace().collect::<Vec<_>>().join(" ");
    ensure!(!normalized.is_empty(), "empty V2 text after normalization");
    if !normalized.ends_with(['.', '!', '?']) {
        normalized.push('.');
    }
    normalized.push('~');
    normalized
        .chars()
        .map(|c| {
            ids.get(&c)
                .map(|&id| id as i64)
                .with_context(|| format!("unknown V2 symbol {c:?}"))
        })
        .collect()
}
