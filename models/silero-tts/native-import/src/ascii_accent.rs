use anyhow::{Context, Result, bail, ensure};
use crate::{dictionary, ids, string, require_tensor, DType, PickleValue as V, Tensor, WeightSet};
use std::collections::{HashMap, HashSet};

pub struct AsciiAccentCheckpoint {
    pub weights: WeightSet,
    pub ngrams: HashMap<String, usize>,
    pub exceptions: HashMap<String, (i64, i64)>,
    pub vowels: String,
    pub stop_words: HashSet<String>,
}

pub(crate) fn load_ascii_accent(script: &HashMap<String,V>, tensors: &HashMap<String,Tensor>) -> Result<AsciiAccentCheckpoint> {
    let node = dictionary(script.get("accentor").context("missing ASCII accentor")?, "ASCII accentor")?;
    let emb = dictionary(node.get("embedding").context("missing ASCII accentor embedding")?, "ASCII accentor embedding")?;
    let ngrams: HashMap<_, _> = ids(emb.get("ngram_dict"), "ASCII ngrams")?.into_iter().collect();
    ensure!(ngrams.len() <= 1_000_000 && ngrams.contains_key("UNK") && ngrams.values().all(|&id| id < ngrams.len()), "invalid ASCII ngram IDs");
    let embedding = tensors.get("accentor.embedding.weight").context("missing ASCII accentor embedding weight")?;
    ensure!(embedding.shape() == [ngrams.len(),16] && matches!(embedding.dtype(), DType::F32 | DType::I8), "invalid ASCII embedding shape or type");
    for (name, output) in [("stress_clf",10),("yo_clf",7)] {
        for (index, input, output) in [(0,16,64),(2,64,128),(4,128,64),(6,64,output)] {
            require_tensor(&tensors, &format!("accentor.{name}.{index}.weight"), DType::F32, &[output,input])?;
            require_tensor(&tensors, &format!("accentor.{name}.{index}.bias"), DType::F32, &[output])?;
        }
    }
    let exceptions = dictionary(script.get("exceptions").context("missing ASCII exceptions")?, "ASCII exceptions")?.iter().map(|(k,v)| {
        let V::List(items) = v else { bail!("invalid ASCII exception") };
        let [V::Int(stress),V::Int(yo)] = items.as_slice() else { bail!("invalid ASCII exception positions") };
        Ok((k.clone(), (*stress,*yo)))
    }).collect::<Result<_>>()?;
    let vowels = string(script.get("vowels"), "ASCII vowels")?.to_owned();
    ensure!(string(script.get("stress_token"), "stress token")? == "+" && string(script.get("yo_token"), "yo token")? == "1" && string(script.get("e_token"), "e token")? == "e", "unsupported ASCII accentor alphabet");
    ensure!(matches!(script.get("stress_trs"), Some(V::Float(x)) if *x == 0.5) && matches!(script.get("yo_trs"), Some(V::Float(x)) if *x == 0.5), "unsupported ASCII accent thresholds");
    let Some(V::List(stop)) = script.get("stop_words") else { bail!("invalid ASCII accent stop words") };
    let stop_words = stop.iter().map(|v| Ok(string(Some(v), "stop word")?.to_owned())).collect::<Result<_>>()?;
    let mut weights = WeightSet::new(tensors.iter().filter_map(|(k,v)| k.strip_prefix("accentor.").map(|s| (s.to_owned(),v.clone()))).collect())?;
    if embedding.dtype() == DType::I8 {
        require_tensor(&tensors, "accentor.scale", DType::F32, &[])?;
        require_tensor(&tensors, "accentor.zero_point", DType::F32, &[])?;
        let scale = weights.read_f32("scale")?[0];
        let zero = weights.read_f32("zero_point")?[0];
        weights.set_quantization("embedding.weight", scale, zero)?;
    }
    Ok(AsciiAccentCheckpoint { weights, ngrams, exceptions, vowels, stop_words })
}
