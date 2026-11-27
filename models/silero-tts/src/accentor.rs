use anyhow::{Context, Result, bail};
use burn::tensor::{Device, Tensor, TensorData, activation::relu, module::linear};
use serde::Deserialize;
use std::{collections::HashMap, fs, path::Path};

use crate::store::TensorStore;

const VOWELS: &str = "аоуыэиеяёю";

#[derive(Clone, Debug, Default)]
pub struct AccentOptions {
    pub put_stress: bool,
    pub put_yo: bool,
    pub stress_single_vowel: bool,
    pub skip_stress_words: Vec<String>,
    pub skip_yo_words: Vec<String>,
}

impl AccentOptions {
    pub fn original_defaults() -> Self {
        Self {
            put_stress: true,
            put_yo: true,
            stress_single_vowel: true,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct ExceptionEntry(i64, i64);

#[derive(Clone, Debug)]
pub struct AccentPrediction {
    pub stress_logits: Vec<f32>,
    pub yo_logits: Vec<f32>,
}

impl AccentPrediction {
    fn best(logits: &[f32]) -> (usize, f32) {
        let (index, value) = logits.iter().copied().enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1)).unwrap_or((0, 0.));
        let total = logits.iter().map(|v| (v - value).exp()).sum::<f32>();
        (index, 1. / total)
    }
}

struct Classifier {
    layers: Vec<(Tensor<2>, Tensor<1>)>,
}

impl Classifier {
    fn load(store: &TensorStore, name: &str, device: &Device) -> Result<Self> {
        let layers = [0, 2, 4, 6].into_iter().map(|idx| {
            Ok((store.float::<2>(&format!("{name}.{idx}.weight"), device)?.transpose(),
                store.float::<1>(&format!("{name}.{idx}.bias"), device)?))
        }).collect::<Result<_>>()?;
        Ok(Self { layers })
    }

    fn forward(&self, mut x: Tensor<2>) -> Tensor<2> {
        for (idx, (weight, bias)) in self.layers.iter().enumerate() {
            x = linear(x, weight.clone(), Some(bias.clone()));
            if idx < 3 {
                x = relu(x);
            }
        }
        x
    }
}

#[derive(Clone, Debug)]
struct Token {
    raw: String,
    clean: String,
    needs_prediction: bool,
}

pub struct Accentor {
    ngrams: HashMap<String, usize>,
    exceptions: HashMap<String, ExceptionEntry>,
    embedding: Vec<f32>,
    stress: Classifier,
    yo: Classifier,
    device: Device,
}

impl Accentor {
    pub fn load(model_dir: &Path, device: &Device) -> Result<Self> {
        let weights = TensorStore::open(&model_dir.join("accentor.safetensors"))?;
        let ngrams: HashMap<String, usize> = serde_json::from_slice(
            &fs::read(model_dir.join("accentor-ngrams.json"))?)?;
        let exceptions: HashMap<String, ExceptionEntry> = serde_json::from_slice(
            &fs::read(model_dir.join("accentor-exceptions.json"))?)?;
        Self::from_parts(&weights, ngrams, exceptions.into_iter().map(|(k, v)| (k, (v.0, v.1))).collect(), device)
    }

    pub fn from_parts(weights: &TensorStore, ngrams: HashMap<String, usize>,
        exceptions: HashMap<String, (i64, i64)>, device: &Device) -> Result<Self> {
        let embedding = weights.float_data("embedding.weight")?;
        let exceptions = exceptions.into_iter().map(|(k, (stress, yo))| (k, ExceptionEntry(stress, yo))).collect();
        if embedding.len() != ngrams.len() * 16 || ngrams.values().any(|&id| id >= ngrams.len()) {
            bail!("Accentor ngram table shape does not match the checkpoint dictionary");
        }
        if !ngrams.contains_key("UNK") {
            bail!("Accentor ngram dictionary has no UNK");
        }
        Ok(Self {
            ngrams,
            exceptions,
            embedding,
            stress: Classifier::load(weights, "stress_clf", device)?,
            yo: Classifier::load(weights, "yo_clf", device)?,
            device: device.clone(),
        })
    }

    fn ngram_ids(&self, word: &str) -> Vec<usize> {
        let chars: Vec<char> = format!("<{word}>").chars().collect();
        let word_length = word.chars().count();
        let mut ids = Vec::new();
        for n in 1..=word_length + 3 {
            for subgram in chars.windows(n) {
                let gram: String = subgram.iter().collect();
                if let Some(id) = self.ngrams.get(&gram) {
                    ids.push(*id);
                }
            }
        }
        if word_length == 0 {
            if let Some(id) = self.ngrams.get("") { ids.push(*id); }
        }
        if ids.is_empty() {
            ids.push(self.ngrams["UNK"]);
        }
        ids
    }

    pub fn embedding_vectors(&self, words: &[String]) -> Vec<f32> {
        let mut results = Vec::with_capacity(words.len() * 16);
        for word in words {
            let ids = self.ngram_ids(word);
            let mut average = [0f32; 16];
            for id in &ids {
                let row = &self.embedding[id * 16..id * 16 + 16];
                for (result, source) in average.iter_mut().zip(row) {
                    *result += source;
                }
            }
            for value in average {
                results.push(value / ids.len() as f32);
            }
        }
        results
    }

    pub fn predict(&self, words: &[String]) -> Result<Vec<AccentPrediction>> {
        if words.is_empty() { return Ok(Vec::new()); }
        let embedded = self.embedding_vectors(words);
        let input = Tensor::<2>::from_data(
            TensorData::new(embedded, [words.len(), 16]), &self.device);
        let stress_logits = self.stress.forward(input.clone()).into_data().try_to_vec::<f32>()?;
        let yo_logits = self.yo.forward(input).into_data().try_to_vec::<f32>()?;
        let predictions = (0..words.len()).map(|i| AccentPrediction {
            stress_logits: stress_logits[i * 10..(i + 1) * 10].to_vec(),
            yo_logits: yo_logits[i * 7..(i + 1) * 7].to_vec(),
        }).collect();
        Ok(predictions)
    }

    fn tokenize(sentence: &str) -> Vec<Token> {
        let mut pieces = Vec::<String>::new();
        let mut current = String::new();
        let mut whitespace = None;
        for ch in sentence.chars() {
            let is_space = ch.is_whitespace();
            if whitespace.is_some_and(|old| old != is_space) {
                pieces.push(std::mem::take(&mut current));
            }
            current.push(ch);
            whitespace = Some(is_space);
        }
        if !current.is_empty() {
            pieces.push(current);
        }
        let mut tokens = Vec::new();
        for piece in pieces {
            let parts: Vec<&str> = piece.split('-').collect();
            let count = parts.len();
            for (i, part) in parts.into_iter().enumerate() {
                let raw = if i + 1 < count { format!("{part}-") } else { part.to_string() };
                let clean: String = raw.to_lowercase().chars()
                    .filter(|c| ('а'..='я').contains(c) || *c == 'ё').collect();
                let needs_prediction = !clean.is_empty() && (i + 1 < count || clean != "то" || count == 1);
                tokens.push(Token {raw, clean, needs_prediction});
            }
        }
        tokens
    }

    fn exception(&self, raw: &str, entry: &ExceptionEntry) -> String {
        let clean = raw.chars().collect::<Vec<_>>();
        if raw.contains('+') {
            let positions: Vec<usize> = clean.iter().enumerate().filter_map(|(i, c)| (*c == '+').then_some(i)).collect();
            let mut chars: Vec<char> = clean.iter().copied().filter(|ch| *ch != '+').collect();
            if entry.1 >= 0 && positions.contains(&(entry.1 as usize + 1)) {
                let i = entry.1 as usize;
                if i < chars.len() {
                    chars[i] = if chars[i].is_uppercase() { 'Ё' } else { 'ё' };
                }
            }
            for position in positions {
                chars.insert(position.min(chars.len()), '+');
            }
            chars.into_iter().collect()
        } else {
            let mut chars = clean;
            if entry.1 >= 0 && (entry.1 as usize) < chars.len() {
                let i = entry.1 as usize;
                chars[i] = if chars[i].is_uppercase() { 'Ё' } else { 'ё' };
            }
            chars.insert((entry.0.max(0) as usize).min(chars.len()), '+');
            chars.into_iter().collect()
        }
    }

    fn predicted_token(&self, token: &Token, pred: &AccentPrediction, opts: &AccentOptions) -> String {
        let raw = &token.raw;
        let raw_lower = raw.to_lowercase();
        let have_stress = raw_lower.contains('+');
        let have_yo = raw_lower.contains('ё');
        if have_stress && have_yo { return raw.clone(); }
        if !have_stress && have_yo && opts.put_stress {
            let vowel_count = raw_lower.chars().filter(|c| VOWELS.contains(*c)).count();
            if (vowel_count == 1 && !opts.stress_single_vowel)
                || opts.skip_stress_words.contains(&token.clean.replace('ё', "е")) { return raw.clone(); }
            let mut chars: Vec<char> = raw.chars().collect();
            for i in (0..chars.len()).rev() {
                if chars[i] == 'ё' || chars[i] == 'Ё' { chars.insert(i, '+'); }
            }
            return chars.into_iter().collect();
        }
        if let Some(exception) = self.exceptions.get(&token.clean) {
            return self.exception(raw, exception);
        }
        let (stress_id, stress_probability) = AccentPrediction::best(&pred.stress_logits);
        let (yo_id, yo_probability) = AccentPrediction::best(&pred.yo_logits);
        let mut set_stress = opts.put_stress && stress_probability > 0.5 && !have_stress
            && !opts.skip_stress_words.contains(&token.clean.replace('ё', "е"));
        let set_yo = opts.put_yo && yo_probability > 0.5
            && !opts.skip_yo_words.contains(&token.clean.replace('ё', "е"));

        let chars: Vec<char> = raw_lower.chars().collect();
        let vowels: Vec<usize> = chars.iter().enumerate()
            .filter_map(|(i, c)| VOWELS.contains(*c).then_some(i)).collect();
        let ye: Vec<usize> = chars.iter().enumerate()
            .filter_map(|(i, c)| (*c == 'е').then_some(i)).collect();
        if vowels.is_empty() { return raw.clone(); }
        let mut stress_positions = if have_stress {
            let mut positions = Vec::new();
            for part in raw_lower.split('+') {
                positions.push(part.chars().filter(|c| VOWELS.contains(*c)).count());
            }
            positions.into_iter().filter_map(|i| vowels.get(i).copied()).collect::<Vec<_>>()
        } else { vowels.get(stress_id).copied().into_iter().collect::<Vec<_>>() };
        let yo_position = (yo_id > 0).then(|| ye.get(yo_id.saturating_sub(1)).copied()).flatten();
        let mut output: Vec<char> = raw.chars().collect();
        if let Some(i) = yo_position {
            if stress_positions.contains(&i) && set_yo && output[i] == 'е' { output[i] = 'ё'; }
            if stress_positions.contains(&i) && set_yo && output[i] == 'Е' { output[i] = 'Ё'; }
        }
        if vowels.len() == 1 {
            stress_positions = vec![vowels[0]];
            set_stress = opts.stress_single_vowel && opts.put_stress;
        }
        if !have_stress && set_stress {
            for (offset, position) in stress_positions.into_iter().enumerate() {
                output.insert((position + offset).min(output.len()), '+');
            }
        }
        output.into_iter().collect()
    }

    /// V4's serialized frontend classifies the ASCII-transliterated words,
    /// rather than the Unicode strings used by the V5 stress pipeline.
    pub(crate) fn accentuate_ascii(&self, sentence: &str, vowels: &str,
        stop_words: &std::collections::HashSet<String>, put_yo: bool) -> Result<String> {
        let clean: String = sentence.chars().filter(|c| c.is_ascii_lowercase() || ('1'..='9').contains(c) || *c == '-' || *c == ' ').collect();
        let mut tokens = Vec::new();
        for (word, cleaned) in sentence.split(' ').zip(clean.split(' ')) {
            let parts: Vec<_> = word.split('-').collect();
            let cleaned_parts: Vec<_> = cleaned.split('-').collect();
            for (i, (raw, clean)) in parts.iter().zip(cleaned_parts).enumerate() {
                let last = i + 1 == parts.len();
                let raw = if last { raw.to_string() } else { format!("{raw}-") };
                let needs_prediction = !clean.is_empty() && (!last || parts.len() == 1 || !stop_words.contains(clean));
                tokens.push(Token { raw, clean: clean.to_owned(), needs_prediction });
            }
        }
        let inputs: Vec<_> = tokens.iter().map(|t| t.clean.clone()).collect();
        let predictions = self.predict(&inputs)?;
        let words: Vec<_> = tokens.iter().zip(predictions.iter()).map(|(token, pred)| {
            if !token.needs_prediction { return token.raw.clone(); }
            let raw = &token.raw;
            let have_stress = raw.contains('+');
            let have_yo = raw.contains('1');
            if have_stress && have_yo { return raw.clone(); }
            let mut chars: Vec<_> = raw.chars().collect();
            if !have_stress && have_yo {
                for i in (0..chars.len()).rev() { if chars[i] == '1' { chars.insert(i, '+'); } }
                return chars.into_iter().collect();
            }
            if let Some(entry) = self.exceptions.get(&token.clean) {
                if have_stress {
                    let positions: Vec<_> = chars.iter().enumerate().filter_map(|(i, c)| (*c == '+').then_some(i)).collect();
                    chars.retain(|c| *c != '+');
                    if put_yo && entry.1 >= 0 && positions.contains(&(entry.1 as usize + 1)) && (entry.1 as usize) < chars.len() {
                        chars[entry.1 as usize] = '1';
                    }
                    for i in positions { chars.insert(i.min(chars.len()), '+'); }
                } else {
                    if put_yo && entry.1 >= 0 && (entry.1 as usize) < chars.len() { chars[entry.1 as usize] = '1'; }
                    chars.insert((entry.0.max(0) as usize).min(chars.len()), '+');
                }
                return chars.into_iter().collect();
            }
            let (stress_id, stress_prob) = AccentPrediction::best(&pred.stress_logits);
            let (yo_id, yo_prob) = AccentPrediction::best(&pred.yo_logits);
            let vowel_positions: Vec<_> = chars.iter().enumerate().filter_map(|(i,c)| vowels.contains(*c).then_some(i)).collect();
            if vowel_positions.is_empty() { return raw.clone(); }
            let ye_positions: Vec<_> = chars.iter().enumerate().filter_map(|(i,c)| (*c == 'e').then_some(i)).collect();
            let mut stress_positions = if have_stress {
                raw.split('+').filter_map(|part| vowel_positions.get(part.chars().filter(|c| vowels.contains(*c)).count()).copied()).collect::<Vec<_>>()
            } else { vowel_positions.get(stress_id).copied().into_iter().collect() };
            if put_yo && yo_prob > 0.5 && (stress_prob > 0.5 || have_stress) && yo_id > 0 {
                if let Some(&i) = ye_positions.get(yo_id-1).filter(|i| stress_positions.contains(i)) { chars[i] = '1'; }
            }
            let single_vowel = vowel_positions.len() == 1;
            if single_vowel { stress_positions = vec![vowel_positions[0]]; }
            if !have_stress && (stress_prob > 0.5 || single_vowel) {
                for (offset,i) in stress_positions.into_iter().enumerate() { chars.insert((i+offset).min(chars.len()), '+'); }
            }
            chars.into_iter().collect()
        }).collect();
        Ok(words.join(" ").replace("- ", "-"))
    }

    pub fn accentuate(&self, sentence: &str, opts: &AccentOptions) -> Result<String> {
        if !(opts.put_stress || opts.put_yo) { return Ok(sentence.to_string()); }
        let tokens = Self::tokenize(sentence);
        let inputs = tokens.iter().map(|t| t.clean.clone()).collect::<Vec<_>>();
        let predictions = self.predict(&inputs).context("Accentor classifier inference")?;
        let mut result = String::new();
        for (token, prediction) in tokens.iter().zip(&predictions) {
            if token.needs_prediction {
                result.push_str(&self.predicted_token(token, prediction, opts));
            } else {
                result.push_str(&token.raw);
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_punctuation_and_hyphens() {
        let text = "Я - кот, а ты? кто-то  недалеко!";
        let t = Accentor::tokenize(text);
        assert_eq!(t.iter().map(|x| x.raw.as_str()).collect::<String>(), text);
        assert!(t.iter().any(|x| x.raw == "то" && !x.needs_prediction));
        assert!(t.iter().any(|x| x.raw == "кто-" && x.needs_prediction));
    }

    #[test]
    fn ngram_generation_uses_bounded_unicode_windows() {
        let model = Accentor {
            ngrams: [("<".into(), 1), ("ё".into(), 2), (">".into(), 3), ("<ё".into(), 4), ("ё>".into(), 5), ("<ё>".into(), 6), ("UNK".into(), 0)]
                .into_iter().collect(),
            exceptions: HashMap::new(), embedding: vec![0.; 16 * 7],
            stress: Classifier {layers: Vec::new()}, yo: Classifier {layers: Vec::new()}, device: Device::default(),
        };
        assert_eq!(model.ngram_ids("ё"), [1, 2, 3, 4, 5, 6]);
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod parity {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixture {
        words: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        stress_logits: Vec<Vec<f32>>,
        yo_logits: Vec<Vec<f32>>,
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        sentence: String,
        result: Option<String>,
    }

    #[test]
    fn original_accentor_logits_and_text() {
        let Ok(folder) = std::env::var("SILERO_PARITY_DIR") else { return };
        let path = Path::new(&folder);
        let fixture: Fixture = serde_json::from_slice(&fs::read(path.join("accentor-parity.json")).unwrap()).unwrap();
        let model = Accentor::load(path, &Device::flex()).unwrap();
        let vectors = model.embedding_vectors(&fixture.words);
        let mut embedding_error = 0.0f32;
        for (actual, reference) in vectors.chunks_exact(16).zip(&fixture.embeddings) {
            for (a, b) in actual.iter().zip(reference) { embedding_error = embedding_error.max((a - b).abs()); }
        }
        println!("Accentor embedding max_abs={embedding_error:.8}");
        assert!(embedding_error < 2e-5);
        let prediction = model.predict(&fixture.words).unwrap();
        let mut stress_error = 0.0f32;
        let mut yo_error = 0.0f32;
        for ((p, s), y) in prediction.iter().zip(&fixture.stress_logits).zip(&fixture.yo_logits) {
            for (a,b) in p.stress_logits.iter().zip(s) { stress_error = stress_error.max((a-b).abs()); }
            for (a,b) in p.yo_logits.iter().zip(y) { yo_error = yo_error.max((a-b).abs()); }
        }
        println!("Accentor MLP stress max_abs={stress_error:.8}; yo max_abs={yo_error:.8}");
        assert!(stress_error < 0.1 && yo_error < 0.1);
        for case in &fixture.cases {
            let Some(reference) = &case.result else { continue };
            let actual = model.accentuate(&case.sentence, &AccentOptions::original_defaults()).unwrap();
            assert_eq!(&actual, reference, "Accentor mismatch for {:?}", case.sentence);
        }
    }
}
