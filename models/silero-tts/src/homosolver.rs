use anyhow::{Result, bail};
use regex::Regex;
use serde_json;
use std::{collections::HashMap, fs, path::Path};
use crate::tokenizer::WordPieceTokenizer;
use burn::tensor::{Device, Int, Tensor, TensorData, activation::{gelu, relu, softmax}, module::{embedding, layer_norm, linear}};

use crate::store::TensorStore;

struct Dense {
    weight: Tensor<2>,
    bias: Tensor<1>,
}

impl Dense {
    fn load(store: &TensorStore, name: &str, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: store.float::<2>(&format!("{name}.weight"), device)?.transpose(),
            bias: store.float::<1>(&format!("{name}.bias"), device)?,
        })
    }

    fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        linear(input, self.weight.clone(), Some(self.bias.clone()))
    }

    fn forward_2d(&self, input: Tensor<2>) -> Tensor<2> {
        linear(input, self.weight.clone(), Some(self.bias.clone()))
    }
}

struct BertLayer {
    query: Dense,
    key: Dense,
    value: Dense,
    attn_output: Dense,
    norm1_weight: Tensor<1>,
    norm1_bias: Tensor<1>,
    intermediate: Dense,
    output: Dense,
    norm2_weight: Tensor<1>,
    norm2_bias: Tensor<1>,
}

impl BertLayer {
    fn load(weights: &TensorStore, idx: usize, device: &Device) -> Result<Self> {
        let prefix = format!("bert.encoder.layer.{idx}");
        let attn = format!("{prefix}.attention");
        Ok(Self {
            query: Dense::load(weights, &format!("{attn}.self.query"), device)?,
            key: Dense::load(weights, &format!("{attn}.self.key"), device)?,
            value: Dense::load(weights, &format!("{attn}.self.value"), device)?,
            attn_output: Dense::load(weights, &format!("{attn}.output.dense"), device)?,
            norm1_weight: weights.float(&format!("{attn}.output.LayerNorm.weight"), device)?,
            norm1_bias: weights.float(&format!("{attn}.output.LayerNorm.bias"), device)?,
            intermediate: Dense::load(weights, &format!("{prefix}.intermediate.dense"), device)?,
            output: Dense::load(weights, &format!("{prefix}.output.dense"), device)?,
            norm2_weight: weights.float(&format!("{prefix}.output.LayerNorm.weight"), device)?,
            norm2_bias: weights.float(&format!("{prefix}.output.LayerNorm.bias"), device)?,
        })
    }

    fn forward(&self, hidden: Tensor<3>) -> Tensor<3> {
        let [batch, len, channels] = hidden.dims();
        let head_count = 12;
        let head_dim = channels / head_count;
        let q = self.query.forward(hidden.clone()).reshape([batch, len, head_count, head_dim]).swap_dims(1, 2);
        let k = self.key.forward(hidden.clone()).reshape([batch, len, head_count, head_dim]).swap_dims(1, 2);
        let v = self.value.forward(hidden.clone()).reshape([batch, len, head_count, head_dim]).swap_dims(1, 2);
        let scores = q.matmul(k.swap_dims(2, 3)) / (head_dim as f32).sqrt();
        let context = softmax(scores, 3).matmul(v).swap_dims(1, 2).reshape([batch, len, channels]);
        let attended = self.attn_output.forward(context);
        let x = layer_norm(hidden + attended, self.norm1_weight.clone(), Some(self.norm1_bias.clone()), 1e-12);
        let intermediate = gelu(self.intermediate.forward(x.clone()));
        layer_norm(x + self.output.forward(intermediate), self.norm2_weight.clone(), Some(self.norm2_bias.clone()), 1e-12)
    }
}

pub struct HomoSolverNetwork {
    word_embedding: Tensor<2>,
    position_embedding: Tensor<2>,
    token_type_embedding: Tensor<2>,
    embedding_norm_weight: Tensor<1>,
    embedding_norm_bias: Tensor<1>,
    layers: Vec<BertLayer>,
    classifier1: Dense,
    classifier2: Dense,
    device: Device,
}

impl HomoSolverNetwork {
    pub fn load(weights: &TensorStore, device: &Device) -> Result<Self> {
        Ok(Self {
            word_embedding: weights.float("bert.embeddings.word_embeddings.weight", device)?,
            position_embedding: weights.float("bert.embeddings.position_embeddings.weight", device)?,
            token_type_embedding: weights.float("bert.embeddings.token_type_embeddings.weight", device)?,
            embedding_norm_weight: weights.float("bert.embeddings.LayerNorm.weight", device)?,
            embedding_norm_bias: weights.float("bert.embeddings.LayerNorm.bias", device)?,
            layers: (0..3).map(|i| BertLayer::load(weights, i, device)).collect::<Result<_>>()?,
            classifier1: Dense::load(weights, "homo_clf.0", device)?,
            classifier2: Dense::load(weights, "homo_clf.3", device)?,
            device: device.clone(),
        })
    }

    pub fn embeddings(&self, ids: &[i64]) -> Result<Tensor<3>> {
        if ids.is_empty() || ids.len() > 2048 { bail!("BERT token count must be 1..=2048"); }
        let len = ids.len();
        let sequence = Tensor::<2, Int>::from_data(TensorData::new(ids.to_vec(), [1, len]), &self.device);
        let positions = Tensor::<2, Int>::from_data(
            TensorData::new((0..len as i64).collect::<Vec<_>>(), [1, len]), &self.device);
        let types = Tensor::<2, Int>::zeros([1, len], &self.device);
        let combined = embedding(self.word_embedding.clone(), sequence)
            + embedding(self.position_embedding.clone(), positions)
            + embedding(self.token_type_embedding.clone(), types);
        Ok(layer_norm(combined, self.embedding_norm_weight.clone(),
            Some(self.embedding_norm_bias.clone()), 1e-12))
    }

    pub fn hidden(&self, ids: &[i64]) -> Result<Tensor<3>> {
        let mut hidden = self.embeddings(ids)?;
        for layer in &self.layers { hidden = layer.forward(hidden); }
        Ok(hidden)
    }

    pub fn infer_logits(&self, ids: &[i64], start: usize, end: usize) -> Result<Tensor<2>> {
        if start + 1 >= end || end > ids.len() {
            bail!("HomoSolver markers must bound one or more tokens");
        }
        let hidden = self.hidden(ids)?;
        let marker = hidden.clone().slice([0..1, start..start + 1, 0..312]).reshape([1, 312]);
        let word = hidden.slice([0..1, start + 1..end, 0..312])
            .mean_dim(1).reshape([1, 312]);
        let pair = Tensor::cat(vec![marker, word], 1);
        let result = self.classifier1.forward_2d(pair);
        Ok(self.classifier2.forward_2d(relu(result)))
    }
}


/// Context-aware binary homograph resolver, reconstructed from SileroStress.
pub struct HomoSolver {
    network: HomoSolverNetwork,
    tokenizer: WordPieceTokenizer,
    variants: HashMap<String, Vec<String>>,
    word_pattern: Regex,
}

impl HomoSolver {
    pub fn load(model_dir: &Path, device: &Device) -> Result<Self> {
        let weights = TensorStore::open(&model_dir.join("homosolver.safetensors"))?;
        Self::from_parts(&weights, WordPieceTokenizer::open(model_dir)?,
            serde_json::from_slice(&fs::read(model_dir.join("homographs.json"))?)?, device)
    }

    pub fn from_parts(weights: &TensorStore, tokenizer: WordPieceTokenizer,
        variants: HashMap<String, Vec<String>>, device: &Device) -> Result<Self> {
        Ok(Self {
            network: HomoSolverNetwork::load(weights, device)?,
            tokenizer,
            variants,
            word_pattern: Regex::new(r"(?i)[а-яё+]+")?,
        })
    }

    pub fn is_homograph(&self, word: &str) -> bool {
        self.variants.contains_key(&word.to_lowercase())
    }

    pub fn variant_words(&self) -> impl Iterator<Item = (&String, &Vec<String>)> {
        self.variants.iter()
    }

    pub fn accentuate(&self, text: &str, put_stress: bool, put_yo: bool,
        stress_single_vowel: bool) -> Result<String> {
        if !put_stress && !put_yo { return Ok(text.to_string()); }
        let mut replacements = Vec::new();
        for hit in self.word_pattern.find_iter(text) {
            let word = hit.as_str();
            if !word.chars().any(|ch| ('а'..='я').contains(&ch.to_lowercase().next().unwrap_or(ch)) || ch == 'ё' || ch == 'Ё') {
                continue;
            }
            let Some(variants) = self.variants.get(&word.to_lowercase()) else { continue; };
            if variants.len() != 2 { bail!("Silero homograph {word:?} must have exactly two variants"); }
            let mark = format!("{} [HOMO] {} [/HOMO] {}", &text[..hit.start()], word, &text[hit.end()..]);
            let ids = self.tokenizer.encode(&mark)?;
            if ids.len() > self.tokenizer.config().model_max_length {
                bail!("HomoSolver context exceeds tokenizer model_max_length");
            }
            let start = ids.iter().position(|id| *id == self.tokenizer.config().homo_start_id)
                .ok_or_else(|| anyhow::anyhow!("Missing [HOMO] tokenizer marker"))?;
            let end = ids.iter().position(|id| *id == self.tokenizer.config().homo_end_id)
                .ok_or_else(|| anyhow::anyhow!("Missing [/HOMO] tokenizer marker"))?;
            let input = ids.iter().map(|v| *v as i64).collect::<Vec<_>>();
            let logit = self.network.infer_logits(&input, start, end)?
                .into_data().try_to_vec::<f32>()?[0];
            let predicted = (logit > 0.) as usize;
            let mut variants = variants.clone();
            variants.sort();
            let chosen = &variants[predicted];
            let stress_idx = chosen.chars().position(|ch| ch == '+')
                .ok_or_else(|| anyhow::anyhow!("No stress token in homograph variant"))?;
            let mut chars: Vec<char> = chosen.chars().filter(|ch| *ch != '+').collect();
            if !put_yo { for ch in &mut chars { if *ch == 'ё' { *ch = 'е'; } } }
            let source: Vec<char> = word.chars().collect();
            if chars.len() != source.len() {
                bail!("HomoSolver output changed number of characters in {word:?}");
            }
            for (dst, src) in chars.iter_mut().zip(source) {
                if src.is_uppercase() { *dst = dst.to_uppercase().next().unwrap_or(*dst); }
            }
            let syllables = chars.iter().filter(|ch| "аоуыэиеяёю".contains(ch.to_lowercase().next().unwrap_or(**ch))).count();
            if (syllables > 1 || stress_single_vowel) && put_stress {
                chars.insert(stress_idx, '+');
            }
            replacements.push((hit.start(), hit.end(), chars.into_iter().collect::<String>()));
        }
        let mut result = text.to_string();
        for (start, end, replacement) in replacements.into_iter().rev() {
            result.replace_range(start..end, &replacement);
        }
        Ok(result)
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod parity {
    use super::*;
    use serde::Deserialize;
    use std::{fs, path::Path};

    #[derive(Deserialize)]
    struct Fixture { cases: Vec<Case>, sentences: Vec<Sentence> }

    #[derive(Deserialize)]
    struct Sentence {
        sentence: String,
        result: String,
    }

    #[derive(Deserialize)]
    struct Case {
        raw_mark: String,
        tokens: Vec<i64>,
        start: usize,
        end: usize,
        embedding_slice: Vec<f32>,
        hidden_slice: Vec<f32>,
        logit: f32,
    }

    #[test]
    fn bert_and_classifier_match_pytorch() {
        let Ok(dir) = std::env::var("SILERO_PARITY_DIR") else { return };
        let dir = Path::new(&dir);
        let fixture: Fixture = serde_json::from_slice(&fs::read(dir.join("homosolver-parity.json")).unwrap()).unwrap();
        let weights = TensorStore::open(&dir.join("homosolver.safetensors")).unwrap();
        let model = HomoSolverNetwork::load(&weights, &Device::flex()).unwrap();
        let tokenizer = WordPieceTokenizer::open(dir).unwrap();
        for (i, case) in fixture.cases.iter().enumerate() {
            let actual_tokens = tokenizer.encode(&case.raw_mark).unwrap();
            assert_eq!(actual_tokens, case.tokens.iter().map(|v| *v as usize).collect::<Vec<_>>(),
                "Tokenizer differs from PyTorch for case {i}");
            let embeddings = model.embeddings(&case.tokens).unwrap().into_data().try_to_vec::<f32>().unwrap();
            let embed_err = embeddings.iter().take(32).zip(&case.embedding_slice)
                .map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
            let hidden = model.hidden(&case.tokens).unwrap().into_data().try_to_vec::<f32>().unwrap();
            let hidden_err = hidden.iter().take(32).zip(&case.hidden_slice)
                .map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
            let logit = model.infer_logits(&case.tokens, case.start, case.end).unwrap()
                .into_data().try_to_vec::<f32>().unwrap()[0];
            let logit_err = (logit-case.logit).abs();
            println!("HomoSolver case={i}: embeddings={embed_err:.7}, hidden={hidden_err:.7}, logits={logit_err:.7}");
            assert!(embed_err < 1e-3 && hidden_err < 1e-3 && logit_err < 1e-3);
        }
    }
    #[test]
    fn homograph_sentence_replacements_match_pytorch() {
        let Ok(dir) = std::env::var("SILERO_PARITY_DIR") else { return };
        let dir = Path::new(&dir);
        let fixture: Fixture = serde_json::from_slice(&fs::read(dir.join("homosolver-parity.json")).unwrap()).unwrap();
        let model = HomoSolver::load(dir, &Device::flex()).unwrap();
        for item in fixture.sentences.iter().take(5) {
            let result = model.accentuate(&item.sentence, true, true, true).unwrap();
            assert_eq!(result, item.result, "Sentence mismatch: {:?}", item.sentence);
        }
    }

}
