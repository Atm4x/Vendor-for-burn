use crate::{mel, model::Whisper, weights::Weights};
use anyhow::{Context, Result, anyhow, bail};
use burn::tensor::{Device, Tensor, TensorData};
use std::path::Path;
use tokenizers::Tokenizer;

const LANGS: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

pub struct Special {
    pub sot: u32,
    pub eot: u32,
    pub transcribe: Option<u32>,
    pub translate: Option<u32>,
    pub no_timestamps: u32,
    pub langs: Vec<(String, u32)>,
}

impl Special {
    fn from_tokenizer(tok: &Tokenizer) -> Result<Self> {
        let id = |s: &str| tok.token_to_id(s);
        let langs = LANGS
            .iter()
            .filter_map(|l| id(&format!("<|{l}|>")).map(|i| (l.to_string(), i)))
            .collect();
        Ok(Self {
            sot: id("<|startoftranscript|>").context("tokenizer has no <|startoftranscript|>")?,
            eot: id("<|endoftext|>").context("tokenizer has no <|endoftext|>")?,
            transcribe: id("<|transcribe|>"),
            translate: id("<|translate|>"),
            no_timestamps: id("<|notimestamps|>").context("tokenizer has no <|notimestamps|>")?,
            langs,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub language: Option<String>,
    pub translate: bool,
    pub max_new_tokens: usize,
    pub verbose: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            language: None,
            translate: false,
            max_new_tokens: 224,
            verbose: false,
        }
    }
}

pub struct Transcriber {
    pub model: Whisper,
    tok: Tokenizer,
    sp: Special,
    bias: Tensor<1>,
    bias_first: Tensor<1>,
}

impl Transcriber {
    pub fn load(
        weights: &Path,
        tokenizer: &Path,
        generation_config: &Path,
        device: &Device,
    ) -> Result<Self> {
        let weights = Weights::load(weights)?;
        let model = Whisper::load(weights, device)?;
        let tok = Tokenizer::from_file(tokenizer)
            .map_err(|e| anyhow!("cannot load tokenizer.json: {e}"))?;
        let sp = Special::from_tokenizer(&tok)?;

        let mut suppress = vec![false; model.n_vocab];
        for (i, s) in suppress.iter_mut().enumerate() {
            *s = i as u32 > sp.eot;
        }
        anyhow::ensure!(
            sp.eot < model.n_vocab as u32
                && sp.sot < model.n_vocab as u32
                && sp
                    .langs
                    .iter()
                    .all(|(_, id)| (*id as usize) < model.n_vocab),
            "tokenizer does not match checkpoint vocabulary"
        );
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(generation_config)?)?;
        if let Some(list) = config["suppress_tokens"].as_array() {
            for id in list.iter().filter_map(serde_json::Value::as_u64) {
                if let Some(suppress) = suppress.get_mut(id as usize) {
                    *suppress = true;
                }
            }
        }
        let make_bias = |mask: &[bool]| {
            let v: Vec<f32> = mask
                .iter()
                .map(|&s| if s { f32::NEG_INFINITY } else { 0.0 })
                .collect();
            Tensor::<1>::from_data(TensorData::new(v, [mask.len()]), device)
        };
        let bias = make_bias(&suppress);
        let mut first = suppress.clone();
        first[sp.eot as usize] = true;
        if let Some(sid) = tok.token_to_id("Д ") {
            if let Some(f) = first.get_mut(sid as usize) {
                *f = true;
            }
        }
        let bias_first = make_bias(&first);
        Ok(Self {
            model,
            tok,
            sp,
            bias,
            bias_first,
        })
    }

    pub fn is_multilingual(&self) -> bool {
        !self.sp.langs.is_empty() && self.sp.transcribe.is_some()
    }

    pub fn transcribe(&self, samples: &[f32], opts: &Options) -> Result<String> {
        let mut lang = match &opts.language {
            Some(code) => {
                if !self.is_multilingual() {
                    bail!("this model is English-only; drop --language");
                }
                let found = self.sp.langs.iter().find(|(c, _)| c == code);
                Some(
                    found
                        .with_context(|| format!("unknown language `{code}`"))?
                        .1,
                )
            }
            None => None,
        };

        let mut parts: Vec<String> = Vec::new();
        for (ci, range) in chunking::split(samples).into_iter().enumerate() {
            let chunk = &samples[range.clone()];
            if opts.verbose {
                let sr = mel::SAMPLE_RATE as f32;
                eprintln!(
                    "[chunk {ci}] {:.1} s .. {:.1} s",
                    range.start as f32 / sr,
                    range.end as f32 / sr
                );
            }
            let peak = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if peak < 1e-4 || (ci > 0 && chunk.len() < mel::SAMPLE_RATE / 10) {
                continue;
            }
            let t_mel = std::time::Instant::now();
            let m = mel::log_mel_spectrogram(chunk, self.model.n_mels);
            let mel_ms = t_mel.elapsed().as_secs_f64() * 1e3;
            let t_enc = std::time::Instant::now();
            let enc = self.model.encode(&m);
            if opts.verbose {
                let _ = enc.clone().slice([0..1, 0..1]).into_data();
                eprintln!(
                    "[chunk {ci}] mel {:.0} ms, encoder {:.0} ms",
                    mel_ms,
                    t_enc.elapsed().as_secs_f64() * 1e3
                );
            }

            if lang.is_none() && self.is_multilingual() {
                let mut st = self.model.start_decoding(&enc);
                let logits = self.model.decode_step(&mut st, &[self.sp.sot])?;
                let (code, id) = self
                    .sp
                    .langs
                    .iter()
                    .max_by(|a, b| logits[a.1 as usize].total_cmp(&logits[b.1 as usize]))
                    .map(|(c, i)| (c.clone(), *i))
                    .context("no language tokens")?;
                if opts.verbose {
                    eprintln!("detected language: {code}");
                }
                lang = Some(id);
            }

            let text = self.decode_chunk(&enc, lang, opts)?;
            if opts.verbose {
                eprintln!("[chunk {ci}] {text}");
            }
            let text = text.trim().to_string();
            if !text.is_empty() {
                parts.push(text);
            }
        }
        Ok(parts.join(" "))
    }

    fn decode_chunk(
        &self,
        enc: &burn::tensor::Tensor<2>,
        lang: Option<u32>,
        opts: &Options,
    ) -> Result<String> {
        let sp = &self.sp;
        let mut prompt = vec![sp.sot];
        if let Some(l) = lang {
            prompt.push(l);
        }
        if self.is_multilingual() {
            let task = if opts.translate {
                sp.translate
            } else {
                sp.transcribe
            };
            prompt.extend(task);
        }
        prompt.push(sp.no_timestamps);

        let t_dec = std::time::Instant::now();
        let mut st = self.model.start_decoding(enc);
        let mut next = self
            .model
            .decode_step_argmax(&mut st, &prompt, &self.bias_first)?;
        let mut out: Vec<u32> = Vec::new();

        while out.len() < opts.max_new_tokens {
            if next == sp.eot || st.position() >= self.model.n_text_ctx {
                break;
            }
            out.push(next);
            if out.len() >= opts.max_new_tokens {
                break;
            }
            next = self
                .model
                .decode_step_argmax(&mut st, &[next], &self.bias)?;
        }
        if opts.verbose && !out.is_empty() {
            let ms = t_dec.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "decoder: {} tokens in {:.0} ms ({:.1} ms/token)",
                out.len(),
                ms,
                ms / out.len() as f64
            );
        }
        self.tok
            .decode(&out, true)
            .map_err(|e| anyhow!("detokenize: {e}"))
    }
}

mod chunking {
    use crate::mel::{N_SAMPLES, SAMPLE_RATE};
    use std::ops::Range;

    const SEARCH: usize = 5 * SAMPLE_RATE;
    const WINDOW: usize = SAMPLE_RATE / 10;
    const STEP: usize = SAMPLE_RATE / 100;

    pub fn split(samples: &[f32]) -> Vec<Range<usize>> {
        let mut out = Vec::new();
        let mut start = 0;
        while start < samples.len() {
            let limit = start + N_SAMPLES;
            if limit >= samples.len() {
                out.push(start..samples.len());
                break;
            }
            let end = quietest_cut(samples, limit - SEARCH, limit);
            out.push(start..end);
            start = end;
        }
        out
    }

    fn quietest_cut(samples: &[f32], from: usize, to: usize) -> usize {
        let part = &samples[from..to];
        let mut cum = Vec::with_capacity(part.len() + 1);
        cum.push(0.0f64);
        for &x in part {
            let last = *cum.last().unwrap();
            cum.push(last + (x as f64) * (x as f64));
        }
        let (mut best_pos, mut best_e) = (0usize, f64::INFINITY);
        let mut pos = 0;
        while pos + WINDOW <= part.len() {
            let e = cum[pos + WINDOW] - cum[pos];
            if e <= best_e {
                best_e = e;
                best_pos = pos;
            }
            pos += STEP;
        }
        from + best_pos + WINDOW / 2
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn noise(n: usize) -> Vec<f32> {
            let mut x = 12345u32;
            (0..n)
                .map(|_| {
                    x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                    (x >> 8) as f32 / (1u32 << 24) as f32 * 0.6 - 0.3
                })
                .collect()
        }

        fn silence(x: &mut [f32], from_s: f32, to_s: f32) {
            let (a, b) = (
                (from_s * SAMPLE_RATE as f32) as usize,
                (to_s * SAMPLE_RATE as f32) as usize,
            );
            x[a..b].iter_mut().for_each(|v| *v = 0.0);
        }

        fn check_cover(r: &[Range<usize>], len: usize) {
            assert_eq!(r.first().unwrap().start, 0);
            assert_eq!(r.last().unwrap().end, len);
            for w in r.windows(2) {
                assert_eq!(w[0].end, w[1].start, "chunks must be contiguous");
            }
            assert!(
                r.iter().all(|c| c.len() <= N_SAMPLES && !c.is_empty()),
                "chunk too long / empty"
            );
        }

        #[test]
        fn cuts_inside_pauses() {
            let len = 80 * SAMPLE_RATE;
            let mut x = noise(len);
            silence(&mut x, 27.0, 27.6);
            silence(&mut x, 55.0, 55.4);
            let r = split(&x);
            check_cover(&r, len);
            assert_eq!(r.len(), 3);
            let s = SAMPLE_RATE as f32;
            let (c1, c2) = (r[0].end as f32 / s, r[1].end as f32 / s);
            assert!(
                (27.0..=27.6).contains(&c1),
                "first cut at {c1} s, not in the pause"
            );
            assert!(
                (55.0..=55.4).contains(&c2),
                "second cut at {c2} s, not in the pause"
            );
        }

        #[test]
        fn short_and_exact_inputs_stay_whole() {
            for n in [1, SAMPLE_RATE, N_SAMPLES] {
                let r = split(&noise(n));
                assert_eq!(r, vec![0..n]);
            }
            assert!(split(&[]).is_empty());
        }

        #[test]
        fn no_pause_still_cuts_within_limits() {
            let len = 31 * SAMPLE_RATE;
            let r = split(&noise(len));
            check_cover(&r, len);
            assert_eq!(r.len(), 2);
            assert!(r[0].end >= N_SAMPLES - SEARCH && r[0].end <= N_SAMPLES);
        }
    }
}
