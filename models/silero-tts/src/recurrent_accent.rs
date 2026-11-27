use anyhow::{Context, Result, ensure};
use silero_package::{QuantizedProjection, RecurrentAccentCheckpoint};
use std::collections::BTreeMap;

const WIDTH: usize = 128;
const GATES: usize = 4 * WIDTH;

struct QuantizedBatch {
    values: Vec<i32>,
    scale: f32,
    width: usize,
}
impl QuantizedBatch {
    fn new(input: &[f32], width: usize) -> Self {
        let (mut min, mut max) = input
            .iter()
            .fold((0f32, 0f32), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        let mut scale64 = (max as f64 - min as f64) / 127.0;
        if scale64 as f32 == 0.0 || (1.0 / scale64 as f32).is_infinite() {
            scale64 = 0.1;
        }
        const MIN_SCALE: f32 = 6.1e-5;
        if scale64 < MIN_SCALE as f64 {
            let amp = MIN_SCALE / scale64 as f32;
            scale64 = MIN_SCALE as f64;
            if min == 0.0 {
                max = MIN_SCALE * 127.0;
            } else if max == 0.0 {
                min = -MIN_SCALE * 127.0;
            } else {
                min *= amp;
                max *= amp;
            }
        }
        let from_min = min as f64 / scale64;
        let from_max = max as f64 / scale64;
        let zero = (if -from_min.abs() < 127.0 - from_max.abs() {
            -from_min
        } else {
            127.0 - from_max
        })
        .round_ties_even()
        .clamp(0., 127.) as i32;
        let scale = scale64 as f32;
        let inv = 1.0 / scale;
        // Quantized dynamic linear packs the affine transform with a single rounding.
        let values = input
            .iter()
            .map(|&v| (v.mul_add(inv, zero as f32).round_ties_even() as i32).clamp(0, 255) - zero)
            .collect();
        Self {
            values,
            scale,
            width,
        }
    }
    fn row(&self, index: usize) -> &[i32] {
        &self.values[index * self.width..(index + 1) * self.width]
    }
}
fn project(p: &QuantizedProjection, batch: &QuantizedBatch, index: usize, output: &mut [f32]) {
    let factor = p.scale * batch.scale;
    let x = batch.row(index);
    for ((w, b), y) in p.weights.chunks_exact(p.input).zip(&p.bias).zip(output) {
        let sum = w
            .iter()
            .zip(x)
            .map(|(&w, &x)| (w as i32 - p.zero) * x)
            .sum::<i32>();
        *y = sum as f32 * factor + b;
    }
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

struct Dense {
    input: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}
impl Dense {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        self.weights
            .chunks_exact(self.input)
            .zip(&self.bias)
            .map(|(w, &b)| w.iter().zip(x).map(|(&w, &x)| w * x).sum::<f32>() + b)
            .collect()
    }
}

pub(crate) struct RecurrentAccent {
    graphemes: BTreeMap<String, usize>,
    exceptions: BTreeMap<String, usize>,
    vowels: String,
    stress_threshold: f32,
    yo_threshold: f32,
    embedding: Vec<f32>,
    cells: Vec<[QuantizedProjection; 2]>,
    attention: Vec<f32>,
    stress: Vec<Dense>,
    yo: Vec<Dense>,
}
impl RecurrentAccent {
    pub(crate) fn load(checkpoint: RecurrentAccentCheckpoint) -> Result<Self> {
        let head = |name: &str| -> Result<Vec<Dense>> {
            [256, 128, 64]
                .into_iter()
                .enumerate()
                .map(|(i, input)| {
                    Ok(Dense {
                        input,
                        weights: checkpoint.head.read_f32(&format!("{name}.{i}.weight"))?,
                        bias: checkpoint.head.read_f32(&format!("{name}.{i}.bias"))?,
                    })
                })
                .collect()
        };
        let stress = head("stress_clf")?;
        let yo = head("yo_clf")?;
        let attention = checkpoint.head.read_f32("attention.attention_weights")?;
        Ok(Self {
            graphemes: checkpoint.graphemes,
            exceptions: checkpoint.exceptions,
            vowels: checkpoint.vowels,
            stress_threshold: checkpoint.stress_threshold,
            yo_threshold: checkpoint.yo_threshold,
            embedding: checkpoint.embedding,
            cells: checkpoint.layers,
            attention,
            stress,
            yo,
        })
    }
    fn predictions(&self, words: &[String]) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let mut sequences: Vec<Vec<f32>> = words
            .iter()
            .map(|w| -> Result<_> {
                ensure!(
                    !w.is_empty() && w.len() <= 1000,
                    "invalid recurrent accentor word length"
                );
                let mut rows = Vec::with_capacity(w.len() * 300);
                for c in w.chars() {
                    let id = self
                        .graphemes
                        .get(&c.to_string())
                        .context("unknown recurrent accentor grapheme")?;
                    rows.extend_from_slice(&self.embedding[id * 300..(id + 1) * 300]);
                }
                Ok(rows)
            })
            .collect::<Result<_>>()?;
        let lengths: Vec<usize> = sequences.iter().map(|x| x.len() / 300).collect();
        let maxlen = *lengths.iter().max().unwrap();
        let total: usize = lengths.iter().sum();
        ensure!(total <= 4096, "recurrent accentor input budget exceeded");
        let offsets: Vec<usize> = lengths
            .iter()
            .scan(0, |s, &n| {
                let old = *s;
                *s += n;
                Some(old)
            })
            .collect();
        let n = words.len();
        let mut gates = [0.; GATES];
        let mut recurrent = [0.; GATES];
        for layer in 0..4 {
            let dim = if layer == 0 { 300 } else { 2 * WIDTH };
            let joined: Vec<f32> = sequences.iter().flatten().copied().collect();
            let xq = QuantizedBatch::new(&joined, dim);
            let mut next: Vec<Vec<f32>> = lengths.iter().map(|l| vec![0.; l * 2 * WIDTH]).collect();
            for direction in 0..2 {
                let [wi, wh] = &self.cells[layer * 2 + direction];
                let mut hidden = vec![vec![0.; WIDTH]; n];
                let mut cell = hidden.clone();
                for step in 0..maxlen {
                    // PackedSequence reverse visits the global time axis in reverse, not each word's end at once.
                    let t = if direction == 0 {
                        step
                    } else {
                        maxlen - 1 - step
                    };
                    let active: Vec<usize> = (0..n).filter(|&i| t < lengths[i]).collect();
                    let hflat: Vec<f32> = active
                        .iter()
                        .flat_map(|&i| hidden[i].iter().copied())
                        .collect();
                    let hq = QuantizedBatch::new(&hflat, WIDTH);
                    for (j, &i) in active.iter().enumerate() {
                        project(wi, &xq, offsets[i] + t, &mut gates);
                        project(wh, &hq, j, &mut recurrent);
                        for k in 0..GATES {
                            gates[k] += recurrent[k];
                        }
                        for k in 0..WIDTH {
                            cell[i][k] = sigmoid(gates[WIDTH + k]) * cell[i][k]
                                + sigmoid(gates[k]) * gates[2 * WIDTH + k].tanh();
                            hidden[i][k] = sigmoid(gates[3 * WIDTH + k]) * cell[i][k].tanh();
                        }
                        let p = t * 2 * WIDTH + direction * WIDTH;
                        next[i][p..p + WIDTH].copy_from_slice(&hidden[i]);
                    }
                }
            }
            sequences = next;
        }
        Ok(sequences
            .iter()
            .zip(lengths)
            .map(|(rows, len)| {
                let scores: Vec<f32> = rows
                    .chunks_exact(2 * WIDTH)
                    .map(|row| {
                        row.iter()
                            .zip(&self.attention)
                            .map(|(&x, &w)| x * w)
                            .sum::<f32>()
                            .tanh()
                    })
                    .collect();
                let max = scores
                    .iter()
                    .copied()
                    .fold(if len < maxlen { 0. } else { f32::NEG_INFINITY }, f32::max);
                let denom = scores.iter().map(|x| (x - max).exp()).sum::<f32>()
                    + (maxlen - len) as f32 * (-max).exp();
                let mut pooled = vec![0.; 2 * WIDTH];
                for (row, score) in rows.chunks_exact(2 * WIDTH).zip(scores) {
                    let a = (score - max).exp() / denom;
                    for (out, x) in pooled.iter_mut().zip(row) {
                        *out += a * x;
                    }
                }
                let forward =
                    |head: &[Dense]| head.iter().fold(pooled.clone(), |x, p| p.forward(&x));
                (forward(&self.stress), forward(&self.yo))
            })
            .collect())
    }
    pub(crate) fn accentuate(&self, sentence: &str, put_yo: bool) -> Result<String> {
        let clean: String = sentence
            .chars()
            .filter(|&c| c.is_ascii_lowercase() || ('1'..='9').contains(&c) || c == '-' || c == ' ')
            .collect();
        let words: Vec<String> = clean.split_whitespace().map(str::to_owned).collect();
        let preds = self.predictions(&words)?;
        let mut index = 0;
        let mut result = Vec::new();
        for word in sentence.split_whitespace() {
            if let Some(&pos) = self.exceptions.get(word) {
                let mut word = word.to_owned();
                word.insert(pos, '+');
                result.push(word);
                continue;
            }
            if !word
                .chars()
                .any(|c| c.is_ascii_lowercase() || ('1'..='9').contains(&c))
                || word.contains('+') && word.contains('1')
            {
                result.push(word.to_owned());
                continue;
            }
            let (stress, yo) = preds
                .get(index)
                .context("recurrent accentor prediction alignment")?;
            index += 1;
            let best = |v: &[f32]| {
                let (i, &m) = v
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap();
                (i, 1.0 / v.iter().map(|x| (x - m).exp()).sum::<f32>())
            };
            let (s, sp) = best(stress);
            let (y, yp) = best(yo);
            let mut chars: Vec<char> = word.chars().collect();
            let vowels: Vec<usize> = chars
                .iter()
                .enumerate()
                .filter_map(|(i, c)| self.vowels.contains(*c).then_some(i))
                .collect();
            let spos = s.checked_sub(1).and_then(|s| vowels.get(s).copied());
            let ypos = y.checked_sub(1).and_then(|y| vowels.get(y).copied());
            let strong = s != 0 && sp > self.stress_threshold;
            if strong && put_yo && yp > self.yo_threshold && spos == ypos && !word.contains('1') {
                if let Some(pos) = ypos {
                    if chars[pos] == 'e' {
                        chars[pos] = '1';
                    }
                }
            }
            if !word.contains('+') {
                let pos = if vowels.len() == 1 {
                    vowels.first().copied()
                } else if strong {
                    spos
                } else {
                    None
                };
                if let Some(pos) = pos {
                    chars.insert(pos, '+');
                }
            }
            result.push(chars.into_iter().collect());
        }
        Ok(result.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reduced_range_quantizer_handles_saturation_and_zero() {
        let q = QuantizedBatch::new(&[-0.7615942, 0.0, 0.7615942], 3);
        assert_eq!(q.values[1], 0);
        assert!(q.values[0] < 0 && q.values[2] > 0);
        assert!((q.scale - (2.0 * 0.7615942 / 127.0)).abs() < 1e-7);
        let z = QuantizedBatch::new(&[0.; 8], 4);
        assert!(z.values.iter().all(|&x| x == 0));
        assert_eq!(z.scale, 0.1);
    }
    #[test]
    fn affine_quantization_rounds_after_fused_zero_point() {
        let extremum = 0.9640276f32;
        let q = QuantizedBatch::new(&[-extremum, 0.0, extremum], 3);
        assert_eq!(q.values, [-63, 0, 64]);
    }
    #[derive(serde::Deserialize)]
    struct Case {
        text: String,
        sentence: String,
        words: Vec<String>,
        accented: String,
        stress: Vec<Vec<f32>>,
        yo: Vec<Vec<f32>>,
    }
    #[test]
    #[ignore = "requires real original recurrent accentors and independent PyTorch fixtures"]
    fn official_recurrent_accentor_parity() {
        let root = std::path::PathBuf::from(std::env::var_os("SILERO_MODEL_MATRIX").unwrap());
        let fixture =
            std::path::PathBuf::from(std::env::var_os("SILERO_RECURRENT_FIXTURE").unwrap());
        for name in ["aidar_v2", "ru_v3"] {
            let checkpoint =
                match silero_package::OfficialCheckpoint::open(root.join(format!("{name}.pt")))
                    .unwrap()
                {
                    silero_package::OfficialCheckpoint::V2(x) => x.recurrent_accent.unwrap(),
                    silero_package::OfficialCheckpoint::V3(x) => x.recurrent_accent.unwrap(),
                    _ => panic!("unexpected graph"),
                };
            let model = RecurrentAccent::load(checkpoint).unwrap();
            let cases: Vec<Case> = serde_json::from_slice(
                &std::fs::read(fixture.join(format!("{name}.json"))).unwrap(),
            )
            .unwrap();
            for case in cases {
                let got = model.predictions(&case.words).unwrap();
                assert_eq!(got.len(), case.stress.len());
                assert_eq!(got.len(), case.yo.len());
                let mut max = 0f32;
                for ((s, y), (rs, ry)) in got.iter().zip(case.stress.iter().zip(&case.yo)) {
                    assert_eq!(s.len(), rs.len());
                    assert_eq!(y.len(), ry.len());
                    for (a, b) in s.iter().chain(y).zip(rs.iter().chain(ry)) {
                        assert!(a.is_finite());
                        max = max.max((a - b).abs());
                    }
                }
                let accented = model.accentuate(&case.sentence, true).unwrap();
                println!(
                    "{name} {} logits_max={max:.6} got={accented:?} expected={:?}",
                    case.text, case.accented
                );
                assert_eq!(accented, case.accented, "{}", case.text);
                assert!(
                    max <= 1e-4,
                    "{}: recurrent accentor logits error {max}",
                    case.text
                );
            }
        }
    }
}
