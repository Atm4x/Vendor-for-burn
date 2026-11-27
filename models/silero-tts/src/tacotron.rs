use crate::store::TensorStore;
use anyhow::{Result, ensure};
use burn::tensor::{
    Device, Distribution, Int, Tensor, TensorData,
    activation::{relu, sigmoid, softmax},
    module::{conv1d, embedding, linear},
    ops::ConvOptions,
};

struct Projection {
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
}
impl Projection {
    fn load(
        store: &TensorStore,
        name: &str,
        input: usize,
        output: usize,
        biased: bool,
        device: &Device,
    ) -> Result<Self> {
        let weight = store.float::<2>(&format!("{name}.weight"), device)?;
        let bias = if biased {
            Some(store.float::<1>(&format!("{name}.bias"), device)?)
        } else {
            None
        };
        ensure!(
            weight.dims() == [output, input] && bias.as_ref().is_none_or(|b| b.dims() == [output]),
            "invalid Tacotron projection {name}"
        );
        Ok(Self {
            weight: weight.transpose(),
            bias,
        })
    }
    fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        linear(x, self.weight.clone(), self.bias.clone())
    }
}

struct LstmCell {
    input: Tensor<2>,
    recurrent: Tensor<2>,
    bias_input: Tensor<1>,
    bias_recurrent: Tensor<1>,
    width: usize,
}
impl LstmCell {
    fn load(
        store: &TensorStore,
        prefix: &str,
        suffix: &str,
        input: usize,
        width: usize,
        device: &Device,
    ) -> Result<Self> {
        let weight = |kind| store.float::<2>(&format!("{prefix}.weight_{kind}{suffix}"), device);
        let bias = |kind| store.float::<1>(&format!("{prefix}.bias_{kind}{suffix}"), device);
        let (wi, wh, bi, bh) = (weight("ih")?, weight("hh")?, bias("ih")?, bias("hh")?);
        ensure!(
            wi.dims() == [4 * width, input]
                && wh.dims() == [4 * width, width]
                && bi.dims() == [4 * width]
                && bh.dims() == [4 * width],
            "invalid LSTM shape {prefix}{suffix}"
        );
        Ok(Self {
            input: wi.transpose(),
            recurrent: wh.transpose(),
            bias_input: bi,
            bias_recurrent: bh,
            width,
        })
    }
    fn projected(&self, input: Tensor<2>) -> Tensor<2> {
        linear(input, self.input.clone(), Some(self.bias_input.clone()))
    }
    fn step_projected(
        &self,
        gates: Tensor<2>,
        hidden: Tensor<2>,
        cell: Tensor<2>,
    ) -> (Tensor<2>, Tensor<2>) {
        let gates = gates
            + linear(
                hidden,
                self.recurrent.clone(),
                Some(self.bias_recurrent.clone()),
            );
        let [batch, _] = gates.dims();
        let w = self.width;
        let part = |i| gates.clone().slice([0..batch, i * w..(i + 1) * w]);
        let c = sigmoid(part(1)) * cell + sigmoid(part(0)) * part(2).tanh();
        let h = sigmoid(part(3)) * c.clone().tanh();
        (h, c)
    }
    fn step(&self, input: Tensor<2>, hidden: Tensor<2>, cell: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        self.step_projected(self.projected(input), hidden, cell)
    }
    fn sequence(&self, input: Tensor<2>, reverse: bool, device: &Device) -> Tensor<2> {
        let len = input.dims()[0];
        let projected = self.projected(input);
        let (mut h, mut c) = (
            Tensor::zeros([1, self.width], device),
            Tensor::zeros([1, self.width], device),
        );
        let mut rows = Vec::with_capacity(len);
        for i in 0..len {
            let j = if reverse { len - 1 - i } else { i };
            (h, c) =
                self.step_projected(projected.clone().slice([j..j + 1, 0..4 * self.width]), h, c);
            rows.push(h.clone());
        }
        if reverse {
            rows.reverse();
        }
        Tensor::cat(rows, 0)
    }
}

struct ConvNorm {
    weight: Tensor<3>,
    bias: Tensor<1>,
    scale: Tensor<3>,
    shift: Tensor<3>,
}
impl ConvNorm {
    fn load(
        store: &TensorStore,
        prefix: &str,
        input: usize,
        output: usize,
        device: &Device,
    ) -> Result<Self> {
        let w = store.float::<3>(&format!("{prefix}.0.conv.weight"), device)?;
        let b = store.float::<1>(&format!("{prefix}.0.conv.bias"), device)?;
        let bn = |name| store.float::<1>(&format!("{prefix}.1.{name}"), device);
        let (gain, offset, mean, var) = (
            bn("weight")?,
            bn("bias")?,
            bn("running_mean")?,
            bn("running_var")?,
        );
        ensure!(
            w.dims() == [output, input, 5]
                && [&b, &gain, &offset, &mean, &var]
                    .iter()
                    .all(|t| t.dims() == [output]),
            "invalid Tacotron ConvNorm {prefix}"
        );
        let scale = gain / (var + 1e-5).sqrt();
        let shift = offset - mean * scale.clone();
        Ok(Self {
            weight: w,
            bias: b,
            scale: scale.reshape([1, output, 1]),
            shift: shift.reshape([1, output, 1]),
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        conv1d(
            x,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new([1], [2], [1], 1),
        ) * self.scale.clone()
            + self.shift.clone()
    }
}

struct Attention {
    query: Projection,
    memory: Projection,
    score: Projection,
    location: Tensor<3>,
    location_projection: Projection,
}
impl Attention {
    fn load(store: &TensorStore, device: &Device) -> Result<Self> {
        let p = "tacotron.decoder.attention_layer";
        let lin = |n, input, output| {
            Projection::load(
                store,
                &format!("{p}.{n}.linear_layer"),
                input,
                output,
                false,
                device,
            )
        };
        let location = store.float::<3>(
            &format!("{p}.location_layer.location_conv.conv.weight"),
            device,
        )?;
        ensure!(
            location.dims() == [32, 2, 31],
            "invalid Tacotron location conv"
        );
        Ok(Self {
            query: lin("query_layer", 1024, 128)?,
            memory: lin("memory_layer", 512, 128)?,
            score: lin("v", 128, 1)?,
            location,
            location_projection: lin("location_layer.location_dense", 32, 128)?,
        })
    }
    fn forward(
        &self,
        hidden: Tensor<2>,
        memory: Tensor<2>,
        processed: Tensor<2>,
        previous: Tensor<2>,
        cumulative: Tensor<2>,
    ) -> (Tensor<2>, Tensor<2>) {
        let len = memory.dims()[0];
        let locations = Tensor::cat(
            vec![
                previous.unsqueeze_dim::<3>(1),
                cumulative.unsqueeze_dim::<3>(1),
            ],
            1,
        );
        let loc = conv1d(
            locations,
            self.location.clone(),
            None,
            ConvOptions::new([1], [15], [1], 1),
        )
        .reshape([32, len])
        .transpose();
        let energies = self
            .score
            .forward(
                (self.query.forward(hidden) + self.location_projection.forward(loc) + processed)
                    .tanh(),
            )
            .transpose();
        let weights = softmax(energies, 1);
        let context = weights.clone().matmul(memory);
        (context, weights)
    }
}

pub(crate) struct Tacotron2 {
    symbols: Tensor<2>,
    speakers: Option<Tensor<2>>,
    encoder: Vec<ConvNorm>,
    forward_lstm: LstmCell,
    reverse_lstm: LstmCell,
    prenet: [Projection; 2],
    attention_cell: LstmCell,
    attention: Attention,
    decoder_cell: LstmCell,
    mel: Projection,
    gate: Projection,
    postnet: Vec<ConvNorm>,
    max_steps: usize,
    gate_threshold: f64,
    device: Device,
}
impl Tacotron2 {
    pub(crate) fn load(
        store: &TensorStore,
        multi: bool,
        max_steps: usize,
        gate_threshold: f64,
        device: &Device,
    ) -> Result<Self> {
        let symbols = store.float::<2>("tacotron.embedding.weight", device)?;
        let speakers = if multi {
            Some(store.float::<2>("tacotron.speaker_embedding.weight", device)?)
        } else {
            None
        };
        let encoder = (0..3)
            .map(|i| {
                ConvNorm::load(
                    store,
                    &format!("tacotron.encoder.convolutions.{i}"),
                    512,
                    512,
                    device,
                )
            })
            .collect::<Result<_>>()?;
        let postnet = (0..5)
            .map(|i| {
                ConvNorm::load(
                    store,
                    &format!("tacotron.postnet.convolutions.{i}"),
                    if i == 0 { 128 } else { 512 },
                    if i == 4 { 128 } else { 512 },
                    device,
                )
            })
            .collect::<Result<_>>()?;
        let p = "tacotron.decoder";
        let lin = |n, input, output, biased| {
            Projection::load(
                store,
                &format!("{p}.{n}.linear_layer"),
                input,
                output,
                biased,
                device,
            )
        };
        Ok(Self {
            symbols,
            speakers,
            encoder,
            forward_lstm: LstmCell::load(store, "tacotron.encoder.lstm", "_l0", 512, 256, device)?,
            reverse_lstm: LstmCell::load(
                store,
                "tacotron.encoder.lstm",
                "_l0_reverse",
                512,
                256,
                device,
            )?,
            prenet: [
                lin("prenet.layers.0", 128, 256, false)?,
                lin("prenet.layers.1", 256, 256, false)?,
            ],
            attention_cell: LstmCell::load(
                store,
                &format!("{p}.attention_rnn"),
                "",
                768,
                1024,
                device,
            )?,
            attention: Attention::load(store, device)?,
            decoder_cell: LstmCell::load(
                store,
                &format!("{p}.decoder_rnn"),
                "",
                1536,
                1024,
                device,
            )?,
            mel: lin("linear_projection", 1536, 128, true)?,
            gate: lin("gate_layer", 1536, 1, true)?,
            postnet,
            max_steps,
            gate_threshold,
            device: device.clone(),
        })
    }
    pub(crate) fn encode(&self, tokens: &[i64], speaker: i64) -> Result<Tensor<2>> {
        let len = tokens.len();
        ensure!(
            (1..=512).contains(&len)
                && tokens
                    .iter()
                    .all(|&id| id >= 0 && (id as usize) < self.symbols.dims()[0]),
            "invalid Tacotron input tokens"
        );
        let ids =
            Tensor::<2, Int>::from_data(TensorData::new(tokens.to_vec(), [1, len]), &self.device);
        let mut x = embedding(self.symbols.clone(), ids);
        if let Some(speakers) = &self.speakers {
            ensure!(
                speaker >= 0 && (speaker as usize) < speakers.dims()[0],
                "invalid Tacotron speaker"
            );
            let id =
                Tensor::<2, Int>::from_data(TensorData::new(vec![speaker], [1, 1]), &self.device);
            x = Tensor::cat(
                vec![x, embedding(speakers.clone(), id).repeat_dim(1, len)],
                2,
            );
        }
        x = x.swap_dims(1, 2);
        for layer in &self.encoder {
            x = relu(layer.forward(x));
        }
        let x = x.reshape([512, len]).transpose();
        Ok(Tensor::cat(
            vec![
                self.forward_lstm.sequence(x.clone(), false, &self.device),
                self.reverse_lstm.sequence(x, true, &self.device),
            ],
            1,
        ))
    }
    pub(crate) fn infer(&self, tokens: &[i64], speaker: i64) -> Result<Tensor<3>> {
        self.infer_with_dropout(tokens, speaker, |x| {
            let mask = x.random_like(Distribution::Bernoulli(0.5));
            x * mask * 2.0
        })
    }
    fn infer_with_dropout(
        &self,
        tokens: &[i64],
        speaker: i64,
        mut dropout: impl FnMut(Tensor<2>) -> Tensor<2>,
    ) -> Result<Tensor<3>> {
        let memory = self.encode(tokens, speaker)?;
        let len = memory.dims()[0];
        let processed = self.attention.memory.forward(memory.clone());
        let zeros = |n| Tensor::<2>::zeros([1, n], &self.device);
        let (mut ah, mut ac, mut dh, mut dc) = (zeros(1024), zeros(1024), zeros(1024), zeros(1024));
        let (mut previous, mut cumulative, mut context, mut input) =
            (zeros(len), zeros(len), zeros(512), zeros(128));
        let mut frames = Vec::new();
        for _ in 0..self.max_steps {
            let x = dropout(relu(self.prenet[0].forward(input)));
            let x = dropout(relu(self.prenet[1].forward(x)));
            (ah, ac) = self
                .attention_cell
                .step(Tensor::cat(vec![x, context], 1), ah, ac);
            (context, previous) = self.attention.forward(
                ah.clone(),
                memory.clone(),
                processed.clone(),
                previous,
                cumulative.clone(),
            );
            cumulative = cumulative + previous.clone();
            (dh, dc) =
                self.decoder_cell
                    .step(Tensor::cat(vec![ah.clone(), context.clone()], 1), dh, dc);
            let out = Tensor::cat(vec![dh.clone(), context.clone()], 1);
            let gate: f32 = sigmoid(self.gate.forward(out.clone())).into_scalar();
            ensure!(gate.is_finite(), "non-finite Tacotron gate");
            if gate as f64 > self.gate_threshold {
                break;
            }
            input = self.mel.forward(out);
            frames.push(input.clone());
        }
        ensure!(
            !frames.is_empty(),
            "Tacotron stopped before producing audio"
        );
        let count = frames.len();
        let mel = Tensor::cat(frames, 0).transpose().reshape([1, 128, count]);
        let mut post = mel.clone();
        for (i, layer) in self.postnet.iter().enumerate() {
            post = layer.forward(post);
            if i + 1 < self.postnet.len() {
                post = post.tanh();
            }
        }
        Ok(mel + post)
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod tests {
    use super::*;
    use crate::english_vocoder::V2Vocoder;
    use serde::Deserialize;
    use silero_package::V2Checkpoint;
    use std::{fs, path::Path};

    #[derive(Deserialize)]
    struct Case {
        key: String,
        text: String,
        speaker: String,
        speaker_id: i64,
        ids: Vec<i64>,
        rates: Vec<u32>,
    }
    fn compare(label: &str, actual: Vec<f32>, expected: Vec<f32>, tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "{label} shape mismatch");
        assert!(actual.iter().all(|v| v.is_finite()));
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2))
            .sum::<f64>();
        let signal = expected.iter().map(|a| (*a as f64).powi(2)).sum::<f64>();
        println!(
            "{label} count={} max_abs={max:.9} snr={:.3}",
            actual.len(),
            10.0 * (signal / error).log10()
        );
        assert!(max <= tolerance, "{label} exceeds {tolerance}");
    }
    #[test]
    #[ignore = "requires original V2 checkpoints and captured dropout fixture"]
    fn original_v2_fixed_mask_parity() {
        let root = std::env::var("SILERO_V2_CHECKPOINTS").unwrap();
        let fixtures = std::env::var("SILERO_V2_FIXTURES").unwrap();
        let names = std::env::var("SILERO_V2_MODELS").unwrap_or_else(|_| "lj_v2,multi_v2,mykyta_v2".into());
        let device = Device::flex();
        for name in names.split(',') {
            let cp = V2Checkpoint::open(Path::new(&root).join(format!("{name}.pt"))).unwrap();
            let model_rates = cp.sample_rates.clone();
            let accent = cp.recurrent_accent.map(crate::recurrent_accent::RecurrentAccent::load).transpose().unwrap();
            let store = TensorStore::native(cp.weights);
            let acoustic =
                Tacotron2::load(&store, cp.multi, cp.max_steps, cp.gate_threshold, &device)
                    .unwrap();
            let vocoder = V2Vocoder::load(&store, &model_rates, &device).unwrap();
            let reference =
                TensorStore::open(&Path::new(&fixtures).join(format!("{name}.safetensors")))
                    .unwrap();
            let cases: Vec<Case> = serde_json::from_slice(
                &fs::read(Path::new(&fixtures).join(format!("{name}.json"))).unwrap(),
            )
            .unwrap();
            for case in cases {
                let k = &case.key;
                let label = format!("{name}/{k}");
                assert_eq!(cp.speakers[&case.speaker] as i64, case.speaker_id);
                assert_eq!(crate::v2::prepare_tokens(&cp.symbols, &cp.encoded_symbols, &cp.ascii_symbols, accent.as_ref(), &case.text).unwrap(), case.ids);
                compare(
                    &format!("{label}/memory"),
                    acoustic
                        .encode(&case.ids, case.speaker_id)
                        .unwrap()
                        .into_data()
                        .try_to_vec()
                        .unwrap(),
                    reference.float_data(&format!("{k}_memory")).unwrap(),
                    2e-4,
                );
                let mask = reference
                    .float::<2>(&format!("{k}_masks"), &device)
                    .unwrap();
                let mut cursor = 0;
                let mel = acoustic
                    .infer_with_dropout(&case.ids, case.speaker_id, |x| {
                        assert!(cursor < mask.dims()[0]);
                        let v = x * mask.clone().slice([cursor..cursor + 1, 0..256]);
                        cursor += 1;
                        v
                    })
                    .unwrap();
                assert_eq!(cursor, mask.dims()[0], "{label}: decoder gate mismatch");
                compare(
                    &format!("{label}/mel"),
                    mel.clone().into_data().try_to_vec().unwrap(),
                    reference.float_data(&format!("{k}_mel")).unwrap(),
                    3e-3,
                );
                for rate in case.rates {
                    let expected = reference.float_data(&format!("{k}_pcm{rate}")).unwrap();
                    let isolated = vocoder
                        .forward(
                            reference.float::<3>(&format!("{k}_mel"), &device).unwrap(),
                            rate,
                        )
                        .unwrap()
                        .into_data()
                        .try_to_vec()
                        .unwrap();
                    compare(
                        &format!("{label}/vocoder{rate}"),
                        isolated,
                        expected.clone(),
                        1e-3,
                    );
                    compare(
                        &format!("{label}/pcm{rate}"),
                        vocoder
                            .forward(mel.clone(), rate)
                            .unwrap()
                            .into_data()
                            .try_to_vec()
                            .unwrap(),
                        expected,
                        3e-3,
                    );
                }
            }
        }
    }
}
